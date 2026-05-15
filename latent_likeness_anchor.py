"""
LTX Likeness Anchor v1.2

Per-block attention residual for identity preservation that pulls generated
tokens toward a reference frame's features.

================================================================================
CHANGES IN v1.2
================================================================================
- New default pull_mode='directional': rotates token feature vectors toward
  reference identity direction WITHOUT changing their magnitude. Preserves
  per-token energy and therefore the frame's tonal/color statistics. Fixes
  the desaturation / color fade observed at strength >= 0.10 in v1.1.
- 'additive' mode retained for users who explicitly want the original
  v1.1 behavior (identity + tonal pull).
- New late_block_falloff parameter: ramps down strength in the last 12
  blocks (37-47, which handle fine refinement). Reduces the over-sharpening
  / texture-flattening that strong late-block pull causes. 0.3-0.6 typical
  if used.
- Updated default tuning guidance: strength can now safely go higher
  (0.12-0.18) with directional mode without producing color drift.

================================================================================
CHANGES IN v1.1
================================================================================
- Now reads reference geometry from REFERENCE_INFO input wired from
  LikenessGuide.
- Updated default depth_curve from configurable to 'flat' as the safe
  default. late_focus is documented as a CAUTION option due to its
  tendency to cause over-sharpening when used with strong pull.
- Tuning guidance added to tooltips and docstring.

================================================================================
ARCHITECTURE
================================================================================

LTXLikenessGuide appends a reference frame at the end of the latent, marks it
preserved in noise_mask (zero noise → stays clean across all sampling steps),
and emits a REFERENCE_INFO dict with frame layout and bbox.

This node reads that info at apply time, hooks attn1 on each block, and
during each forward:
  1. Reshapes the attention sequence into a (B, F_total, H, W, D) grid
  2. Splits into generated portion [:F_orig] and reference portion [F_orig:]
  3. For each generated frame, computes per-token cosine similarity to
     reference tokens (centered features in fp32 for numerical stability)
  4. Computes a target representation per generated token via softmax-weighted
     reference features
  5. Applies a pull: strength × sigmoid_weight × (target - current)
     where sigmoid_weight gates pull to tokens above similarity_threshold

The reference frame is ALWAYS in the attention sequence during sampling
because LikenessGuide put it in the latent. The model's natural self-
attention already cross-talks between generated and reference. This hook
amplifies that pull toward identity-matching features.

================================================================================
COMBINES BEST OF FACE_ANCHOR AND LATENT_ANCHOR_AWARE
================================================================================

From face_anchor (v4.0):
  - Per-block attn1 hook architecture
  - Centered cosine similarity for identity matching
  - depth_curve per-block strength scaling
  - block_index_filter for ablation
  - bbox-aware reference token selection

From latent_anchor_aware (v2.3):
  - Sigmoid-based threshold gating (avoids anatomy distortion that linear
    interpolation produced — tokens are clearly in or out of the pull set)
  - Health diagnostics
  - Sigma-aware activation (can skip when sigma is too high)
  - Simple/advanced mode UI

New (vs both):
  - Reference is EXTERNAL (the guide's preserved frame), not derived from
    the generation itself. Stable identity target that doesn't drift.
  - Correspondence search via softmax over reference tokens — finds face
    tokens in generated frames automatically, no per-frame bbox needed
  - Reads REFERENCE_INFO from LikenessGuide directly — explicit data flow

================================================================================
USAGE
================================================================================

Workflow:
    LoadModel
        ↓
    LikenessGuide(positive, negative, vae, latent, ref_image,
                  face_bbox_within_reference="0.30,0.20,0.70,0.65")
        ↓ positive, negative, latent, reference_info
    LikenessAnchor(model, reference_info, strength=0.10)
        ↓ model
    KSampler(model, positive, negative, latent, ...)
        ↓ latent
    LikenessCrop(latent, reference_info)
        ↓ latent
    VAE Decode

================================================================================
TUNING GUIDANCE
================================================================================

Default recommended:
    strength             = 0.10
    similarity_threshold = 0.50
    depth_curve          = flat
    decay_with_distance  = 0.0

Symptom: over-sharpening / crispness / "still image look"
Cause: pulling strongly on late blocks substitutes still-image fine detail
       into video refinement
Fix:   - use depth_curve = flat or ramp_down (never late_focus at strength > 0.10)
       - reduce strength to 0.06-0.08
       - block_index_filter = "0-36" to skip last 12 blocks
       - similarity_sharpness = 5-6 (softer transitions)

Symptom: weak identity preservation
Fix:   - raise strength to 0.12-0.15 with depth_curve = flat
       - lower similarity_threshold to 0.40-0.45 (pulls more tokens)
       - ensure face_bbox_within_reference is set on LikenessGuide (focuses
         pull source on face features)

Symptom: face is too rigid / loses expression / "frozen face"
Fix:   - lower strength to 0.05-0.07
       - raise similarity_threshold to 0.55-0.60 (pull narrower set)
       - decay_with_distance = 0.3 (less pull in later frames)

================================================================================
MEMORY NOTE
================================================================================

The similarity tensor shape is (B, F_orig, H*W, R) where R is the count of
reference tokens. Setting face_bbox_within_reference in LikenessGuide
substantially reduces R and memory footprint — recommended.
"""

import math
import torch
import torch.nn.functional as F


# Metadata key set by LikenessGuide
METADATA_KEY = "_10s_likeness_reference"

# Sentinels for hook coexistence with other 10S nodes
HOOK_ATTR_ATTN1 = "_10s_likeness_anchor_attn1_hook"
HOOK_ATTR_BACKBONE = "_10s_likeness_anchor_backbone_hook"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_attn_tensor(output):
    """Handle both plain Tensor and tuple/wrapped attention outputs."""
    if isinstance(output, torch.Tensor):
        return output, lambda t: t
    if isinstance(output, tuple) and len(output) >= 1:
        return output[0], lambda t: (t,) + output[1:]
    return output, lambda t: t


def _parse_block_filter(s):
    """Parse '10-30' or '5,7,15-20' into a set of block indices."""
    if not s or not s.strip():
        return None
    result = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            try:
                a, b = part.split("-")
                result.update(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                result.add(int(part))
            except ValueError:
                continue
    return result if result else None


def _depth_factor(block_idx, total_blocks, curve):
    """Compute per-block depth scaling for strength."""
    p = block_idx / max(1, total_blocks - 1)
    if curve == "flat":
        return 1.0
    if curve == "ramp_up":
        return p
    if curve == "ramp_down":
        return 1.0 - p
    if curve == "late_focus":
        return p * p
    if curve == "middle":
        return 1.0 - abs(2 * p - 1)
    return 1.0


def _parse_bbox_to_indices(bbox_str, H, W):
    """Parse 'x1,y1,x2,y2' (normalized 0-1) into integer index ranges
    (h1, h2, w1, w2). Returns full range on parse failure."""
    if not bbox_str or not bbox_str.strip():
        return 0, H, 0, W
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            return 0, H, 0, W
        x1, y1, x2, y2 = parts
        h1 = max(0, int(y1 * H))
        h2 = min(H, max(h1 + 1, int(math.ceil(y2 * H))))
        w1 = max(0, int(x1 * W))
        w2 = min(W, max(w1 + 1, int(math.ceil(x2 * W))))
        if h2 <= h1 or w2 <= w1:
            return 0, H, 0, W
        return h1, h2, w1, w2
    except (ValueError, TypeError):
        return 0, H, 0, W


def _find_backbone(model):
    """Locate the transformer module that holds .transformer_blocks."""
    # Try common paths first
    for path in ("diffusion_model",
                 "model.diffusion_model",
                 "model.model",
                 "diffusion_model.model"):
        obj = model.model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "transformer_blocks"):
                return obj
        except AttributeError:
            continue
    # Fallback: direct .model with transformer_blocks
    try:
        obj = model.model
        if hasattr(obj, "transformer_blocks"):
            return obj
    except AttributeError:
        pass
    # Fallback: model itself
    if hasattr(model, "transformer_blocks"):
        return model
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTXLikenessAnchor:
    """
    Per-block attention residual for likeness preservation. Reads reference
    frame location from LikenessGuide metadata in conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.80, "step": 0.01,
                    "tooltip": "Magnitude of pull toward reference features. "
                               "0.08-0.18 typical with directional pull. "
                               "Higher values (0.30-0.60) usable for "
                               "stronger identity lock. Variance-collapse "
                               "side effects at very high values are "
                               "possible but practical ceiling unconfirmed "
                               "— test with clean state (no prior runs' "
                               "hooks leaked) before drawing conclusions.",
                }),
            },
            "optional": {
                "reference_info": ("REFERENCE_INFO", {
                    "tooltip": "Optional: wire from LikenessGuide if using "
                               "Guide+Anchor combined approach. Leave "
                               "disconnected to use frame_0 of the latent as "
                               "the reference (recommended for i2v workflows "
                               "where frame_0 is already the conditioning).",
                }),
                "reference_source": (
                    ["auto", "guide", "latent_frame_0"],
                    {
                        "default": "auto",
                        "tooltip": "auto: use guide if reference_info wired, "
                                   "else use latent_frame_0. "
                                   "guide: requires reference_info from "
                                   "LikenessGuide. Reads the appended "
                                   "reference frame. "
                                   "latent_frame_0: read the first frame of "
                                   "the latent as the reference. Avoids the "
                                   "end-keyframe interpolation issue; ideal "
                                   "for i2v workflows where frame_0 is the "
                                   "conditioned start image.",
                    },
                ),
                "frame_0_bbox": ("STRING", {
                    "default": "",
                    "tooltip": "When reference_source=latent_frame_0: bbox "
                               "within frame_0 to use as identity source. "
                               "Format: 'x1,y1,x2,y2' normalized 0-1. "
                               "Empty = whole frame_0 used as reference.",
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum centered-cosine similarity for a "
                               "generated token to receive pull. Lower = "
                               "broader effect (more tokens pulled).",
                }),
                "decay_with_distance": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Per-frame strength decay from frame 0 to "
                               "frame F_orig-1. 0 = uniform across all frames.",
                }),
                "bypass": ("BOOLEAN", {"default": False}),
                "debug":  ("BOOLEAN", {"default": False}),
                "advanced_mode": ("BOOLEAN", {"default": False}),
                "depth_curve": (
                    ["flat", "ramp_up", "ramp_down", "late_focus", "middle"],
                    {"default": "flat",
                     "tooltip": "Per-block strength scaling. flat = uniform "
                                "(recommended). ramp_down = stronger early, "
                                "lighter late (good for avoiding over-sharpening). "
                                "late_focus = stronger late blocks (CAUTION: "
                                "can over-sharpen by pulling still-image detail "
                                "into video refinement steps)."},
                ),
                "block_index_filter": ("STRING", {
                    "default": "",
                    "tooltip": "Limit to specific blocks, e.g. '0-36' to skip "
                               "the final 12 blocks where over-sharpening "
                               "tends to manifest. Empty = all blocks.",
                }),
                "similarity_sharpness": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Sigmoid steepness for similarity gating. "
                               "Higher = more binary in/out. Lower values "
                               "(4-6) soften the transition.",
                }),
                "override_face_bbox": ("STRING", {
                    "default": "",
                    "tooltip": "Override face_bbox_within_reference set in "
                               "LikenessGuide. Format: 'x1,y1,x2,y2' "
                               "normalized 0-1. Empty = use guide's bbox.",
                }),
                "skip_when_sigma_above": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Disable anchor when sigma > this value "
                               "(high-noise early steps). 0 = always active.",
                }),
                "pull_mode": (
                    ["directional", "additive"],
                    {
                        "default": "directional",
                        "tooltip": "directional (recommended): rotate token "
                                   "features toward reference identity "
                                   "direction WITHOUT changing magnitude. "
                                   "Preserves color/saturation. "
                                   "additive (legacy): pull toward reference "
                                   "features including magnitude. Identity + "
                                   "tonal pull; can cause desaturation/fade "
                                   "with strong settings.",
                    },
                ),
                "late_block_falloff": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Additional per-block strength falloff in the "
                               "last 12 blocks (37-47), which handle fine "
                               "detail. 0 = no falloff. 0.5 = late blocks at "
                               "half strength. 1.0 = late blocks disabled. "
                               "Use 0.3-0.6 to reduce over-sharpening or "
                               "texture-flattening from late-block pull.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Per-block attention pull toward reference features for identity "
        "preservation. Two modes: (1) wire reference_info from LikenessGuide "
        "to use Guide's appended reference frame. (2) leave reference_info "
        "unconnected and the node uses frame_0 of the latent as reference — "
        "ideal for i2v workflows where frame_0 is the conditioning start "
        "image. v1.2 uses directional pull mode (preserves color)."
    )

    def apply(self, model, strength,
              reference_info=None,
              reference_source="auto",
              frame_0_bbox="",
              similarity_threshold=0.50, decay_with_distance=0.0,
              bypass=False, debug=False, advanced_mode=False,
              depth_curve="flat", block_index_filter="",
              similarity_sharpness=8.0, override_face_bbox="",
              skip_when_sigma_above=0.0,
              pull_mode="directional",
              late_block_falloff=0.0):

        # ─── Resolve advanced mode defaults ──────────────────────────────────
        if not advanced_mode:
            depth_curve = "flat"
            block_index_filter = ""
            similarity_sharpness = 8.0
            override_face_bbox = ""
            skip_when_sigma_above = 0.0

        if bypass:
            # Clean up any prior LikenessAnchor hooks on the shared modules.
            # model.clone() shares transformer_blocks, so hooks from previous
            # runs remain installed unless we explicitly remove them.
            backbone = _find_backbone(model)
            if backbone is not None:
                blocks_cleanup = backbone.transformer_blocks
                n_removed = 0
                for block in blocks_cleanup:
                    attn1 = getattr(block, "attn1", None)
                    if attn1 is None:
                        continue
                    # Remove any prior registered hook handle stored on the module
                    handle = getattr(attn1, "_10s_likeness_anchor_handle", None)
                    if handle is not None:
                        try:
                            handle.remove()
                        except Exception:
                            pass
                        try:
                            delattr(attn1, "_10s_likeness_anchor_handle")
                        except AttributeError:
                            pass
                    # Clear sentinel
                    if getattr(attn1, HOOK_ATTR_ATTN1, False):
                        try:
                            delattr(attn1, HOOK_ATTR_ATTN1)
                            n_removed += 1
                        except AttributeError:
                            pass
                # Also clear backbone pre-hook sentinel so a future apply can
                # cleanly re-register
                if getattr(backbone, HOOK_ATTR_BACKBONE, False):
                    try:
                        delattr(backbone, HOOK_ATTR_BACKBONE)
                    except AttributeError:
                        pass
                if debug:
                    print(f"\u2192 [10S] LikenessAnchor: bypass=True, "
                          f"removed {n_removed} prior hook(s)")
            else:
                if debug:
                    print("\u2192 [10S] LikenessAnchor: bypass=True, no-op")
            return (model,)

        # ─── Resolve reference source ────────────────────────────────────────
        # auto: use guide if reference_info wired, else frame_0
        has_guide = isinstance(reference_info, dict) and \
            reference_info.get("extended_latent_length") is not None
        if reference_source == "auto":
            effective_source = "guide" if has_guide else "latent_frame_0"
        else:
            effective_source = reference_source

        if effective_source == "guide" and not has_guide:
            print("\u2192 [10S] LikenessAnchor: \u26a0 reference_source=guide "
                  "requested but no valid reference_info wired. Falling back "
                  "to latent_frame_0.")
            effective_source = "latent_frame_0"

        # ─── Metadata path: differs by source mode ───────────────────────────
        if effective_source == "guide":
            metadata = reference_info
            F_orig = metadata.get("original_latent_length")
            F_total = metadata.get("extended_latent_length")
            F_ref = metadata.get("frame_count_latent", 1)
            face_bbox_meta = metadata.get("face_bbox", "")
            spatial_dims = metadata.get("spatial_dims_latent", [0, 0])
            H_latent, W_latent = spatial_dims[0], spatial_dims[1]

            if F_orig is None or F_total is None or H_latent == 0 or W_latent == 0:
                print("\u2192 [10S] LikenessAnchor: \u26a0 metadata incomplete; "
                      "returning unmodified model")
                return (model,)

            face_bbox = override_face_bbox if override_face_bbox else face_bbox_meta
        else:
            # latent_frame_0 mode: dims discovered at runtime from actual latent shape
            # F_orig, F_total, H_latent, W_latent are set inside the hook
            # based on the inferred sequence shape. No metadata required.
            F_orig = None
            F_total = None
            F_ref = 1  # we treat frame 0 as the single reference frame
            face_bbox_meta = frame_0_bbox
            H_latent = 0  # discovered at runtime
            W_latent = 0
            face_bbox = override_face_bbox if override_face_bbox else face_bbox_meta

        if debug:
            print(f"\u2192 [10S] LikenessAnchor v1.2:")
            print(f"  \u00b7 reference_source={effective_source}")
            if effective_source == "guide":
                print(f"  \u00b7 F_orig={F_orig} F_total={F_total} F_ref={F_ref}")
                print(f"  \u00b7 latent dims H={H_latent} W={W_latent}")
            else:
                print(f"  \u00b7 frame_0 mode: dims discovered at runtime")
            print(f"  \u00b7 face_bbox='{face_bbox or '<none — full reference>'}'")
            print(f"  \u00b7 strength={strength} sim_thr={similarity_threshold} "
                  f"sim_sharp={similarity_sharpness}")
            print(f"  \u00b7 pull_mode={pull_mode} late_block_falloff={late_block_falloff}")
            print(f"  \u00b7 depth_curve={depth_curve} block_filter="
                  f"'{block_index_filter or '<all>'}'")
            if skip_when_sigma_above > 0:
                print(f"  \u00b7 skip_when_sigma_above={skip_when_sigma_above}")

        # ─── Find backbone ───────────────────────────────────────────────────
        backbone = _find_backbone(model)
        if backbone is None:
            print("\u2192 [10S] LikenessAnchor: \u26a0 couldn't locate "
                  "transformer_blocks; returning unmodified model")
            return (model,)

        blocks = backbone.transformer_blocks
        total_blocks = len(blocks)
        block_filter = _parse_block_filter(block_index_filter)

        if debug:
            n_filt = (len([i for i in range(total_blocks)
                           if block_filter is None or i in block_filter]))
            print(f"  \u00b7 total_blocks={total_blocks} active={n_filt}")

        # ─── Clone model for patching ────────────────────────────────────────
        m = model.clone()

        # Shared state across hooks
        state = {
            "current_sigma": None,
            "call_count": 0,
            "skipped_count": 0,
            "captured_latent_shape": None,  # for frame_0 mode
        }

        # ─── Backbone pre-hook: capture sigma and latent shape ──────────────
        def backbone_pre_hook(module, args, kwargs):
            to = kwargs.get("transformer_options", {})
            sigmas_in = to.get("sigmas")
            if sigmas_in is not None:
                try:
                    if hasattr(sigmas_in, "max"):
                        state["current_sigma"] = float(sigmas_in.max().item())
                    else:
                        state["current_sigma"] = float(sigmas_in)
                except Exception:
                    pass
            # Capture the latent shape from the first 5D tensor in args/kwargs
            # — frame_0 mode needs this to find dims at runtime
            if effective_source == "latent_frame_0":
                try:
                    candidates = []
                    for a in args:
                        if isinstance(a, torch.Tensor) and a.dim() == 5:
                            candidates.append(a)
                    for v in kwargs.values():
                        if isinstance(v, torch.Tensor) and v.dim() == 5:
                            candidates.append(v)
                    if candidates:
                        # Use the first 5D tensor — typically the latent input
                        state["captured_latent_shape"] = tuple(candidates[0].shape)
                except Exception:
                    pass

        if not getattr(backbone, HOOK_ATTR_BACKBONE, False):
            backbone.register_forward_pre_hook(backbone_pre_hook, with_kwargs=True)
            setattr(backbone, HOOK_ATTR_BACKBONE, True)

        # ─── attn1 hook factory ──────────────────────────────────────────────
        def make_attn1_hook(block_idx):
            depth_scale = _depth_factor(block_idx, total_blocks, depth_curve)
            # Late-block falloff: reduce strength on the last 12 blocks
            # (37-47, which handle fine refinement). late_block_falloff in
            # [0, 1] = fraction of strength removed from late blocks.
            if late_block_falloff > 0 and block_idx >= total_blocks - 12:
                late_progress = (block_idx - (total_blocks - 12)) / 11.0
                late_progress = max(0.0, min(1.0, late_progress))
                # Smoothly ramp the falloff: minimum reduction at block 37,
                # full reduction at block 47.
                falloff = late_block_falloff * late_progress
                depth_scale = depth_scale * (1.0 - falloff)

            def hook(module, inputs, output):
                if bypass:
                    return output

                # ── Sigma gating ─────────────────────────────────────────────
                if skip_when_sigma_above > 0:
                    cur_sigma = state["current_sigma"]
                    if cur_sigma is not None and cur_sigma > skip_when_sigma_above:
                        state["skipped_count"] += 1
                        return output

                tensor, wrap = _extract_attn_tensor(output)
                if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
                    return output

                B, seq, D = tensor.shape

                # ── Discover or use known latent dims ────────────────────────
                if effective_source == "guide":
                    # Use known H_latent, W_latent from metadata; infer F
                    spatial_dim = H_latent * W_latent
                    if spatial_dim <= 0 or seq % spatial_dim != 0:
                        return output
                    F_actual = seq // spatial_dim
                    if F_actual < F_orig + 1:
                        return output
                    H_use, W_use = H_latent, W_latent
                    # Generated = first F_orig frames; reference = remainder
                    F_gen_start, F_gen_end = 0, F_orig
                    F_ref_start, F_ref_end = F_orig, F_actual
                else:
                    # latent_frame_0 mode: discover dims from sequence shape
                    # First try the captured latent shape from backbone pre-hook
                    captured_shape = state.get("captured_latent_shape")
                    if captured_shape is not None:
                        _, _, F_cap, H_cap, W_cap = captured_shape
                        if F_cap * H_cap * W_cap == seq:
                            F_actual, H_use, W_use = F_cap, H_cap, W_cap
                        else:
                            # Captured shape doesn't match — skip
                            return output
                    else:
                        # No captured shape; cannot determine dims
                        return output
                    if F_actual < 2:
                        return output  # need at least 1 ref + 1 gen frame
                    # Frame 0 = reference; frames 1..F_actual-1 = generated
                    F_gen_start, F_gen_end = 1, F_actual
                    F_ref_start, F_ref_end = 0, 1

                try:
                    grid = tensor.view(B, F_actual, H_use, W_use, D)
                except RuntimeError:
                    return output

                F_gen_count = F_gen_end - F_gen_start
                gen_grid = grid[:, F_gen_start:F_gen_end]
                ref_grid = grid[:, F_ref_start:F_ref_end]

                # Apply bbox within reference
                h1, h2, w1, w2 = _parse_bbox_to_indices(
                    face_bbox, H_use, W_use
                )
                ref_sub = ref_grid[:, :, h1:h2, w1:w2, :]
                R_count = ref_sub.shape[1] * ref_sub.shape[2] * ref_sub.shape[3]
                if R_count == 0:
                    return output
                ref_tokens = ref_sub.reshape(B, R_count, D)

                # ── Centered features for cosine similarity ──────────────────
                work_dtype = torch.float32

                gen_flat = gen_grid.reshape(
                    B, F_gen_count, H_use * W_use, D
                ).to(work_dtype)
                gen_mean = gen_flat.mean(dim=2, keepdim=True)
                gen_centered = gen_flat - gen_mean

                ref_centered_full = ref_tokens.to(work_dtype)
                ref_mean = ref_centered_full.mean(dim=1, keepdim=True)
                ref_centered = ref_centered_full - ref_mean

                # Normalize
                gen_norm = F.normalize(gen_centered, dim=-1, eps=1e-6)
                ref_norm = F.normalize(ref_centered, dim=-1, eps=1e-6)

                # Per-generated-token similarity to each reference token:
                # gen_norm: (B, F_orig, T_g, D), ref_norm: (B, R, D)
                # sim:     (B, F_orig, T_g, R)
                sim = torch.einsum("bfid,brd->bfir", gen_norm, ref_norm)

                # Gate weight: only tokens above threshold get pull
                max_sim, _ = sim.max(dim=-1)  # (B, F_orig, T_g)
                weight = torch.sigmoid(
                    (max_sim - similarity_threshold) * similarity_sharpness
                )

                # Target direction: softmax-weighted reference token directions
                # (using already-centered features). This gives the identity
                # direction in feature space for each generated token.
                sim_weights = F.softmax(
                    sim * similarity_sharpness, dim=-1
                )  # (B, F_orig, T_g, R)
                target_centered = torch.einsum(
                    "bfir,brd->bfid", sim_weights, ref_centered
                )  # (B, F_orig, T_g, D)
                target_norm = F.normalize(target_centered, dim=-1, eps=1e-6)

                # ── Pull mode: directional vs additive ──────────────────────
                # Directional (default): rotate the generated token's
                # centered feature vector toward the target direction WITHOUT
                # changing its magnitude. This preserves token-level energy
                # (and therefore the frame's tonal statistics) — pulls
                # identity but not color/saturation.
                # Additive (legacy): pull toward target features including
                # their magnitude. Identity + tonal pull. Causes
                # desaturation/fade with strong settings across many frames.
                if pull_mode == "directional":
                    # Project gen_centered onto plane perpendicular to its own
                    # current direction, toward target direction. The result
                    # is a rotation that preserves the token's magnitude.
                    gen_magnitude = gen_centered.norm(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-6)
                    # Blend current direction with target direction by weight*strength,
                    # then renormalize and rescale to original magnitude.
                    cur_strength = strength * depth_scale
                    blend_amount = cur_strength * weight.unsqueeze(-1)
                    blend_amount = blend_amount.clamp(0.0, 1.0)
                    blended_dir = gen_norm * (1.0 - blend_amount) + \
                                  target_norm * blend_amount
                    blended_dir_renorm = F.normalize(
                        blended_dir, dim=-1, eps=1e-6
                    )
                    # Reconstruct centered feature with original magnitude
                    new_gen_centered = blended_dir_renorm * gen_magnitude
                    # Pull is the difference from current centered features
                    pull = new_gen_centered - gen_centered
                else:
                    # Additive: target_centered + gen_mean, then difference
                    targets = target_centered + gen_mean
                    cur_strength = strength * depth_scale
                    pull = cur_strength * weight.unsqueeze(-1) * (targets - gen_flat)

                # Per-frame decay (across generated frames in order)
                if decay_with_distance > 0 and F_gen_count > 1:
                    frame_idx = torch.arange(
                        F_gen_count, dtype=work_dtype, device=tensor.device
                    )
                    frame_weights = 1.0 - decay_with_distance * (
                        frame_idx / (F_gen_count - 1)
                    )
                    pull = pull * frame_weights.view(1, F_gen_count, 1, 1)

                # Cast pull back to tensor's dtype and apply to gen portion
                pull = pull.to(dtype=tensor.dtype)
                new_gen_flat = gen_grid.reshape(
                    B, F_gen_count, H_use * W_use, D
                ) + pull
                new_gen_grid = new_gen_flat.reshape(
                    B, F_gen_count, H_use, W_use, D
                )

                # Reassemble: modified generated frames + unchanged reference
                new_grid = grid.clone()
                new_grid[:, F_gen_start:F_gen_end] = new_gen_grid
                # Reference portion stays at grid[:, F_ref_start:F_ref_end]
                new_tensor = new_grid.reshape(B, seq, D)

                state["call_count"] += 1
                if debug and state["call_count"] % 96 == 0:
                    cur_sigma = state["current_sigma"]
                    sigma_str = (f"{cur_sigma:.3f}" if cur_sigma is not None
                                 else "n/a")
                    avg_sim = max_sim.mean().item()
                    avg_weight = weight.mean().item()
                    n_active = (weight > 0.5).sum().item()
                    n_total = weight.numel()
                    pct_active = 100.0 * n_active / max(1, n_total)
                    print(f"  \u00b7 [blk {block_idx:02d}] sigma={sigma_str} "
                          f"avg_sim={avg_sim:.3f} avg_w={avg_weight:.3f} "
                          f"pulled={pct_active:.1f}% of tokens")

                return wrap(new_tensor)

            return hook

        # ─── Register hooks ──────────────────────────────────────────────────
        # Clear stale LikenessAnchor sentinels before registering. model.clone()
        # shares the underlying transformer_blocks ModuleList by reference, so
        # if this node ran previously (in this workflow or even a prior
        # ComfyUI session that didn't unload the model), our sentinels persist
        # on the same module objects. Without this cleanup, the registration
        # loop sees "already hooked" and skips every block, registering 0.
        cleared = 0
        for block in blocks:
            attn1 = getattr(block, "attn1", None)
            if attn1 is None:
                continue
            # Remove prior handle if present
            handle = getattr(attn1, "_10s_likeness_anchor_handle", None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
                try:
                    delattr(attn1, "_10s_likeness_anchor_handle")
                except AttributeError:
                    pass
            # Clear sentinel
            if getattr(attn1, HOOK_ATTR_ATTN1, False):
                try:
                    delattr(attn1, HOOK_ATTR_ATTN1)
                    cleared += 1
                except AttributeError:
                    pass

        if debug and cleared > 0:
            print(f"  \u00b7 cleared {cleared} stale sentinel(s) from prior runs")

        registered = 0
        for i, block in enumerate(blocks):
            if block_filter is not None and i not in block_filter:
                continue
            attn1 = getattr(block, "attn1", None)
            if attn1 is None:
                continue
            if getattr(attn1, HOOK_ATTR_ATTN1, False):
                continue
            handle = attn1.register_forward_hook(make_attn1_hook(i))
            setattr(attn1, HOOK_ATTR_ATTN1, True)
            setattr(attn1, "_10s_likeness_anchor_handle", handle)
            registered += 1

        if debug:
            print(f"  \u00b7 registered hooks on {registered} blocks")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXLikenessAnchor": LTXLikenessAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLikenessAnchor": "\U0001f3ad LTX Likeness Anchor",
}
