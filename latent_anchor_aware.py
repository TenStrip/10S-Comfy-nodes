"""
LTX Latent Anchor Aware v2.1 — spatial energy modulation.

================================================================================
ARCHITECTURAL CHANGE FROM v2.0
================================================================================
v2.0 attempted to use an external reference image as both the matching
target AND the energy source. The matching half failed due to feature-
space mismatch: post-patchify features don't match deep-block (post
attn1) features that v1.x's matching mechanism is designed for. Result:
sim scores 6-8x lower than v1.x, residuals at noise floor.

v2.1 demotes the reference's role to spatial-energy-only:

  - Matching mechanism is identical to LTXLatentAnchor v1.4. The anchor
    target is extracted from the RUNNING latent's anchor frame at the
    cache point, which produces correct deep-block features in the right
    feature space. Cosine similarity discriminates cleanly.

  - The reference image (or alternative LATENT input) is used to compute
    a spatial energy map: for each (h, w) position, how distinctive is
    the content there? High-energy regions (subjects, edges, salient
    content) get more anchor pull. Low-energy regions (uniform sky,
    plain backgrounds) get less.

  - Energy map is computed in VAE space (128-channel post-VAE), then
    bilinear-resampled to match the running latent's spatial dims.

This adds genuine content awareness without requiring parallel forward
passes, text/audio context borrowing, or feature-space-matching
compromises. The cost is a smaller role for the reference: it informs
WHERE intervention happens, not WHAT it pulls toward. But the spatial
information is real and the matching feature space is correct.

================================================================================
INPUTS
================================================================================
  Required:
    model            : MODEL.

  Optional energy sources (use one):
    reference_image  : IMAGE. VAE-encoded internally. Best for explicit
                       face / composition / style references separate from
                       the I2V conditioning frame.
    vae              : VAE. Required only if reference_image is connected.
    energy_latent    : LATENT. Alternative direct VAE-space input. Useful
                       for piping in a workflow's existing conditioned
                       latent without re-encoding.

  If neither energy source is connected, node behaves identically to
  LTXLatentAnchor v1.4 (uniform per-token strength, no spatial weighting).

  Schedule + standard parameters:
    sigmas, strength, cache_at_step, similarity_threshold,
    decay_with_distance, energy_threshold, bypass, debug.

  Advanced mode toggle exposes: cache_mode, forwards_per_step,
  cache_warmup, anchor_frame, depth_curve, block_index_filter.

================================================================================
ENERGY MODULATION SEMANTICS (v2.2)
================================================================================
  energy_threshold = 0.0  : uniform mask, no spatial weighting (v1.4 behaviour)
  energy_threshold = 0.30 : pull tokens with above-lower-third energy (default)
  energy_threshold = 0.50 : pull tokens with above-median energy
  energy_threshold = 0.80 : pull only top ~20% of energy regions

  Math:
    if energy_threshold > 0:
      factor = sigmoid((energy_norm - energy_threshold) * 16.0)
    else:
      factor = 1.0  (no gating)

  Steep sigmoid (sharpness 16) gives narrow transition zone — most tokens
  are clearly in-or-out of the pull set. This avoids the gradient-mask
  distortion problem seen with v2.1's linear interpolation, where
  intermediate values left every token under partial pull at varying
  strengths within coherent objects (causing anatomy artifacts).

================================================================================
WHAT CHANGED IN v2.2 (over v2.1)
================================================================================
- Replaced energy_modulation (linear interp) with energy_threshold
  (sigmoid threshold). See energy semantics above for math.
- Fixed bug: debug flag was being silently overridden to False when
  advanced_mode=False. Debug now works in both simple and advanced modes.

================================================================================
COMPATIBILITY WITH OTHER NODES
================================================================================
Coexists with LTXFaceAttentionAnchor and LTXLatentAnchor — different hook
sentinels mean all three can register simultaneously. Sensible chaining:
  Model → Latent Anchor → Latent Anchor Aware → Face Anchor → KSampler
gives broad scene stabilisation, content-aware modulation, and face-
targeted identity correction in sequence.
"""

import torch
import torch.nn.functional as F


# ─── Hardcoded constants (LTX2) ──────────────────────────────────────────────
TRACK_SHARPNESS = 8.0
SPATIAL_PATCH   = 1
TEMPORAL_PATCH  = 1
LTX_VAE_CHANNELS = 128


HOOK_ATTR_BACKBONE = "_10s_aware_anchor_pre_hook"
HOOK_ATTR_ATTN1    = "_10s_aware_anchor_attn1_hook"


# ─────────────────────────────────────────────────────────────────────────────
# Reference encoding + energy extraction
# ─────────────────────────────────────────────────────────────────────────────

def _vae_encode_reference(vae, image, debug=False):
    """
    VAE-encode an IMAGE input to LTX VAE-space latent.
    Tries multiple ComfyUI input layout conventions; returns the first
    that produces a valid latent (B, 128, F, H, W) or None on total failure.
    """
    if not hasattr(vae, "encode"):
        if debug:
            print(f"  \u00b7 [ref_encode] VAE has no .encode method.")
        return None

    if debug:
        print(f"  \u00b7 [ref_encode] VAE wrapper type: {type(vae).__name__}")

    img = image
    if img.dtype in (torch.uint8, torch.int8, torch.int32):
        img = img.float() / 255.0
    if img.dim() != 4:
        if debug:
            print(f"  \u00b7 [ref_encode] expected 4D image, got dim={img.dim()}")
        return None

    B, H, W, C = img.shape
    if C not in (3, 4):
        if debug:
            print(f"  \u00b7 [ref_encode] unexpected channel count {C}")
        return None
    if C == 4:
        img = img[..., :3]

    attempts = [
        ("BHWC_4d_image",    lambda x: x),                                  # ComfyUI image VAE convention
        ("BFHWC_5d_native",  lambda x: x.unsqueeze(1)),                     # video VAE 5D
        ("BCHW_4d_chfirst",  lambda x: x.permute(0, 3, 1, 2)),              # diffusers convention
        ("BCFHW_5d_chfirst", lambda x: x.permute(0, 3, 1, 2).unsqueeze(2)), # video diffusers
    ]

    last_error = None
    for label, prep in attempts:
        try:
            prepared = prep(img)
            latent = vae.encode(prepared)
            if isinstance(latent, dict):
                latent = latent.get("samples", latent.get("latent", latent))
            if not torch.is_tensor(latent):
                last_error = f"{label} returned {type(latent).__name__}"
                continue
            if debug:
                print(f"  \u00b7 [ref_encode] {label} \u2192 latent shape={tuple(latent.shape)}")

            # Normalise to (B, C, F, H, W)
            if latent.dim() == 5:
                if latent.shape[1] == LTX_VAE_CHANNELS:
                    return latent
                if latent.shape[-1] == LTX_VAE_CHANNELS:
                    return latent.permute(0, 4, 1, 2, 3).contiguous()
                return latent  # unknown layout, return as-is
            if latent.dim() == 4:
                if latent.shape[1] == LTX_VAE_CHANNELS:
                    return latent.unsqueeze(2)
                if latent.shape[-1] == LTX_VAE_CHANNELS:
                    return latent.permute(0, 3, 1, 2).unsqueeze(2)
                return latent.unsqueeze(2)

        except Exception as e:
            last_error = f"{label}: {type(e).__name__}: {e}"
            if debug:
                print(f"  \u00b7 [ref_encode] {label} raised: {type(e).__name__}: {e}")
            continue

    if debug:
        print(f"  \u00b7 [ref_encode] all layouts failed. Last error: {last_error}")
    return None


def _extract_latent_from_input(latent_input, debug=False):
    """
    Normalise a LATENT input to a 5D (B, C, F, H, W) tensor.
    ComfyUI's LATENT type is typically a dict {"samples": tensor, ...}.
    """
    if latent_input is None:
        return None
    samples = latent_input
    if isinstance(latent_input, dict):
        samples = latent_input.get("samples", latent_input.get("latent"))
    if not torch.is_tensor(samples):
        if debug:
            print(f"  \u00b7 [latent_in] no usable tensor in input "
                  f"({type(latent_input).__name__})")
        return None
    if samples.dim() == 4:
        samples = samples.unsqueeze(2)  # add frame dim
    if samples.dim() != 5:
        if debug:
            print(f"  \u00b7 [latent_in] unexpected dim {samples.dim()}, "
                  f"shape={tuple(samples.shape)}")
        return None
    if debug:
        print(f"  \u00b7 [latent_in] using direct latent shape={tuple(samples.shape)}")
    return samples


def _extract_energy_map(latent_5d, anchor_frame=0, debug=False):
    """
    Per-position structural energy from a VAE-space latent.
    Input:  (B, C, F, H, W) — typically C=128 for LTX VAE
    Output: (B, H, W) normalised to [0, 1]

    Method:
      1. Take anchor frame's features: (B, C, H, W)
      2. Center per-channel across spatial positions (subtract spatial mean)
      3. L2 norm across channel dim per position → scalar per position
      4. Min/max normalise across spatial positions

    High-energy positions = features distinct from spatial average =
    structurally salient content (subjects, edges, texture detail).
    Low-energy positions = features close to spatial average = uniform
    background, sky, etc.
    """
    if latent_5d.dim() != 5:
        if debug:
            print(f"  \u00b7 [energy] expected 5D latent, got dim={latent_5d.dim()}")
        return None

    B, C, F_dim, H, W = latent_5d.shape
    if F_dim == 0 or H == 0 or W == 0:
        return None

    f_idx = max(0, min(anchor_frame, F_dim - 1))
    frame = latent_5d[:, :, f_idx, :, :]                           # (B, C, H, W)

    # Center per-channel across spatial positions
    spatial_mean = frame.mean(dim=(2, 3), keepdim=True)            # (B, C, 1, 1)
    centered = frame - spatial_mean

    # L2 norm per position (cast to float for numerical stability)
    energy = centered.float().norm(dim=1)                          # (B, H, W)

    # Normalise to [0, 1] per-batch
    e_min = energy.amin(dim=(1, 2), keepdim=True)
    e_max = energy.amax(dim=(1, 2), keepdim=True)
    energy_norm = (energy - e_min) / (e_max - e_min + 1e-6)

    if debug:
        print(f"  \u00b7 [energy] map extracted: shape=(B={B}, H={H}, W={W}) "
              f"raw_min={energy.min().item():.3f} "
              f"raw_max={energy.max().item():.3f} "
              f"raw_mean={energy.mean().item():.3f}")

    return energy_norm


def _resample_energy_map(energy_map, target_H, target_W, debug=False):
    """
    Bilinear resample 2D energy map to target spatial dims.
    Input:  (B, H_src, W_src)
    Output: (B, target_H, target_W)
    """
    if energy_map is None:
        return None
    B, H_src, W_src = energy_map.shape
    if H_src == target_H and W_src == target_W:
        return energy_map
    em = energy_map.unsqueeze(1).float()                            # (B, 1, H, W)
    resampled = F.interpolate(em, size=(target_H, target_W),
                              mode="bilinear", align_corners=False)
    out = resampled.squeeze(1)                                      # (B, target_H, target_W)
    if debug:
        print(f"  \u00b7 [energy] resampled ({H_src},{W_src}) \u2192 ({target_H},{target_W})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Standard helpers (shared structure with v1.4)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_index_filter(s, n_blocks):
    if not s or not s.strip():
        return None
    indices = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a_i, b_i = int(a.strip()), int(b.strip())
                lo, hi = min(a_i, b_i), max(a_i, b_i)
                indices.update(range(max(0, lo), min(n_blocks - 1, hi) + 1))
            except Exception:
                continue
        else:
            try:
                idx = int(part)
                if 0 <= idx < n_blocks:
                    indices.add(idx)
            except Exception:
                continue
    return frozenset(indices) if indices else None


def _resolve_diffusion_model(m):
    for path in ("diffusion_model", "model", "transformer", "dit", "net"):
        obj = getattr(m.model, path, None)
        if obj is not None and hasattr(obj, "named_modules"):
            return obj, path
    if hasattr(m.model, "named_modules"):
        return m.model, "model"
    return None, None


def _extract_attn_tensor(out):
    if torch.is_tensor(out):
        return out, lambda t: t
    if isinstance(out, tuple):
        if len(out) > 0 and torch.is_tensor(out[0]):
            tail = out[1:]
            return out[0], (lambda t: (t,) + tail)
    if isinstance(out, list):
        if len(out) > 0 and torch.is_tensor(out[0]):
            tail = out[1:]
            return out[0], (lambda t: [t] + tail)
    if isinstance(out, dict):
        for k in ("hidden_states", "sample", "output"):
            if k in out and torch.is_tensor(out[k]):
                key = k
                base = dict(out)
                def _wrap(t, _base=base, _key=key):
                    new = dict(_base)
                    new[_key] = t
                    return new
                return out[key], _wrap
    return None, None


def _remove_prior_hooks(m, backbone):
    removed = 0
    h = getattr(backbone, HOOK_ATTR_BACKBONE, None)
    if h is not None:
        try:
            h.remove(); removed += 1
        except Exception:
            pass
        try:
            delattr(backbone, HOOK_ATTR_BACKBONE)
        except Exception:
            pass
    blocks = getattr(backbone, "transformer_blocks", None)
    if blocks is not None:
        for block in blocks:
            attn1 = getattr(block, "attn1", None)
            if attn1 is not None:
                h = getattr(attn1, HOOK_ATTR_ATTN1, None)
                if h is not None:
                    try:
                        h.remove(); removed += 1
                    except Exception:
                        pass
                    try:
                        delattr(attn1, HOOK_ATTR_ATTN1)
                    except Exception:
                        pass
    return removed


def _depth_multiplier(curve, block_idx, n_blocks):
    if n_blocks <= 1:
        return 1.0
    p = block_idx / (n_blocks - 1)
    if curve == "flat":
        return 1.0
    if curve == "ramp_up":
        return 2.0 * p
    if curve == "ramp_down":
        return 2.0 * (1.0 - p)
    if curve == "late_focus":
        raw = pow(2.71828, 3.0 * (p - 0.7))
        return raw / 0.779
    if curve == "middle":
        diff = p - 0.5
        raw = pow(2.71828, -(diff * diff) / (2 * 0.2 * 0.2))
        return raw / 0.499
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTXLatentAnchorAware:
    """
    Content-aware latent anchor: matching identical to v1.4, plus optional
    spatial energy modulation derived from a reference image or LATENT input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX2 model to patch with per-block attention "
                               "hooks. The returned model adds identity / "
                               "scene anchoring with optional reference-image "
                               "spatial energy weighting.",
                }),
            },
            "optional": {
                # ── Energy sources (use one or none) ──────────────────────────
                "reference_image":      ("IMAGE", {
                    "tooltip": "Optional reference image for spatial energy "
                               "weighting. High-energy regions of this image "
                               "(edges, faces, details) get stronger anchor "
                               "pull; low-energy regions (sky, flat areas) "
                               "get less. Wire alongside vae. For best "
                               "results, use the i2v conditioning image or "
                               "a composition reference.",
                }),
                "vae":                  ("VAE", {
                    "tooltip": "VAE required when reference_image is wired "
                               "(encodes the image into latent space for "
                               "energy computation). Wire the LTX2 VAE.",
                }),
                "energy_latent":        ("LATENT", {
                    "tooltip": "Alternative to reference_image+vae: provide a "
                               "pre-encoded LATENT for energy weighting. "
                               "Useful when you already have an encoded "
                               "latent and want to skip re-encoding.",
                }),
                # ── Schedule + simple-mode parameters ─────────────────────────
                "sigmas":               ("SIGMAS", {
                    "tooltip": "Connect from your sigma scheduler. Required "
                               "for predictable cache timing (cache_mode="
                               "schedule). Without it the node falls back to "
                               "manual_calls mode which is less reliable.",
                }),
                "strength":             ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Magnitude of pull toward cached anchor state. "
                               "0.05-0.20 typical. 0.10 is default. Higher "
                               "values can over-damp motion; lower values "
                               "barely register.",
                }),
                "cache_at_step":        ("INT", {
                    "default": 6, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Sampling step at which to lock the anchor "
                               "(snapshot the model's representation as "
                               "stable target). Mid-sampling values (3-9 on "
                               "13-step schedules) work best — peak "
                               "conditioning alignment before refinement.",
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum centered-cosine similarity for a "
                               "token to receive pull. Lower = broader effect. "
                               "0.50 is balanced; raise to 0.60-0.70 to "
                               "narrow pull to most-similar tokens only.",
                }),
                "decay_with_distance":  ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Per-frame strength decay from frame 0 to "
                               "last frame. 0 = uniform. 0.3-0.5 lets later "
                               "frames drift more while keeping early frames "
                               "anchored.",
                }),
                "energy_threshold":     ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Energy gating cutoff (when reference_image / "
                               "energy_latent is wired). 0 = uniform energy "
                               "(no spatial weighting, like plain LatentAnchor). "
                               "0.30 = broad: above-30%-percentile regions "
                               "get pull. 0.50 = above-median only. 0.80 = "
                               "top 20% energy regions only.",
                }),
                "bypass":               ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, model passes through unchanged. "
                               "Useful for A/B comparison without rewiring.",
                }),
                "debug":                ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Verbose per-block diagnostic output during "
                               "sampling. Shows cache state, similarity "
                               "stats, energy gating, and per-block timing.",
                }),
                # ── Advanced mode ────────────────────────────────────────────
                "advanced_mode":        ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle to expose research/diagnostic "
                               "parameters: cache_mode, forwards_per_step, "
                               "cache_warmup, anchor_frame, depth_curve, "
                               "block_index_filter.",
                }),
                "cache_mode":           (["schedule", "live_extraction", "manual_calls"], {
                    "default": "schedule",
                    "tooltip": "How cache timing is determined. schedule "
                               "(default, requires sigmas wired): cache "
                               "fires at the sigma corresponding to "
                               "cache_at_step. live_extraction: no cache, "
                               "regenerate target every call (softer "
                               "effect). manual_calls: cache fires after "
                               "cache_warmup block calls (fallback when "
                               "sigmas not wired).",
                }),
                "forwards_per_step":    ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "How many model forward passes occur per "
                               "sampling step. 1 for distilled CFG=1. 2 for "
                               "standard CFG > 1. 3+ for CFG+STG. Affects "
                               "cache timing math when in schedule mode.",
                }),
                "cache_warmup":         ("INT", {
                    "default": 144, "min": 0, "max": 5000, "step": 1,
                    "tooltip": "Number of block calls before cache fires "
                               "(when cache_mode=manual_calls). 144 ≈ 3 "
                               "sampling steps × 48 blocks at CFG=1.",
                }),
                "anchor_frame":         ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Which frame's features to use as anchor "
                               "source. 0 = first frame (conditioning "
                               "frame in i2v — recommended).",
                }),
                "depth_curve":          (["flat", "ramp_up", "ramp_down", "late_focus", "middle"], {
                    "default": "flat",
                    "tooltip": "Per-block strength scaling. flat = uniform "
                               "(recommended). ramp_up = stronger late "
                               "blocks. ramp_down = stronger early. "
                               "late_focus = quadratic late emphasis. "
                               "middle = strongest in middle blocks.",
                }),
                "block_index_filter":   ("STRING", {
                    "default": "",
                    "tooltip": "Limit hooks to specific blocks, e.g. '10-30' "
                               "or '5,7,15-20'. Empty = all 48 blocks. "
                               "Useful for ablation studies.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Latent anchor with optional spatial energy modulation from external reference. "
        "Reference image (or LATENT input) provides spatial weighting; matching itself "
        "uses running latent (correct feature space)."
    )

    def patch(self, model,
              reference_image=None, vae=None, energy_latent=None,
              sigmas=None,
              strength=0.10, cache_at_step=6,
              similarity_threshold=0.50, decay_with_distance=0.0,
              energy_threshold=0.30,
              bypass=False, debug=False,
              advanced_mode=False,
              cache_mode="schedule", forwards_per_step=1, cache_warmup=144,
              anchor_frame=0, depth_curve="flat", block_index_filter=""):

        if not advanced_mode:
            cache_mode         = "schedule"
            forwards_per_step  = 1
            cache_warmup       = 144
            anchor_frame       = 0
            depth_curve        = "flat"
            block_index_filter = ""
            # NOTE: debug is intentionally NOT reset here — it's a diagnostic
            # tool that should work in both simple and advanced modes.

        m = model.clone()
        backbone, _ = _resolve_diffusion_model(m)
        if backbone is None:
            print("\u2192 [10S] AwareAnchor v2.3: could not locate diffusion backbone.")
            return (m,)
        if not hasattr(backbone, "transformer_blocks"):
            print(f"\u2192 [10S] AwareAnchor v2.3: backbone {type(backbone).__name__} has no "
                  f"'transformer_blocks' attribute.")
            return (m,)

        n_removed = _remove_prior_hooks(m, backbone)
        if n_removed > 0:
            print(f"\u2192 [10S] AwareAnchor v2.3: removed {n_removed} prior hook(s)")

        if bypass or strength <= 0.0:
            reason = "bypass=True" if bypass else "strength == 0"
            print(f"\u2192 [10S] AwareAnchor v2.3: {reason} \u2014 hooks cleared")
            return (m,)

        blocks = backbone.transformer_blocks
        n_blocks = len(blocks)
        idx_filter = _parse_index_filter(block_index_filter, n_blocks)

        # ─── Energy source resolution ────────────────────────────────────────
        # Priority: explicit reference_image > energy_latent > none
        ref_latent_raw = None
        energy_source = "none"

        if reference_image is not None:
            if vae is None:
                print("\u2192 [10S] AwareAnchor v2.3: reference_image provided but VAE not "
                      "connected. Energy modulation disabled.")
            else:
                if debug:
                    print(f"  \u00b7 [ref] VAE-encoding reference image "
                          f"shape={tuple(reference_image.shape)}")
                ref_latent_raw = _vae_encode_reference(vae, reference_image, debug=debug)
                if ref_latent_raw is not None:
                    energy_source = "reference_image"

        if ref_latent_raw is None and energy_latent is not None:
            ref_latent_raw = _extract_latent_from_input(energy_latent, debug=debug)
            if ref_latent_raw is not None:
                energy_source = "energy_latent"

        # Compute initial energy map (resample happens later when running shape known)
        initial_energy_map = None
        if ref_latent_raw is not None:
            initial_energy_map = _extract_energy_map(ref_latent_raw,
                                                     anchor_frame=anchor_frame,
                                                     debug=debug)

        # ─── Schedule resolution ──────────────────────────────────────────────
        target_call_count_from_schedule = None
        target_idx_in_schedule = None
        target_sigma_at_idx = None
        schedule_health_warnings = []
        if sigmas is not None and cache_mode == "schedule":
            try:
                sched = sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas)
                if len(sched) > 0:
                    target_idx = max(0, min(cache_at_step, len(sched) - 1))
                    target_idx_in_schedule = target_idx
                    target_sigma_at_idx = sched[target_idx]
                    target_call_count_from_schedule = target_idx * forwards_per_step
                    if debug:
                        print(f"  \u00b7 [schedule] step{cache_at_step} \u2192 "
                              f"sigma{target_sigma_at_idx:.4f} \u2192 "
                              f"call{target_call_count_from_schedule}")

                    # Health checks on schedule alignment
                    if cache_at_step >= len(sched) - 1:
                        schedule_health_warnings.append(
                            f"cache_at_step={cache_at_step} \u2265 schedule length {len(sched)-1}; "
                            f"clamped to last step (sigma={target_sigma_at_idx:.3f}). "
                            f"Cache will lock at end of sampling \u2014 effectively never used. "
                            f"Reduce cache_at_step (try {max(1, (len(sched)-1)//2)} for "
                            f"mid-sampling on this {len(sched)-1}-step schedule)."
                        )
                    elif target_sigma_at_idx < 0.05:
                        schedule_health_warnings.append(
                            f"resolved sigma={target_sigma_at_idx:.3f} is very near zero; "
                            f"cache locks at end of sampling. Try smaller cache_at_step."
                        )
                    elif target_sigma_at_idx > 0.95:
                        schedule_health_warnings.append(
                            f"resolved sigma={target_sigma_at_idx:.3f} is very near start "
                            f"(pure noise). Cache will lock before model has integrated "
                            f"conditioning. Try larger cache_at_step."
                        )
            except Exception as e:
                if debug:
                    print(f"  \u00b7 [schedule] failed: {type(e).__name__}: {e}")

        # ─── Shared state ────────────────────────────────────────────────────
        state = {
            "latent_shape":      None,
            "shape_logged":      False,
            "hook_logged":       False,
            "calls":             0,
            "anchor_cache":      {},
            "block_calls":       {},
            "cache_logged":      False,
            "current_sigma":     None,
            "energy_map_resampled": None,    # set on first sampling forward
            "energy_resampled_failed": False,
            "health_logged":     False,
        }

        # ─── Backbone pre-hook for shape capture ─────────────────────────────
        def _capture_5d(it, label):
            for v in it:
                if torch.is_tensor(v) and v.dim() == 5:
                    state["latent_shape"] = tuple(v.shape)
                    if debug and not state["shape_logged"]:
                        print(f"  \u00b7 captured 5D latent from {label}: "
                              f"{state['latent_shape']}")
                        state["shape_logged"] = True
                    return True
            return False

        def backbone_pre_hook_kw(module, args, kwargs):
            if state["latent_shape"] is None:
                if args:
                    _capture_5d(args, "args")
                if state["latent_shape"] is None and kwargs:
                    _capture_5d(kwargs.values(), "kwargs")
            return None

        def backbone_pre_hook_args_only(module, args):
            return backbone_pre_hook_kw(module, args, {})

        try:
            pre_handle = backbone.register_forward_pre_hook(backbone_pre_hook_kw, with_kwargs=True)
            pre_hook_mode = "with_kwargs"
        except TypeError:
            pre_handle = backbone.register_forward_pre_hook(backbone_pre_hook_args_only)
            pre_hook_mode = "args_only"
        setattr(backbone, HOOK_ATTR_BACKBONE, pre_handle)

        # ─── Reshape helpers ──────────────────────────────────────────────────
        def _to_grid(t, B, F_tok, H_tok, W_tok, D):
            return t.reshape(B, F_tok, H_tok, W_tok, D)

        def _from_grid(grid, B, F_tok, H_tok, W_tok, D, seq):
            return grid.reshape(B, seq, D)

        def _should_cache_now(block_idx, call_count):
            if cache_mode == "live_extraction":
                return False
            if target_call_count_from_schedule is not None:
                return call_count >= target_call_count_from_schedule
            if cache_mode == "schedule":
                return call_count >= cache_warmup
            return call_count >= cache_warmup

        def _resample_energy_if_needed(target_H, target_W, target_dtype, target_device):
            """Lazy resample energy map once we know running latent shape."""
            if state["energy_map_resampled"] is not None:
                return
            if state["energy_resampled_failed"]:
                return
            if initial_energy_map is None:
                state["energy_resampled_failed"] = True
                return
            try:
                resampled = _resample_energy_map(initial_energy_map, target_H, target_W,
                                                 debug=debug)
                if resampled is None:
                    state["energy_resampled_failed"] = True
                    return
                state["energy_map_resampled"] = resampled.to(
                    dtype=target_dtype, device=target_device
                )
            except Exception as e:
                state["energy_resampled_failed"] = True
                if debug:
                    print(f"  \u00b7 [energy] resample failed: {type(e).__name__}: {e}")

        # ─── Core blend (matching identical to v1.4, energy modulation added) ─
        def _apply_blend(tensor, block_idx, depth_mult):
            B, seq, D = tensor.shape
            _, _, F_lat, H_lat, W_lat = state["latent_shape"]
            F_tok = max(1, F_lat // TEMPORAL_PATCH)
            H_tok = max(1, H_lat // SPATIAL_PATCH)
            W_tok = max(1, W_lat // SPATIAL_PATCH)
            if F_tok * H_tok * W_tok != seq:
                return None

            anchor_idx = max(0, min(anchor_frame, F_tok - 1))

            # Lazy resample energy to running grid dims
            _resample_energy_if_needed(H_tok, W_tok, tensor.dtype, tensor.device)
            energy_active = (state["energy_map_resampled"] is not None
                             and energy_threshold > 0.0)

            if not state["hook_logged"]:
                K = H_tok * W_tok
                if cache_mode == "live_extraction":
                    cache_str = "live_extraction (no cache)"
                elif target_call_count_from_schedule is not None:
                    cache_str = (f"schedule@call>={target_call_count_from_schedule} "
                                 f"(step{cache_at_step}, sigma{target_sigma_at_idx:.3f})")
                else:
                    cache_str = f"manual@{cache_warmup}calls"
                energy_str = (f"energy={energy_source}@thr{energy_threshold}"
                              if energy_active else "energy=off")
                print(f"\u2192 [10S] AwareAnchor v2.3: HOOK ACTIVE | first fire on blk{block_idx} | "
                      f"grid=(F={F_tok},H={H_tok},W={W_tok}) seq={seq} D={D} "
                      f"curve={depth_curve} strength={strength} decay={decay_with_distance} "
                      f"sim_thr={similarity_threshold} cache={cache_str} {energy_str} "
                      f"K={K} (whole anchor frame)")
                state["hook_logged"] = True

            grid = _to_grid(tensor, B, F_tok, H_tok, W_tok, D)

            # Per-frame strength schedule
            if F_tok > 1 and decay_with_distance > 0.0:
                dist = torch.arange(F_tok, dtype=tensor.dtype, device=tensor.device)
                dist = (dist - anchor_idx).abs() / max(1, F_tok - 1)
                fs = strength * depth_mult * (1.0 - decay_with_distance * dist).clamp(min=0.0)
            else:
                fs = torch.full((F_tok,), strength * depth_mult,
                                dtype=tensor.dtype, device=tensor.device)
            fs[anchor_idx] = 0.0

            # ─── Anchor extraction (v1.4-identical: from running latent) ─────
            shape_key = (F_tok, H_tok, W_tok)
            cache_entry = state["anchor_cache"].get(block_idx)
            cache_valid = cache_entry is not None and cache_entry.get("shape_key") == shape_key

            call_count = state["block_calls"].get(block_idx, 0)
            state["block_calls"][block_idx] = call_count + 1

            if cache_valid:
                anchor_flat = cache_entry["anchor_flat"]
                if anchor_flat.shape[0] != B:
                    anchor_flat = anchor_flat[:1].expand(B, -1, -1)
                K = anchor_flat.shape[1]
                cached_afm = cache_entry["anchor_frame_mean"]
                if cached_afm.shape[0] != B:
                    cached_afm = cached_afm[:1].expand(B, -1, -1)
                anchor_frame_mean_for_sim = cached_afm
            else:
                anchor_full = grid[:, anchor_idx, :, :, :]
                K = H_tok * W_tok
                anchor_flat = anchor_full.reshape(B, K, D)
                anchor_frame_mean_for_sim = None

                if _should_cache_now(block_idx, call_count):
                    grid_for_mean = grid.reshape(B, F_tok, H_tok * W_tok, D)
                    afm = grid_for_mean[:, anchor_idx, :, :].mean(dim=1, keepdim=True)
                    state["anchor_cache"][block_idx] = {
                        "anchor_flat":       anchor_flat[:1].detach().clone(),
                        "anchor_frame_mean": afm[:1].detach().clone(),
                        "shape_key":         shape_key,
                    }
                    if debug and not state["cache_logged"]:
                        print(f"  \u00b7 [cache] populated at blk{block_idx} "
                              f"after {call_count} calls (K={K}, mode={cache_mode})")
                        state["cache_logged"] = True

            # ─── Per-frame centered cosine similarity ────────────────────────
            N = F_tok * H_tok * W_tok
            grid_for_sim = grid.reshape(B, F_tok, H_tok * W_tok, D)
            frame_mean = grid_for_sim.mean(dim=2, keepdim=True)
            centered_grid = grid_for_sim - frame_mean
            all_for_sim = centered_grid.reshape(B, N, D)

            if anchor_frame_mean_for_sim is not None:
                anchor_frame_mean = anchor_frame_mean_for_sim.expand(B, 1, D)
            else:
                anchor_frame_mean = frame_mean[:, anchor_idx, :, :]
            anchor_for_sim = anchor_flat - anchor_frame_mean.expand(B, K, D)

            all_norm = F.normalize(all_for_sim, dim=-1, eps=1e-6)
            anchor_norm = F.normalize(anchor_for_sim, dim=-1, eps=1e-6)
            sim = torch.bmm(all_norm, anchor_norm.transpose(1, 2))

            best_sim, best_idx = sim.max(dim=-1)

            expanded_idx = best_idx.unsqueeze(-1).expand(-1, -1, D)
            gathered = torch.gather(anchor_flat, 1, expanded_idx)

            mask = torch.sigmoid((best_sim - similarity_threshold) * TRACK_SHARPNESS)
            mask_grid = mask.reshape(B, F_tok, H_tok, W_tok, 1)

            # ─── Energy modulation (v2.2: sigmoid threshold) ─────────────────
            # Linear interp (v2.1) created intermediate-value distortion: at
            # mod=0.5, every token still got partial pull, just at varying
            # strengths within a coherent object — model couldn't reconcile
            # the gradient and produced anatomy artifacts. v2.2 uses a steep
            # sigmoid threshold so tokens are clearly in-or-out of the pull
            # set, with only a narrow ambiguous transition zone.
            if energy_active:
                e_map = state["energy_map_resampled"]
                if e_map.shape[0] != B:
                    e_map = e_map[:1].expand(B, -1, -1)
                e_grid = e_map.unsqueeze(1).unsqueeze(-1)              # (B, 1, H, W, 1)
                # Steep sigmoid (sharpness 16) gives narrow transition zone.
                # energy_threshold = cutoff:
                #   0.0  = uniform mask (energy_active is False at 0; this is just safeguard)
                #   0.30 = pull above lower-third energy (default; broad but selective)
                #   0.50 = pull above-median energy
                #   0.80 = pull only top ~20% energy regions
                e_factor = torch.sigmoid((e_grid - energy_threshold) * 16.0)
                mask_grid = mask_grid * e_factor

            diff_grid = gathered.reshape(B, F_tok, H_tok, W_tok, D) - grid
            residual = fs.view(1, F_tok, 1, 1, 1) * mask_grid * diff_grid
            grid_modified = grid + residual

            if debug and state["calls"] < 3:
                try:
                    mpf = mask_grid.squeeze(-1).sum(dim=(2, 3))[0].tolist()
                    preview = [f"{v:.1f}" for v in mpf[:8]]
                    tail = (f" ...({len(mpf)-8} more)" if len(mpf) > 8 else "")
                    sim_f = best_sim.float()
                    sim_mean_v = sim_f.mean().item()
                    sim_p90_v = sim_f.quantile(0.9).item()
                    res_max_v = residual.float().abs().max().item()
                    avg_mask = mask_grid.float().mean().item()

                    e_str = ""
                    e_p90_v = None
                    if energy_active:
                        e_map_disp = state["energy_map_resampled"]
                        e_p90_v = e_map_disp.float().quantile(0.9).item()
                        e_str = (f" energy_map_mean={e_map_disp.mean().item():.3f}"
                                 f"/p90={e_p90_v:.3f}")
                    print(f"  \u00b7 blk{block_idx} call {state['calls']}: "
                          f"depth_mult={depth_mult:.3f} K={K} "
                          f"cache={'hit' if cache_valid else 'miss'} "
                          f"sim_mean={sim_mean_v:.3f} "
                          f"sim_p90={sim_p90_v:.3f} "
                          f"|residual|max={res_max_v:.4f}"
                          f"{e_str}")
                    print(f"    tokens_in_mask per_frame: [{', '.join(preview)}]{tail}")

                    # ─── One-shot health analysis on first call ─────────────
                    if not state["health_logged"]:
                        warns = list(schedule_health_warnings)  # carry forward registration warnings
                        if sim_mean_v < 0.10:
                            warns.append(f"sim_mean={sim_mean_v:.3f} very low \u2014 features "
                                         f"may be in mismatched space; tracking unstable")
                        if sim_p90_v < 0.20:
                            warns.append(f"sim_p90={sim_p90_v:.3f} very low \u2014 no tokens "
                                         f"matching well; lower similarity_threshold or check "
                                         f"feature-space alignment")
                        if res_max_v > 10.0:
                            warns.append(f"|residual|max={res_max_v:.2f} very high \u2014 "
                                         f"approaching corruption regime; lower strength")
                        if avg_mask < 0.005:
                            warns.append(f"avg mask={avg_mask:.4f} effectively zero \u2014 "
                                         f"intervention is no-op; lower thresholds or check "
                                         f"energy_threshold")
                        if avg_mask > 0.85:
                            warns.append(f"avg mask={avg_mask:.3f} very high \u2014 nearly all "
                                         f"tokens being pulled; raise similarity_threshold or "
                                         f"energy_threshold for selectivity")
                        if e_p90_v is not None:
                            if e_p90_v < 0.30:
                                warns.append(f"energy_map_p90={e_p90_v:.3f} very low \u2014 "
                                             f"reference has flat energy distribution "
                                             f"(uniform image?). Spatial gating ineffective.")
                            elif e_p90_v > 0.97:
                                warns.append(f"energy_map_p90={e_p90_v:.3f} very high \u2014 "
                                             f"reference is heavily edge-dominated; gating "
                                             f"may be too narrow")
                        for w in warns:
                            print(f"  \u26a0  [health] {w}")
                        state["health_logged"] = True

                except Exception as _e:
                    print(f"  \u00b7 blk{block_idx} diagnostic failed: "
                          f"{type(_e).__name__}: {_e}")
            state["calls"] += 1

            return _from_grid(grid_modified, B, F_tok, H_tok, W_tok, D, seq)

        def make_attn1_hook(block_idx):
            depth_mult = _depth_multiplier(depth_curve, block_idx, n_blocks)
            def hook(module, inputs, output):
                try:
                    if strength <= 0.0 or depth_mult <= 0.0:
                        return None
                    if state["latent_shape"] is None:
                        return None
                    tensor, wrap = _extract_attn_tensor(output)
                    if tensor is None or tensor.dim() != 3:
                        return None
                    new_tensor = _apply_blend(tensor, block_idx, depth_mult)
                    if new_tensor is None:
                        return None
                    return wrap(new_tensor)
                except Exception as e:
                    if debug:
                        print(f"\u2192 [10S] AwareAnchor v2.3: blk{block_idx} hook error: "
                              f"{type(e).__name__}: {e}")
                    return None
            return hook

        # ─── Register hooks ───────────────────────────────────────────────────
        hooked = 0
        skipped = 0
        missing = 0
        for i, block in enumerate(blocks):
            if idx_filter is not None and i not in idx_filter:
                skipped += 1
                continue
            if not hasattr(block, "attn1"):
                missing += 1
                continue
            try:
                h = block.attn1.register_forward_hook(make_attn1_hook(i))
                setattr(block.attn1, HOOK_ATTR_ATTN1, h)
                hooked += 1
            except Exception as e:
                missing += 1
                if debug:
                    print(f"\u2192 [10S] AwareAnchor v2.3: blk{i}.attn1 hook failed: "
                          f"{type(e).__name__}: {e}")

        if cache_mode == "live_extraction":
            cache_str = "live_extraction"
        elif target_call_count_from_schedule is not None:
            cache_str = (f"step{cache_at_step}/sigma{target_sigma_at_idx:.3f}/"
                         f"call{target_call_count_from_schedule}")
        else:
            cache_str = f"manual@{cache_warmup}calls"

        energy_str = (f"energy={energy_source}@thr{energy_threshold}"
                      if initial_energy_map is not None and energy_threshold > 0
                      else "energy=off")

        adv_str = " [advanced]" if advanced_mode else ""
        print(f"\u2192 [10S] AwareAnchor v2.3: {hooked}/{n_blocks} blocks hooked "
              f"(skipped={skipped}, missing={missing}){adv_str} | "
              f"backbone={type(backbone).__name__} pre_hook={pre_hook_mode} "
              f"curve={depth_curve} | "
              f"strength={strength} decay={decay_with_distance} "
              f"sim_thr={similarity_threshold} cache={cache_str} {energy_str}")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXLatentAnchorAware": LTXLatentAnchorAware,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLatentAnchorAware": "\U0001f3af LTX Latent Anchor Aware",
}
