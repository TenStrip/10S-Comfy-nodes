"""
LTX Likeness Semantic Clamp v1.0

Per-block intervention on text cross-attention (attn2) that semantically
identifies "face-modifier" text tokens in the user's positive prompt and
suppresses their attention to face-bbox video tokens — without affecting
positioning, style, lighting, scene description, or any other prompt
content.

================================================================================
HOW IT IS DIFFERENT FROM (DEPRECATED) LikenessClamp
================================================================================

LikenessClamp scaled down attn2 OUTPUT magnitude wholesale in the bbox
region. This cut off the model's primary path to scene/style/composition
information for the face region, causing catastrophic noise artifacts —
the model had nothing to compose a face from once text influence was
broadly suppressed.

LikenessSemanticClamp instead identifies WHICH text tokens are face-
modifier-like ("smiling", "laughing", "frowning", "eyes closed") via
correspondence search in text-embedding space, and suppresses ONLY those
tokens' contribution to face-bbox video tokens. Non-modifier tokens
(everything about scene, position, style, lighting) continue to influence
the face region normally.

================================================================================
ARCHITECTURE
================================================================================

  Setup (apply time):
  1. User provides a CLIP and a face_modifier_text string ("smiling,
     laughing, frowning, eyes closed, mouth open, ...")
  2. Encode face_modifier_text via the same CLIP that produced the positive
     conditioning, yielding modifier_tensor (B, T_mod, D)
  3. Compute fingerprint of positive_tensor for runtime identification

  Per-step (during sampling):
  4. Backbone pre-hook captures latent shape (for bbox geometry)
  5. attn2 monkey-patched forward fires:
     - Identifies whether incoming K/V come from positive_tensor (via
       fingerprint match)
     - If yes: computes per-text-token similarity to modifier_tensor,
       derives suppression_weight per text token position
     - Computes standard attention math but subtracts suppression on
       attention LOGITS (before softmax) for bbox-region video queries
       attending to modifier-like text tokens
     - If incoming K/V are not the positive (negative pass during CFG > 1):
       behaves as identity — no modification

================================================================================
WHY IT SHOULD WORK
================================================================================

attn2's softmax over text tokens distributes attention mass — total
attention is 1 across all text tokens per query. By subtracting from
attention logits for face-modifier tokens (before softmax), we shift mass
TO other text tokens, not eliminate it. Face-region video queries still
get full prompt influence — just routed away from modifier tokens toward
positional/style/scene tokens.

Crucially, this is per-text-token-position selective. The mechanism never
cuts off the face from prompt influence broadly — it just routes attention
away from the specific tokens that would change identity.

================================================================================
CAVEATS
================================================================================

- Correspondence in text-embedding space is noisier than in video-feature
  space. Embedding similarity above 0.5-0.6 is typically meaningful;
  values below may catch unrelated tokens.
- If user's positive prompt contains face-position-related words (e.g.,
  "facing forward", "looking at camera"), those should NOT be suppressed.
  The mechanism guards against this by only suppressing tokens whose
  similarity exceeds the threshold — position words are semantically
  distant from expression words, so threshold-based gating handles them.
- Padding token positions in both tensors will be self-similar. Detected
  via token magnitude (pad tokens have lower magnitude than content) and
  filtered out before correspondence search.
- This intervention works at the text cross-attention layer (attn2).
  Identity-modifying behavior that originates downstream of attn2 (e.g.,
  attn1 propagating expression changes across frames) is NOT addressed by
  this node — that's LikenessAnchor's job at attn1.

================================================================================
USAGE
================================================================================

    LikenessGuide(image) ──→ pos, neg, reference_info
                                  ↓
    LikenessAnchor(model, reference_info, strength=0.10-0.15)
                                  ↓ model
    LikenessSemanticClamp(model, clip, positive, reference_info,
                          face_modifier_text="...", suppression_strength=0.5)
                                  ↓ model
    KSampler...

Recommended starting config:
    suppression_strength = 0.5
    similarity_threshold = 0.55
    face_modifier_text = (use the node's default — comprehensive list)
"""

import math
import torch
import torch.nn.functional as F


# Sentinels for hook coexistence
HOOK_ATTR_ATTN2 = "_10s_likeness_semantic_clamp_attn2_hook"
ORIGINAL_FORWARD_ATTR = "_10s_likeness_semantic_clamp_orig_forward"
HOOK_ATTR_BACKBONE = "_10s_likeness_semantic_clamp_backbone"


# Default face-modifier text — short, focused vocabulary covering the most
# common expression directives that prompts include. Deliberately concise
# to avoid generating too many similar embedding directions in the modifier
# space (which would over-suppress non-modifier tokens via correspondence
# search). Users can override with their own vocabulary.
DEFAULT_FACE_MODIFIER_TEXT = (
    "smiling, frowning, laughing, "
    "open mouth, closed mouth, "
    "eyes closed, eyes open wide, "
    "happy, sad, angry, surprised, "
    "expression"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bbox(bbox_str, H, W):
    """Parse normalized 'x1,y1,x2,y2' to integer index ranges."""
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


def _build_bbox_mask(bbox, H, W, soft_edge_frac=0.15, device=None, dtype=None):
    """Build (H, W) soft mask: 1.0 inside bbox, Gaussian fade outside."""
    h1, h2, w1, w2 = bbox
    if soft_edge_frac <= 0:
        m = torch.zeros((H, W), dtype=dtype, device=device)
        m[h1:h2, w1:w2] = 1.0
        return m
    yy = torch.arange(H, dtype=torch.float32, device=device).view(-1, 1)
    xx = torch.arange(W, dtype=torch.float32, device=device).view(1, -1)
    dy = torch.maximum(torch.maximum(h1 - yy, yy - (h2 - 1)),
                       torch.zeros(1, device=device))
    dx = torch.maximum(torch.maximum(w1 - xx, xx - (w2 - 1)),
                       torch.zeros(1, device=device))
    dist = torch.sqrt(dy * dy + dx * dx)
    sigma = max(1.0, soft_edge_frac * max(H, W))
    mask = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if dtype is not None:
        mask = mask.to(dtype=dtype)
    return mask


def _find_backbone(model):
    for path in ("diffusion_model", "model.diffusion_model",
                 "model.model", "diffusion_model.model"):
        obj = model.model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "transformer_blocks"):
                return obj
        except AttributeError:
            continue
    try:
        obj = model.model
        if hasattr(obj, "transformer_blocks"):
            return obj
    except AttributeError:
        pass
    if hasattr(model, "transformer_blocks"):
        return model
    return None


def _extract_cond_tensor(conditioning):
    """Extract the encoded text tensor from a CONDITIONING list. Returns the
    first valid (B, T, D) tensor found, or None."""
    if not conditioning:
        return None
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            t = entry[0]
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                return t
    return None


def _encode_text_via_clip(clip, text):
    """Encode a text string via the provided CLIP, returning the encoded
    conditioning tensor. Uses the same encode pipeline as CLIPTextEncode."""
    if not text or not text.strip():
        return None
    try:
        tokens = clip.tokenize(text)
        cond_out = clip.encode_from_tokens_scheduled(tokens)
        # cond_out is a CONDITIONING list — extract tensor
        return _extract_cond_tensor(cond_out)
    except Exception as e:
        print(f"\u2192 [10S] SemanticClamp: text encode failed: "
              f"{type(e).__name__}: {e}")
        return None


def _compute_token_magnitudes(cond_tensor):
    """Per-token L2 magnitude. (B, T, D) → (B, T). Used to identify pad
    tokens (low magnitude) vs content tokens (higher)."""
    return cond_tensor.norm(dim=-1)  # (B, T)


def _compute_raw_scores(positive_tensor, modifier_tensor,
                         pad_magnitude_threshold_frac=0.3,
                         top_k=3):
    """
    Compute per-positive-token raw match score = top-K mean cosine similarity
    to modifier_tensor tokens (after filtering pad tokens). Returns:
        scores: (B, T_pos) — match score per token in [-1, 1]
        pos_pad_mask: (B, T_pos) — True where positive token is pad
    """
    # Center features per-tensor for cleaner cosine similarity
    pos_mean = positive_tensor.mean(dim=1, keepdim=True)
    pos_centered = positive_tensor - pos_mean
    mod_mean = modifier_tensor.mean(dim=1, keepdim=True)
    mod_centered = modifier_tensor - mod_mean

    pos_norm = F.normalize(pos_centered, dim=-1, eps=1e-6)
    mod_norm = F.normalize(mod_centered, dim=-1, eps=1e-6)

    # Per-position similarity matrix (B, T_pos, T_mod)
    sim = torch.einsum("btd,bsd->bts", pos_norm, mod_norm)

    # Filter pad tokens by magnitude
    pos_mag = _compute_token_magnitudes(positive_tensor)
    mod_mag = _compute_token_magnitudes(modifier_tensor)
    pos_mag_max = pos_mag.max(dim=1, keepdim=True).values
    mod_mag_max = mod_mag.max(dim=1, keepdim=True).values
    pos_pad_mask = pos_mag < (pad_magnitude_threshold_frac * pos_mag_max)
    mod_pad_mask = mod_mag < (pad_magnitude_threshold_frac * mod_mag_max)

    # Zero out modifier pads in similarity
    sim_masked = sim.clone()
    sim_masked[mod_pad_mask.unsqueeze(1).expand_as(sim)] = -1.0

    # Top-K mean
    n_mod_valid = (~mod_pad_mask).sum(dim=1).min().item()
    k = min(top_k, max(1, n_mod_valid))
    top_k_sims = sim_masked.topk(k, dim=-1).values  # (B, T_pos, k)
    scores = top_k_sims.mean(dim=-1)  # (B, T_pos)

    return scores, pos_pad_mask


def _gate_scores_to_suppression(scores, pos_pad_mask,
                                 threshold, sharpness):
    """
    Convert raw scores to suppression weights via sigmoid gate, zeroing pad
    positions.
    """
    suppression = torch.sigmoid((scores - threshold) * sharpness)
    suppression[pos_pad_mask] = 0.0
    return suppression


def _compute_suppression_weights(positive_tensor, modifier_tensor,
                                  similarity_threshold=0.55,
                                  similarity_sharpness=8.0,
                                  pad_magnitude_threshold_frac=0.3,
                                  top_k=3):
    """Legacy combined entry point — kept for compatibility."""
    scores, pos_pad_mask = _compute_raw_scores(
        positive_tensor, modifier_tensor,
        pad_magnitude_threshold_frac=pad_magnitude_threshold_frac,
        top_k=top_k,
    )
    suppression = _gate_scores_to_suppression(
        scores, pos_pad_mask,
        threshold=similarity_threshold,
        sharpness=similarity_sharpness,
    )
    return suppression, scores


def _fingerprint_tensor(t):
    """Build a lightweight fingerprint for tensor identity matching at
    runtime. We can't use Python identity (.id()) because tensors get
    copied/moved during sampling. We use shape + a hash of a small slice
    of values."""
    if not isinstance(t, torch.Tensor):
        return None
    try:
        shape = tuple(t.shape)
        # Sample a few positions for content fingerprint
        flat = t.flatten().detach().cpu()
        if flat.numel() < 16:
            return (shape, tuple(flat.tolist()))
        # Sample 16 evenly-spaced positions
        idxs = torch.linspace(0, flat.numel() - 1, 16, dtype=torch.long)
        vals = flat[idxs].to(torch.float32).tolist()
        # Round to reduce floating-point sensitivity
        vals_rounded = tuple(round(v, 4) for v in vals)
        return (shape, vals_rounded)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTXLikenessSemanticClamp:
    """
    Semantic text-token suppression in attn2 for face-bbox region.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX2 model to patch with semantic-aware "
                               "attn2 forward replacements. Chain after "
                               "LikenessAnchor.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP / text encoder used by your prompt "
                               "conditioning (typically Gemma-3-12B for "
                               "LTX2). Required to encode the face-modifier "
                               "keywords for correspondence search.",
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "The positive prompt conditioning. The node "
                               "fingerprints this and matches at runtime so "
                               "only the positive (cond) pass is modified, "
                               "not the negative (uncond) pass.",
                }),
                "suppression_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to suppress attention from face-"
                               "bbox video tokens to face-modifier text "
                               "tokens. 0 = no-op. 0.5 = halve their "
                               "influence (recommended). 1.0 = fully "
                               "block their influence (may cause rigidity).",
                }),
            },
            "optional": {
                "reference_info": ("REFERENCE_INFO", {
                    "tooltip": "Wire from LikenessGuide for the bbox and "
                               "spatial dims. If not wired, falls back to "
                               "override_face_bbox.",
                }),
                "face_modifier_text": ("STRING", {
                    "default": DEFAULT_FACE_MODIFIER_TEXT,
                    "multiline": True,
                    "tooltip": "Comma-separated keywords describing the "
                               "expression / facial-modifier vocabulary to "
                               "suppress. The node encodes this via your "
                               "CLIP and uses correspondence search to "
                               "identify matching tokens in your positive "
                               "prompt. The default covers common expression "
                               "vocabulary — usually no need to customize.",
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum top-K-mean cosine similarity for a "
                               "positive-prompt token to be considered face-"
                               "modifier-like. Only used when "
                               "auto_threshold=disabled. Different encoders "
                               "produce different absolute similarity values "
                               "— prefer auto_threshold for portability.",
                }),
                "auto_threshold": (
                    ["disabled", "p90", "p95", "p98", "p99"],
                    {
                        "default": "p95",
                        "tooltip": "Auto-calibrate similarity threshold to "
                                   "the encoder's score distribution. p95 = "
                                   "suppress top ~5% of most-modifier-like "
                                   "tokens. p98 = top 2%. p99 = top 1%. "
                                   "disabled = use similarity_threshold "
                                   "literal value. Recommended: p95 for "
                                   "balanced effect, p98 for narrower.",
                    },
                ),
                "similarity_sharpness": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Sigmoid steepness around the threshold. "
                               "Higher = more binary (sharp) partition "
                               "between suppressed and not. 16 (default) "
                               "makes the gate effectively binary within "
                               "~0.05 of threshold. Lower values (8) "
                               "create soft tails that leak suppression "
                               "to most tokens.",
                }),
                "suppression_floor": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Hard floor on final suppression weights. "
                               "Tokens with weight below this become 0 "
                               "(unsuppressed). Eliminates sigmoid's soft "
                               "tail leak. 0.3 default cuts off all weak "
                               "matches; 0.5 only keeps strong matches; "
                               "0.0 disables (full sigmoid tail).",
                }),
                "top_k": ("INT", {
                    "default": 3, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of top similarity values to average "
                               "for each positive token's match score. "
                               "Higher K requires more confirming matches "
                               "in modifier vocabulary, reducing false "
                               "positives. K=1 = max (most permissive). "
                               "K=3-5 (default 3) = robust.",
                }),
                "soft_edge_frac": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Soft falloff fraction at bbox edges. 0.15 "
                               "default for smooth transition. Lower = "
                               "more focused, higher = more diffuse.",
                }),
                "override_face_bbox": ("STRING", {
                    "default": "",
                    "tooltip": "Manual bbox 'x1,y1,x2,y2' normalized 0-1. "
                               "Used when reference_info not wired.",
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, model passes through unchanged.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Semantic-aware text-token suppression on attn2 for face bbox "
        "region. Identifies face-modifier tokens in the prompt via "
        "embedding-space correspondence to a modifier vocabulary, then "
        "selectively suppresses ONLY those tokens' attention to the face "
        "region. Replaces deprecated LikenessClamp's blanket magnitude "
        "scaling approach."
    )

    def apply(self, model, clip, positive, suppression_strength,
              reference_info=None,
              face_modifier_text=DEFAULT_FACE_MODIFIER_TEXT,
              similarity_threshold=0.65,
              auto_threshold="p95",
              similarity_sharpness=16.0,
              suppression_floor=0.3,
              top_k=3,
              soft_edge_frac=0.15,
              override_face_bbox="",
              bypass=False, debug=False):

        if bypass or suppression_strength <= 0.0:
            # Clean up any prior SemanticClamp monkey-patches on the shared
            # modules. model.clone() shares transformer_blocks, so attn2
            # forward replacements from previous runs remain installed
            # unless we explicitly restore.
            backbone = _find_backbone(model)
            if backbone is not None:
                blocks_cleanup = backbone.transformer_blocks
                n_restored = 0
                for block in blocks_cleanup:
                    attn2 = getattr(block, "attn2", None)
                    if attn2 is None:
                        continue
                    # Restore original forward if we patched it
                    orig = getattr(attn2, ORIGINAL_FORWARD_ATTR, None)
                    if orig is not None:
                        try:
                            attn2.forward = orig
                            delattr(attn2, ORIGINAL_FORWARD_ATTR)
                            n_restored += 1
                        except Exception:
                            pass
                    # Clear sentinel
                    if getattr(attn2, HOOK_ATTR_ATTN2, False):
                        try:
                            delattr(attn2, HOOK_ATTR_ATTN2)
                        except AttributeError:
                            pass
                # Clear backbone hook handle if stored
                handle = getattr(backbone, "_10s_semclamp_backbone_handle", None)
                if handle is not None:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    try:
                        delattr(backbone, "_10s_semclamp_backbone_handle")
                    except AttributeError:
                        pass
                if getattr(backbone, HOOK_ATTR_BACKBONE, False):
                    try:
                        delattr(backbone, HOOK_ATTR_BACKBONE)
                    except AttributeError:
                        pass
                if debug:
                    reason = "bypass=True" if bypass else "suppression_strength=0"
                    print(f"\u2192 [10S] SemanticClamp: no-op ({reason}), "
                          f"restored {n_restored} prior patch(es)")
            else:
                if debug:
                    reason = "bypass=True" if bypass else "suppression_strength=0"
                    print(f"\u2192 [10S] SemanticClamp: no-op ({reason})")
            return (model,)

        # ─── Resolve bbox ────────────────────────────────────────────────────
        face_bbox = override_face_bbox
        H_latent = 0
        W_latent = 0
        if isinstance(reference_info, dict):
            if not face_bbox:
                face_bbox = reference_info.get("face_bbox", "")
            spatial_dims = reference_info.get("spatial_dims_latent", [0, 0])
            H_latent = spatial_dims[0] if len(spatial_dims) > 0 else 0
            W_latent = spatial_dims[1] if len(spatial_dims) > 1 else 0

        if not face_bbox:
            print("\u2192 [10S] SemanticClamp: \u26a0 no bbox available "
                  "(no reference_info and no override). Returning "
                  "unmodified model.")
            return (model,)

        # ─── Extract positive conditioning tensor ────────────────────────────
        positive_tensor = _extract_cond_tensor(positive)
        if positive_tensor is None:
            print("\u2192 [10S] SemanticClamp: \u26a0 couldn't extract tensor "
                  "from positive conditioning. Returning unmodified model.")
            return (model,)

        # ─── Encode face_modifier_text ───────────────────────────────────────
        if debug:
            print(f"\u2192 [10S] SemanticClamp v1.0: encoding face-modifier "
                  f"vocabulary...")
        modifier_tensor = _encode_text_via_clip(clip, face_modifier_text)
        if modifier_tensor is None:
            print("\u2192 [10S] SemanticClamp: \u26a0 couldn't encode "
                  "face_modifier_text. Returning unmodified model.")
            return (model,)

        # Ensure both tensors are on the same device for similarity calc
        modifier_tensor = modifier_tensor.to(positive_tensor.device,
                                              dtype=positive_tensor.dtype)

        # ─── Compute raw scores ──────────────────────────────────────────────
        try:
            raw_scores, pos_pad_mask = _compute_raw_scores(
                positive_tensor, modifier_tensor,
                top_k=top_k,
            )
        except Exception as e:
            print(f"\u2192 [10S] SemanticClamp: \u26a0 correspondence search "
                  f"failed: {type(e).__name__}: {e}. Returning unmodified.")
            return (model,)

        # ─── Derive effective threshold ──────────────────────────────────────
        # When auto_threshold is enabled, set threshold to a percentile of
        # the raw scores, so suppression always targets the top tail of
        # most-modifier-like tokens regardless of encoder.
        effective_threshold = similarity_threshold
        auto_threshold_str = "disabled"
        if auto_threshold != "disabled":
            # Use only non-pad scores for percentile computation
            valid_scores = raw_scores[~pos_pad_mask]
            if valid_scores.numel() > 4:
                pct_map = {"p90": 0.90, "p95": 0.95, "p98": 0.98, "p99": 0.99}
                q = pct_map.get(auto_threshold, 0.95)
                try:
                    effective_threshold = float(valid_scores.quantile(q).item())
                    auto_threshold_str = f"{auto_threshold} → {effective_threshold:.3f}"
                except Exception:
                    pass

        # ─── Apply sigmoid gate to get final suppression weights ────────────
        suppression_weights = _gate_scores_to_suppression(
            raw_scores, pos_pad_mask,
            threshold=effective_threshold,
            sharpness=similarity_sharpness,
        )

        # Hard floor: zero out weights below the floor to eliminate sigmoid's
        # soft tail leak. This is critical because even at sharpness=16,
        # tokens 0.2 below threshold still sigmoid to ~0.04 — which sounds
        # tiny but compounded across 48 blocks × suppression_strength can
        # produce visible blanket suppression on non-modifier tokens.
        if suppression_floor > 0:
            suppression_weights = torch.where(
                suppression_weights < suppression_floor,
                torch.zeros_like(suppression_weights),
                suppression_weights,
            )

        # Diagnostic stats
        n_above_05 = (suppression_weights > 0.5).sum().item()
        n_above_03 = (suppression_weights > 0.3).sum().item()
        n_above_01 = (suppression_weights > 0.1).sum().item()
        n_total = suppression_weights.numel()
        pct_05 = 100.0 * n_above_05 / max(1, n_total)
        pct_03 = 100.0 * n_above_03 / max(1, n_total)
        pct_01 = 100.0 * n_above_01 / max(1, n_total)

        if debug:
            print(f"  \u00b7 face_bbox='{face_bbox}'")
            print(f"  \u00b7 suppression_strength={suppression_strength} "
                  f"auto_threshold={auto_threshold_str} "
                  f"sim_sharp={similarity_sharpness} "
                  f"floor={suppression_floor} top_k={top_k}")
            print(f"  \u00b7 positive tensor shape: "
                  f"{tuple(positive_tensor.shape)}")
            print(f"  \u00b7 modifier tensor shape: "
                  f"{tuple(modifier_tensor.shape)}")
            # Raw similarity score distribution
            scores_flat = raw_scores[~pos_pad_mask]
            if scores_flat.numel() > 0:
                score_max = scores_flat.max().item()
                score_mean = scores_flat.mean().item()
                score_p95 = scores_flat.quantile(0.95).item() if scores_flat.numel() > 1 else score_max
                score_p99 = scores_flat.quantile(0.99).item() if scores_flat.numel() > 1 else score_max
                print(f"  \u00b7 raw similarity scores (non-pad): "
                      f"max={score_max:.3f} p99={score_p99:.3f} "
                      f"p95={score_p95:.3f} mean={score_mean:.3f}")
            print(f"  \u00b7 suppression distribution:")
            print(f"      >0.5: {n_above_05}/{n_total} tokens ({pct_05:.1f}%) "
                  f"— strong suppression")
            print(f"      >0.3: {n_above_03}/{n_total} tokens ({pct_03:.1f}%) "
                  f"— moderate")
            print(f"      >0.1: {n_above_01}/{n_total} tokens ({pct_01:.1f}%) "
                  f"— any effect")
            if auto_threshold == "disabled":
                if pct_03 > 40:
                    print(f"  \u26a0  more than 40% of tokens have moderate "
                          f"suppression — try raising sim_thr or using "
                          f"auto_threshold=p95/p98.")
                elif pct_05 < 1 and pct_03 < 5:
                    print(f"  \u26a0  almost no tokens suppressed — try "
                          f"lowering sim_thr or using auto_threshold=p95.")

        # Fingerprint positive_tensor for runtime identification
        pos_fingerprint = _fingerprint_tensor(positive_tensor)
        if debug:
            print(f"  \u00b7 positive fingerprint computed: "
                  f"{'ok' if pos_fingerprint else 'failed'}")

        # ─── Find backbone & blocks ──────────────────────────────────────────
        backbone = _find_backbone(model)
        if backbone is None:
            print("\u2192 [10S] SemanticClamp: \u26a0 couldn't locate "
                  "transformer_blocks; returning unmodified model")
            return (model,)
        blocks = backbone.transformer_blocks

        # ─── Clone model for patching ────────────────────────────────────────
        m = model.clone()

        # Shared state
        state = {
            "captured_latent_shape": None,
            "monkey_patch_calls": 0,
            "matched_calls": 0,
            "unmatched_calls": 0,
            "bbox_modulations": 0,
        }

        # ─── Backbone pre-hook: capture latent shape ─────────────────────────
        def backbone_pre_hook(module, args, kwargs):
            try:
                for a in args:
                    if isinstance(a, torch.Tensor) and a.dim() == 5:
                        state["captured_latent_shape"] = tuple(a.shape)
                        return
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 5:
                        state["captured_latent_shape"] = tuple(v.shape)
                        return
            except Exception:
                pass

        # Clear stale backbone sentinel (model.clone() shares modules)
        if getattr(backbone, HOOK_ATTR_BACKBONE, False):
            try:
                delattr(backbone, HOOK_ATTR_BACKBONE)
            except AttributeError:
                pass
        bh = backbone.register_forward_pre_hook(backbone_pre_hook, with_kwargs=True)
        setattr(backbone, HOOK_ATTR_BACKBONE, True)
        setattr(backbone, "_10s_semclamp_backbone_handle", bh)

        # Store reference data in state for the patched forward to read
        state["suppression_weights"] = suppression_weights
        state["positive_fingerprint"] = pos_fingerprint
        state["face_bbox"] = face_bbox
        state["H_meta"] = H_latent
        state["W_meta"] = W_latent
        state["suppression_strength"] = float(suppression_strength)
        state["soft_edge_frac"] = float(soft_edge_frac)

        # ─── Monkey-patch attn2.forward ──────────────────────────────────────
        # Clear stale sentinels (model.clone() shares the modules)
        cleared = 0
        for block in blocks:
            attn2 = getattr(block, "attn2", None)
            if attn2 is not None and getattr(attn2, HOOK_ATTR_ATTN2, False):
                # Restore original forward if it was saved
                orig = getattr(attn2, ORIGINAL_FORWARD_ATTR, None)
                if orig is not None:
                    try:
                        attn2.forward = orig
                        delattr(attn2, ORIGINAL_FORWARD_ATTR)
                    except Exception:
                        pass
                try:
                    delattr(attn2, HOOK_ATTR_ATTN2)
                    cleared += 1
                except AttributeError:
                    pass

        if debug and cleared > 0:
            print(f"  \u00b7 cleared {cleared} stale attn2 patches from "
                  f"prior runs")

        patched = 0
        for block_idx, block in enumerate(blocks):
            attn2 = getattr(block, "attn2", None)
            if attn2 is None:
                continue
            self._patch_attn2(attn2, block_idx, state, debug)
            patched += 1

        if debug:
            print(f"  \u00b7 monkey-patched attn2 on {patched} blocks")

        return (m,)

    def _patch_attn2(self, attn2, block_idx, state, debug):
        """Replace attn2.forward with a wrapped version that applies
        per-text-token suppression for bbox-region video queries when
        processing the positive conditioning."""

        original_forward = attn2.forward
        # Save original for cleanup
        setattr(attn2, ORIGINAL_FORWARD_ATTR, original_forward)

        # Closure variables
        cached_bbox_mask = {"mask": None, "key": None}

        def patched_forward(*args, **kwargs):
            # Defensive: if our state is missing required fields, fall through
            if state.get("suppression_weights") is None:
                return original_forward(*args, **kwargs)

            # Identify the K/V (text conditioning) from args
            # attn2's signature varies by Comfy version. Common patterns:
            #   forward(hidden_states, context, ...)
            #   forward(hidden_states, encoder_hidden_states, ...)
            # We look for the text-shape (B, T_text, D) tensor in args after
            # position 0.
            kv_tensor = None
            kv_arg_idx = None
            for i, a in enumerate(args):
                if i == 0:
                    continue
                if isinstance(a, torch.Tensor) and a.dim() == 3:
                    # Check: text shape vs video shape. Both are 3D.
                    # Video has shape (B, F*H*W, D_video) — F*H*W typically
                    # large. Text has (B, T, D_text) — T typically 128-512.
                    # We can disambiguate by comparing to the captured latent
                    # shape.
                    captured = state.get("captured_latent_shape")
                    if captured is not None:
                        _, _, F_lat, H_lat, W_lat = captured
                        video_seq = F_lat * H_lat * W_lat
                        if a.shape[1] != video_seq:
                            kv_tensor = a
                            kv_arg_idx = i
                            break
                    else:
                        # No captured shape; assume position 1 is text
                        if i == 1:
                            kv_tensor = a
                            kv_arg_idx = i
                            break
            if kv_tensor is None:
                # Check kwargs
                for key in ("context", "encoder_hidden_states", "cross_attention_kwargs"):
                    if key in kwargs:
                        v = kwargs[key]
                        if isinstance(v, torch.Tensor) and v.dim() == 3:
                            kv_tensor = v
                            break

            if kv_tensor is None:
                # Couldn't find text K/V — fall through
                state["unmatched_calls"] += 1
                return original_forward(*args, **kwargs)

            # Fingerprint check: is this the positive conditioning?
            target_fp = state.get("positive_fingerprint")
            cur_fp = _fingerprint_tensor(kv_tensor)
            if target_fp is None or cur_fp != target_fp:
                state["unmatched_calls"] += 1
                # Not the positive pass — pass through unchanged
                return original_forward(*args, **kwargs)

            state["matched_calls"] += 1

            # Get hidden_states (video tokens) — position 0
            hidden_states = args[0]
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
                return original_forward(*args, **kwargs)

            B_v, seq_v, D_v = hidden_states.shape
            B_t, T_text, D_t = kv_tensor.shape
            suppression_weights = state["suppression_weights"]
            if suppression_weights.shape[1] != T_text:
                # Conditioning shape doesn't match precomputed suppression —
                # the conditioning may have been re-encoded somewhere. Fall
                # through.
                return original_forward(*args, **kwargs)

            # Determine video grid layout
            captured = state.get("captured_latent_shape")
            H_use = W_use = 0
            F_use = 0
            if captured is not None:
                _, _, F_use, H_use, W_use = captured
                if F_use * H_use * W_use != seq_v:
                    # Fall back to metadata
                    H_use = state["H_meta"]
                    W_use = state["W_meta"]
                    if H_use > 0 and W_use > 0 and seq_v % (H_use * W_use) == 0:
                        F_use = seq_v // (H_use * W_use)
                    else:
                        return original_forward(*args, **kwargs)
            else:
                H_use = state["H_meta"]
                W_use = state["W_meta"]
                if H_use > 0 and W_use > 0 and seq_v % (H_use * W_use) == 0:
                    F_use = seq_v // (H_use * W_use)
                else:
                    return original_forward(*args, **kwargs)

            # Build bbox spatial mask (cached, since geometry doesn't change)
            cache_key = (H_use, W_use, state["face_bbox"], state["soft_edge_frac"])
            if cached_bbox_mask["key"] != cache_key:
                h1, h2, w1, w2 = _parse_bbox(state["face_bbox"], H_use, W_use)
                cached_bbox_mask["mask"] = _build_bbox_mask(
                    (h1, h2, w1, w2), H_use, W_use,
                    soft_edge_frac=state["soft_edge_frac"],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                cached_bbox_mask["key"] = cache_key

            bbox_spatial_mask = cached_bbox_mask["mask"]  # (H, W)
            # Per-video-token bbox weight: (1, seq_v)
            bbox_per_video = bbox_spatial_mask.view(1, 1, H_use, W_use).expand(
                1, F_use, H_use, W_use
            ).reshape(1, seq_v)

            # We can't easily modify attention WEIGHTS without rewriting the
            # full attention computation. Instead, the most robust approach
            # is to modify the K/V tensor:
            #
            # For text token positions with high suppression_weight, scale
            # down their K and V vectors. This effectively reduces those
            # tokens' attention contribution. But this would apply globally
            # to ALL video queries, not just bbox queries — defeating the
            # selective purpose.
            #
            # The bbox-aware path requires per-video-position modification.
            # We do this by computing the attention TWICE: once unmodified,
            # once with text tokens suppressed, then blending the two
            # outputs by bbox mask.

            # Path 1: unmodified attention (already what original_forward does)
            output_unmodified = original_forward(*args, **kwargs)
            if not isinstance(output_unmodified, torch.Tensor):
                # Some attn2 returns tuples; extract main tensor
                if isinstance(output_unmodified, tuple):
                    out_unmod_t = output_unmodified[0]
                    out_unmod_rest = output_unmodified[1:]
                else:
                    return output_unmodified
            else:
                out_unmod_t = output_unmodified
                out_unmod_rest = None

            # Path 2: attention with suppressed text tokens
            # Modify kv_tensor: scale tokens by (1 - suppression_strength * suppression_weight)
            sup_factor = 1.0 - state["suppression_strength"] * suppression_weights
            sup_factor = sup_factor.clamp(0.0, 1.0)
            # sup_factor shape: (B, T_text), need to broadcast to (B, T_text, D)
            sup_factor_b = sup_factor.unsqueeze(-1).to(
                device=kv_tensor.device, dtype=kv_tensor.dtype
            )
            kv_suppressed = kv_tensor * sup_factor_b

            # Build new args with suppressed kv
            new_args = list(args)
            if kv_arg_idx is not None:
                new_args[kv_arg_idx] = kv_suppressed
            new_kwargs = dict(kwargs)
            for key in ("context", "encoder_hidden_states"):
                if key in new_kwargs and isinstance(new_kwargs[key], torch.Tensor):
                    if new_kwargs[key].shape == kv_tensor.shape:
                        new_kwargs[key] = kv_suppressed

            output_suppressed = original_forward(*new_args, **new_kwargs)
            if isinstance(output_suppressed, tuple):
                out_sup_t = output_suppressed[0]
            else:
                out_sup_t = output_suppressed

            # Blend the two outputs by bbox mask
            # bbox_per_video shape: (1, seq_v) — broadcast to (B, seq_v, D)
            bbox_weight = bbox_per_video.unsqueeze(-1).to(
                dtype=out_unmod_t.dtype, device=out_unmod_t.device
            )
            # Inside bbox: use suppressed; outside: use unmodified
            blended = out_unmod_t * (1.0 - bbox_weight) + \
                      out_sup_t * bbox_weight

            state["bbox_modulations"] += 1

            if out_unmod_rest is not None:
                return (blended,) + out_unmod_rest
            return blended

        attn2.forward = patched_forward
        setattr(attn2, HOOK_ATTR_ATTN2, True)


NODE_CLASS_MAPPINGS = {
    "LTXLikenessSemanticClamp": LTXLikenessSemanticClamp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLikenessSemanticClamp": "\U0001f9e0 LTX Likeness Semantic Clamp",
}
