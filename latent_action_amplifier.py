"""
LTX Action Amplifier v1.0

Per-block intervention on text cross-attention (attn2) that selectively
amplifies action / motion verb tokens in the positive prompt, making i2v
output more responsive to verb-driven motion directives without globally
boosting the prompt (which is the failure mode of blanket amplification).

================================================================================
WHAT THIS SOLVES
================================================================================

In enhanced i2v prompts, verb/action tokens ("turning", "walking",
"reaching", "tilting") often get diluted in a dense lattice of scene
description, lighting cues, mood adjectives, and camera notes. Even
though the prompt clearly specifies motion, the model's attention to
those specific tokens is a small slice of total text attention.

This node identifies which positive-prompt tokens semantically correspond
to a motion vocabulary, then slightly scales up their K/V magnitudes at
attn2. Verb tokens compete more strongly for attention mass; surrounding
descriptive tokens still contribute, but proportionally less.

Crucially: SELECTIVE, not blanket. Only matched tokens get boosted. The
deprecated TextAmplifier scaled the whole conditioning uniformly and
produced noise; this targets specific positions.

================================================================================
ARCHITECTURE — symmetric mirror of LikenessSemanticClamp
================================================================================

Setup (apply time):
  1. Encode action_vocabulary_text via CLIP
  2. Compute top-K-mean cosine similarity per positive-prompt token to the
     vocabulary
  3. Apply auto_threshold (p95 default) to derive per-token boost weights
  4. Hard floor (default 0.3) eliminates sigmoid soft-tail leak

Runtime (per attn2 call):
  1. Fingerprint-match incoming K/V against the positive conditioning we
     analyzed (so we don't boost on the uncond/negative pass)
  2. If matched: scale K and V by (1 + amplification_strength * weight *
     scale_ceiling) for matched token positions
  3. Single attention pass with modified K/V — uniform across all video
     tokens (no bbox, since actions affect the whole frame)
  4. If not matched: pass through unchanged

================================================================================
USAGE
================================================================================

    LatentActionAmplifier(model, clip, positive, amplification_strength=0.3)
                                  ↓
    KSampler...

Recommended starting config:
    amplification_strength = 0.3      gentle boost
    scale_ceiling          = 0.30     max +30% K/V scaling per matched token
    auto_threshold         = p95      adaptive to encoder
    suppression_floor      = 0.3      hard cutoff

================================================================================
KEY DESIGN CHOICES
================================================================================

- BOOST CEILING IS CAPPED. Scaling K/V by (1 + strength * weight) without
  a ceiling would max at 2.0x at strength=1.0. That's too aggressive — the
  softmax over text tokens turns 2x K into much-more-than-2x attention-mass
  redistribution. The scale_ceiling parameter (default 0.30) limits the
  multiplicative factor to [1.0, 1.30], giving "controlled slight upward
  scaling" rather than aggressive amplification.

- SINGLE ATTENTION PASS. Unlike SemanticClamp which uses dual-attention-
  blend (one normal + one modified, blended by bbox mask), this amplifier
  applies uniformly across video tokens — no spatial scoping needed since
  actions affect the whole frame. Single pass = ~2x faster than the bbox-
  aware approach.

- NO BBOX REQUIRED. Action amplification operates on text-side scaling
  only. The boost applies to text-token K/V before attention is computed,
  so all video queries see the boosted text.
"""

import torch
import torch.nn.functional as F


# Sentinels for hook coexistence
HOOK_ATTR_ATTN2 = "_10s_action_amp_attn2_hook"
ORIGINAL_FORWARD_ATTR = "_10s_action_amp_orig_forward"
HOOK_ATTR_BACKBONE = "_10s_action_amp_backbone"


# Default action / motion vocabulary — broad enough to catch common verb
# tenses and motion-related noun phrases, focused enough to avoid catching
# state-of-being or non-motion descriptors.
DEFAULT_ACTION_VOCABULARY = (
    "moving, walking, running, turning, lifting, reaching, "
    "pushing, pulling, gesturing, waving, pointing, "
    "leaning, tilting, raising, lowering, bending, "
    "stepping forward, walking toward, turning around, "
    "looking around, glancing, pivoting, rotating, "
    "jumping, dancing, swaying, rocking, "
    "extending, retracting, opening, closing, "
    "action, motion, movement, gesture"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (reused / adapted from semantic_clamp)
# ─────────────────────────────────────────────────────────────────────────────

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
    if not conditioning:
        return None
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            t = entry[0]
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                return t
    return None


def _encode_text_via_clip(clip, text):
    if not text or not text.strip():
        return None
    try:
        tokens = clip.tokenize(text)
        cond_out = clip.encode_from_tokens_scheduled(tokens)
        return _extract_cond_tensor(cond_out)
    except Exception as e:
        print(f"\u2192 [10S] ActionAmplifier: text encode failed: "
              f"{type(e).__name__}: {e}")
        return None


def _compute_token_magnitudes(cond_tensor):
    return cond_tensor.norm(dim=-1)


def _compute_raw_scores(positive_tensor, vocabulary_tensor,
                         pad_magnitude_threshold_frac=0.3,
                         top_k=3):
    """Per-positive-token: top-K mean cosine similarity to vocabulary
    tensor, with pad-filtering. Returns (scores, pos_pad_mask)."""
    pos_mean = positive_tensor.mean(dim=1, keepdim=True)
    pos_centered = positive_tensor - pos_mean
    voc_mean = vocabulary_tensor.mean(dim=1, keepdim=True)
    voc_centered = vocabulary_tensor - voc_mean

    pos_norm = F.normalize(pos_centered, dim=-1, eps=1e-6)
    voc_norm = F.normalize(voc_centered, dim=-1, eps=1e-6)

    sim = torch.einsum("btd,bsd->bts", pos_norm, voc_norm)

    pos_mag = _compute_token_magnitudes(positive_tensor)
    voc_mag = _compute_token_magnitudes(vocabulary_tensor)
    pos_mag_max = pos_mag.max(dim=1, keepdim=True).values
    voc_mag_max = voc_mag.max(dim=1, keepdim=True).values
    pos_pad_mask = pos_mag < (pad_magnitude_threshold_frac * pos_mag_max)
    voc_pad_mask = voc_mag < (pad_magnitude_threshold_frac * voc_mag_max)

    sim_masked = sim.clone()
    sim_masked[voc_pad_mask.unsqueeze(1).expand_as(sim)] = -1.0

    n_voc_valid = (~voc_pad_mask).sum(dim=1).min().item()
    k = min(top_k, max(1, n_voc_valid))
    top_k_sims = sim_masked.topk(k, dim=-1).values
    scores = top_k_sims.mean(dim=-1)

    return scores, pos_pad_mask


def _gate_scores_to_weights(scores, pos_pad_mask, threshold, sharpness):
    weights = torch.sigmoid((scores - threshold) * sharpness)
    weights[pos_pad_mask] = 0.0
    return weights


def _fingerprint_tensor(t):
    if not isinstance(t, torch.Tensor):
        return None
    try:
        shape = tuple(t.shape)
        flat = t.flatten().detach().cpu()
        if flat.numel() < 16:
            return (shape, tuple(flat.tolist()))
        idxs = torch.linspace(0, flat.numel() - 1, 16, dtype=torch.long)
        vals = flat[idxs].to(torch.float32).tolist()
        vals_rounded = tuple(round(v, 4) for v in vals)
        return (shape, vals_rounded)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTXActionAmplifier:
    """
    Selectively amplify action/motion text tokens in attn2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX2 model to patch. Chain order is "
                               "flexible — does not conflict with anchor "
                               "nodes (attn1) or SemanticClamp (also attn2 "
                               "but different sentinel).",
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP / text encoder that produced your "
                               "positive conditioning. Required to encode "
                               "the action vocabulary for correspondence "
                               "search.",
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive prompt conditioning. The node "
                               "analyzes this and fingerprint-matches at "
                               "runtime so only the cond pass is amplified.",
                }),
                "amplification_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How aggressively to boost matched action "
                               "tokens. 0 = no-op. 0.3 (default) = gentle "
                               "upward scaling. 0.5-0.7 = stronger. 1.0 = "
                               "maximum (matched tokens scaled by "
                               "1 + scale_ceiling, default +30%).",
                }),
            },
            "optional": {
                "action_vocabulary_text": ("STRING", {
                    "default": DEFAULT_ACTION_VOCABULARY,
                    "multiline": True,
                    "tooltip": "Comma-separated action / motion vocabulary. "
                               "Encoded via CLIP, used as the correspondence "
                               "target. Default covers common verbs and "
                               "phrases. Customize if your domain has "
                               "specific motion vocabulary (sports, dance, "
                               "etc.).",
                }),
                "scale_ceiling": ("FLOAT", {
                    "default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Maximum K/V scale factor delta. At "
                               "amplification_strength=1.0 and weight=1.0, "
                               "K/V are scaled by (1 + scale_ceiling). "
                               "Default 0.30 = max +30%. Higher values "
                               "produce stronger redistribution but risk "
                               "the same noise pattern as the deprecated "
                               "TextAmplifier.",
                }),
                "auto_threshold": (
                    ["disabled", "p90", "p95", "p98", "p99"],
                    {
                        "default": "p95",
                        "tooltip": "Auto-calibrate correspondence threshold "
                                   "to encoder's score distribution. p95 "
                                   "targets top ~5% of most action-like "
                                   "tokens. p98 = top 2%. Encoder-agnostic "
                                   "selection.",
                    },
                ),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Literal threshold when auto_threshold="
                               "disabled. Otherwise overridden by percentile.",
                }),
                "similarity_sharpness": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Sigmoid steepness. 16 (default) = near-"
                               "binary partition. Lower = softer transition.",
                }),
                "amplification_floor": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Hard floor on per-token weights. Below this, "
                               "weights become 0 (no boost). Eliminates "
                               "sigmoid soft-tail leak. 0.3 default cuts "
                               "off weak matches.",
                }),
                "top_k": ("INT", {
                    "default": 3, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of top similarity values averaged "
                               "per positive token. Higher = requires more "
                               "confirming matches in vocabulary.",
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, model passes through unchanged. "
                               "Also restores any prior amplifier patches.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "10S Nodes/Conditioning"
    DESCRIPTION = (
        "Selectively amplify action / motion text tokens in the positive "
        "prompt at attn2. Boosts K/V magnitude for matched verb tokens, "
        "leaving descriptive tokens unchanged. Symmetric inverse of "
        "LikenessSemanticClamp; replaces deprecated TextAmplifier's "
        "blanket-scaling approach."
    )

    def apply(self, model, clip, positive, amplification_strength,
              action_vocabulary_text=DEFAULT_ACTION_VOCABULARY,
              scale_ceiling=0.30,
              auto_threshold="p95",
              similarity_threshold=0.55,
              similarity_sharpness=16.0,
              amplification_floor=0.3,
              top_k=3,
              bypass=False, debug=False):

        if bypass or amplification_strength <= 0.0:
            # Cleanup any prior patches (same pattern as SemanticClamp)
            backbone = _find_backbone(model)
            if backbone is not None:
                blocks_cleanup = backbone.transformer_blocks
                n_restored = 0
                for block in blocks_cleanup:
                    attn2 = getattr(block, "attn2", None)
                    if attn2 is None:
                        continue
                    orig = getattr(attn2, ORIGINAL_FORWARD_ATTR, None)
                    if orig is not None:
                        try:
                            attn2.forward = orig
                            delattr(attn2, ORIGINAL_FORWARD_ATTR)
                            n_restored += 1
                        except Exception:
                            pass
                    if getattr(attn2, HOOK_ATTR_ATTN2, False):
                        try:
                            delattr(attn2, HOOK_ATTR_ATTN2)
                        except AttributeError:
                            pass
                handle = getattr(backbone, "_10s_actamp_backbone_handle", None)
                if handle is not None:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    try:
                        delattr(backbone, "_10s_actamp_backbone_handle")
                    except AttributeError:
                        pass
                if getattr(backbone, HOOK_ATTR_BACKBONE, False):
                    try:
                        delattr(backbone, HOOK_ATTR_BACKBONE)
                    except AttributeError:
                        pass
                if debug:
                    reason = "bypass" if bypass else "strength=0"
                    print(f"\u2192 [10S] ActionAmplifier: no-op ({reason}), "
                          f"restored {n_restored} prior patch(es)")
            return (model,)

        # ─── Extract positive conditioning ──────────────────────────────────
        positive_tensor = _extract_cond_tensor(positive)
        if positive_tensor is None:
            print("\u2192 [10S] ActionAmplifier: \u26a0 couldn't extract "
                  "tensor from positive conditioning.")
            return (model,)

        # ─── Encode vocabulary ──────────────────────────────────────────────
        if debug:
            print(f"\u2192 [10S] ActionAmplifier v1.0: encoding action "
                  f"vocabulary...")
        vocabulary_tensor = _encode_text_via_clip(clip, action_vocabulary_text)
        if vocabulary_tensor is None:
            print("\u2192 [10S] ActionAmplifier: \u26a0 couldn't encode "
                  "action vocabulary.")
            return (model,)
        vocabulary_tensor = vocabulary_tensor.to(positive_tensor.device,
                                                   dtype=positive_tensor.dtype)

        # ─── Compute raw scores ─────────────────────────────────────────────
        try:
            raw_scores, pos_pad_mask = _compute_raw_scores(
                positive_tensor, vocabulary_tensor, top_k=top_k,
            )
        except Exception as e:
            print(f"\u2192 [10S] ActionAmplifier: \u26a0 correspondence "
                  f"search failed: {type(e).__name__}: {e}.")
            return (model,)

        # ─── Derive effective threshold ─────────────────────────────────────
        effective_threshold = similarity_threshold
        auto_threshold_str = "disabled"
        if auto_threshold != "disabled":
            valid_scores = raw_scores[~pos_pad_mask]
            if valid_scores.numel() > 4:
                pct_map = {"p90": 0.90, "p95": 0.95,
                           "p98": 0.98, "p99": 0.99}
                q = pct_map.get(auto_threshold, 0.95)
                try:
                    effective_threshold = float(valid_scores.quantile(q).item())
                    auto_threshold_str = (f"{auto_threshold} → "
                                          f"{effective_threshold:.3f}")
                except Exception:
                    pass

        # ─── Apply sigmoid gate + floor ─────────────────────────────────────
        amp_weights = _gate_scores_to_weights(
            raw_scores, pos_pad_mask,
            threshold=effective_threshold,
            sharpness=similarity_sharpness,
        )
        if amplification_floor > 0:
            amp_weights = torch.where(
                amp_weights < amplification_floor,
                torch.zeros_like(amp_weights),
                amp_weights,
            )

        # Diagnostic
        n_above_05 = (amp_weights > 0.5).sum().item()
        n_above_03 = (amp_weights > 0.3).sum().item()
        n_above_01 = (amp_weights > 0.1).sum().item()
        n_total = amp_weights.numel()

        if debug:
            print(f"  \u00b7 amplification_strength={amplification_strength} "
                  f"scale_ceiling={scale_ceiling} "
                  f"auto_threshold={auto_threshold_str} "
                  f"floor={amplification_floor}")
            print(f"  \u00b7 positive tensor: {tuple(positive_tensor.shape)}")
            print(f"  \u00b7 vocabulary tensor: "
                  f"{tuple(vocabulary_tensor.shape)}")
            scores_flat = raw_scores[~pos_pad_mask]
            if scores_flat.numel() > 0:
                print(f"  \u00b7 raw scores: "
                      f"max={scores_flat.max().item():.3f} "
                      f"p99={scores_flat.quantile(0.99).item():.3f} "
                      f"p95={scores_flat.quantile(0.95).item():.3f} "
                      f"mean={scores_flat.mean().item():.3f}")
            # Max effective boost = strength * weight_max * scale_ceiling
            max_w = amp_weights.max().item()
            max_boost = amplification_strength * max_w * scale_ceiling
            print(f"  \u00b7 boost distribution:")
            print(f"      >0.5: {n_above_05}/{n_total} tokens "
                  f"({100*n_above_05/max(1,n_total):.1f}%) — strong boost")
            print(f"      >0.3: {n_above_03}/{n_total} tokens "
                  f"({100*n_above_03/max(1,n_total):.1f}%) — moderate")
            print(f"      >0.1: {n_above_01}/{n_total} tokens "
                  f"({100*n_above_01/max(1,n_total):.1f}%) — any effect")
            print(f"  \u00b7 max effective scale factor on any token: "
                  f"{1.0 + max_boost:.3f}x")

        pos_fingerprint = _fingerprint_tensor(positive_tensor)

        # ─── Find backbone ──────────────────────────────────────────────────
        backbone = _find_backbone(model)
        if backbone is None:
            print("\u2192 [10S] ActionAmplifier: \u26a0 couldn't locate "
                  "transformer_blocks.")
            return (model,)
        blocks = backbone.transformer_blocks

        # Clone model
        m = model.clone()

        # Shared state
        state = {
            "captured_latent_shape": None,
            "amp_weights": amp_weights,
            "positive_fingerprint": pos_fingerprint,
            "amplification_strength": float(amplification_strength),
            "scale_ceiling": float(scale_ceiling),
            "matched_calls": 0,
            "unmatched_calls": 0,
        }

        # ─── Backbone pre-hook (capture latent shape) ───────────────────────
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

        # Clean stale backbone sentinel
        if getattr(backbone, HOOK_ATTR_BACKBONE, False):
            try:
                delattr(backbone, HOOK_ATTR_BACKBONE)
            except AttributeError:
                pass
        bh = backbone.register_forward_pre_hook(
            backbone_pre_hook, with_kwargs=True
        )
        setattr(backbone, HOOK_ATTR_BACKBONE, True)
        setattr(backbone, "_10s_actamp_backbone_handle", bh)

        # ─── Patch attn2 ────────────────────────────────────────────────────
        # Clear stale patches
        cleared = 0
        for block in blocks:
            attn2 = getattr(block, "attn2", None)
            if attn2 is not None and getattr(attn2, HOOK_ATTR_ATTN2, False):
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
            print(f"  \u00b7 cleared {cleared} stale patches")

        patched = 0
        for block in blocks:
            attn2 = getattr(block, "attn2", None)
            if attn2 is None:
                continue
            self._patch_attn2(attn2, state)
            patched += 1

        if debug:
            print(f"  \u00b7 monkey-patched attn2 on {patched} blocks")

        return (m,)

    def _patch_attn2(self, attn2, state):
        """Replace attn2.forward with a wrapped version that scales K/V
        for matched text tokens before the standard attention math runs.
        Single-pass — no dual-attention-blend needed since amplification
        applies uniformly across video tokens."""

        original_forward = attn2.forward
        setattr(attn2, ORIGINAL_FORWARD_ATTR, original_forward)

        def patched_forward(*args, **kwargs):
            amp_weights = state.get("amp_weights")
            if amp_weights is None:
                return original_forward(*args, **kwargs)

            # Identify text K/V tensor
            kv_tensor = None
            kv_arg_idx = None
            for i, a in enumerate(args):
                if i == 0:
                    continue
                if isinstance(a, torch.Tensor) and a.dim() == 3:
                    captured = state.get("captured_latent_shape")
                    if captured is not None:
                        _, _, F_lat, H_lat, W_lat = captured
                        video_seq = F_lat * H_lat * W_lat
                        if a.shape[1] != video_seq:
                            kv_tensor = a
                            kv_arg_idx = i
                            break
                    else:
                        if i == 1:
                            kv_tensor = a
                            kv_arg_idx = i
                            break
            if kv_tensor is None:
                for key in ("context", "encoder_hidden_states",
                            "cross_attention_kwargs"):
                    if key in kwargs:
                        v = kwargs[key]
                        if isinstance(v, torch.Tensor) and v.dim() == 3:
                            kv_tensor = v
                            break

            if kv_tensor is None:
                state["unmatched_calls"] += 1
                return original_forward(*args, **kwargs)

            # Fingerprint check: is this the positive pass?
            target_fp = state.get("positive_fingerprint")
            cur_fp = _fingerprint_tensor(kv_tensor)
            if target_fp is None or cur_fp != target_fp:
                state["unmatched_calls"] += 1
                return original_forward(*args, **kwargs)

            state["matched_calls"] += 1

            if amp_weights.shape[1] != kv_tensor.shape[1]:
                # Shape mismatch — fall through
                return original_forward(*args, **kwargs)

            # Build per-token scale factor: (1 + strength * weight * ceiling)
            # Shape: (B, T_text) → (B, T_text, 1) for broadcast over D
            scale_factor = 1.0 + (
                state["amplification_strength"]
                * amp_weights
                * state["scale_ceiling"]
            )
            scale_factor_b = scale_factor.unsqueeze(-1).to(
                device=kv_tensor.device, dtype=kv_tensor.dtype
            )

            kv_amplified = kv_tensor * scale_factor_b

            # Single attention pass with amplified K/V
            new_args = list(args)
            if kv_arg_idx is not None:
                new_args[kv_arg_idx] = kv_amplified
            new_kwargs = dict(kwargs)
            for key in ("context", "encoder_hidden_states"):
                if key in new_kwargs and \
                        isinstance(new_kwargs[key], torch.Tensor):
                    if new_kwargs[key].shape == kv_tensor.shape:
                        new_kwargs[key] = kv_amplified

            return original_forward(*new_args, **new_kwargs)

        attn2.forward = patched_forward
        setattr(attn2, HOOK_ATTR_ATTN2, True)


NODE_CLASS_MAPPINGS = {
    "LTXActionAmplifier": LTXActionAmplifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXActionAmplifier": "\U0001f4a8 LTX Action Amplifier",
}
