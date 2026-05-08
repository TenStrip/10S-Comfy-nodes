"""
LTX Text Attention Amplifier v1.0

Amplifies the influence of text cross-attention (attn2) per block in LTX2.x
during sampling. Designed to compensate for conditioning dilution in
upscale-pass sampling: when the spatial token count grows (e.g. 4x at 2x
upscale), each video token receives proportionally less text influence per
attention pass, manifesting as conditioning drift (hue shifts, prompt
adherence loss).

================================================================================
DIAGNOSIS THIS ADDRESSES
================================================================================
Empirical finding from extensive testing: a second sampling pass on an
upscaled latent produces output drift (broad hue shifts, weakened prompt
adherence) even when the upscaled latent itself is clean (verified by
direct VAE decode). The drift is induced by the sampler operating on a
larger token count than the model was trained at, where attention
dilutes conditioning influence per-token.

This node compensates by amplifying text cross-attention output at each
block. Effective text influence per token is restored toward training-
distribution levels.

================================================================================
HOW IT WORKS
================================================================================
For each transformer block, hooks block.attn2 (the text cross-attention).
On the hook output, multiplies the per-token attention output tensor by
an amplification factor. The result enters the residual stream with
boosted text-conditioning influence.

  Block forward (simplified):
    x = x + gate_attn1 * attn1(norm(x))
    x = x + gate_attn2 * attn2(norm(x), text_context)   [HOOKED]
    x = x + gate_av    * av_cross_attn(...)
    x = x + gate_ff    * ff(norm(x))

  With our hook:
    attn2_out_amplified = attn2_out * amp_factor
    x = x + gate_attn2 * attn2_out_amplified

The amplification compounds across all 48 blocks per sampling step, so
moderate values (1.2-1.5) are usually sufficient. Higher values risk
over-baked / over-prompted output.

================================================================================
PARAMETERS
================================================================================
  text_amplification : Multiplier for attn2 output. 1.0 = no change.
                       1.2-1.5 typical for upscale-pass dilution recovery.
                       2.0+ likely produces over-baked output.
                       Default 1.3.

  spatial_focus      : 0.0 = uniform amplification across all spatial
                       positions (default, recommended for global drift).
                       > 0.0 = Gaussian-weighted with more amplification
                       at the spatial center, tapering to no amplification
                       at edges. Use when you want the model to stay
                       within its "comfort zone" centrally and let edges
                       behave naturally.

  block_index_filter : Block range filter. e.g. "10-30". Empty = all 48
                       blocks. Limit blocks for finer control over where
                       amplification applies in the stack depth.

  bypass             : Clear hooks, return model unmodified.

  debug              : Verbose logging.

================================================================================
RECOMMENDED CONFIGURATIONS
================================================================================

Light upscale-pass dilution recovery (default):
  text_amplification = 1.3
  spatial_focus      = 0.0

Stronger conditioning fidelity:
  text_amplification = 1.5
  spatial_focus      = 0.0

Center-focused (model "comfort zone" experiment):
  text_amplification = 1.5
  spatial_focus      = 0.6

================================================================================
COMPATIBILITY
================================================================================
Coexists with face_anchor, latent_anchor, and latent_anchor_aware nodes.
Different sentinel attributes mean all hook types can register on the
same model simultaneously. PyTorch fires hooks in registration order;
chain as needed for your workflow.
"""

import torch


# ─── Hardcoded constants (LTX2 architecture) ────────────────────────────────
SPATIAL_PATCH  = 1
TEMPORAL_PATCH = 1


# Sentinel attribute names — distinct from anchor nodes
HOOK_ATTR_BACKBONE = "_10s_text_amp_pre_hook"
HOOK_ATTR_ATTN2    = "_10s_text_amp_attn2_hook"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
    """For attn2 output: extract the lone tensor regardless of container."""
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


def _remove_prior_hooks(backbone):
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
            attn2 = getattr(block, "attn2", None)
            if attn2 is not None:
                h = getattr(attn2, HOOK_ATTR_ATTN2, None)
                if h is not None:
                    try:
                        h.remove(); removed += 1
                    except Exception:
                        pass
                    try:
                        delattr(attn2, HOOK_ATTR_ATTN2)
                    except Exception:
                        pass
    return removed


def _build_spatial_weight(H_lat, W_lat, spatial_focus, dtype, device):
    """
    Per-spatial-position amplification weight in [0, 1].
    At spatial_focus=0: returns None (caller uses uniform multiplier).
    At spatial_focus>0: Gaussian centered on (H/2, W/2), normalized to [0,1].
    """
    if spatial_focus <= 0.0:
        return None
    smaller_dim = min(H_lat, W_lat)
    # spatial_focus 0.5 -> sigma = 0.65 * smaller_dim (broad)
    # spatial_focus 1.0 -> sigma = 0.30 * smaller_dim (tight)
    sigma = max(0.3, 1.0 - 0.7 * spatial_focus) * smaller_dim
    cy = (H_lat - 1) / 2.0
    cx = (W_lat - 1) / 2.0
    y_idx = torch.arange(H_lat, dtype=torch.float32, device=device)
    x_idx = torch.arange(W_lat, dtype=torch.float32, device=device)
    dy = y_idx - cy
    dx = x_idx - cx
    dist_sq = dy.unsqueeze(1).pow(2) + dx.unsqueeze(0).pow(2)
    gaussian = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    g_min = gaussian.min()
    g_max = gaussian.max()
    gaussian = (gaussian - g_min) / (g_max - g_min + 1e-6)
    return gaussian.to(dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTXTextAttentionAmplifier:
    """
    Hooks each transformer block's attn2 (text cross-attention) and
    multiplies its output by an amplification factor. Compensates for
    conditioning dilution at upscaled token counts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "text_amplification":  ("FLOAT",   {"default": 1.30, "min": 1.0, "max": 3.0,  "step": 0.01}),
                "spatial_focus":       ("FLOAT",   {"default": 0.0,  "min": 0.0, "max": 1.0,  "step": 0.05}),
                "block_index_filter":  ("STRING",  {"default": ""}),
                "bypass":              ("BOOLEAN", {"default": False}),
                "debug":               ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Amplifies LTX2 text cross-attention (attn2) per block to compensate "
        "for conditioning dilution at upscaled token counts. Use during "
        "upscale-pass sampling to recover prompt adherence and color stability."
    )

    def patch(self, model, text_amplification=1.30, spatial_focus=0.0,
              block_index_filter="", bypass=False, debug=False):

        m = model.clone()
        backbone, _ = _resolve_diffusion_model(m)
        if backbone is None:
            print("\u2192 [10S] TextAmplifier v1.0: could not locate diffusion backbone.")
            return (m,)
        if not hasattr(backbone, "transformer_blocks"):
            print(f"\u2192 [10S] TextAmplifier v1.0: backbone {type(backbone).__name__} "
                  f"has no 'transformer_blocks'.")
            return (m,)

        n_removed = _remove_prior_hooks(backbone)
        if n_removed > 0:
            print(f"\u2192 [10S] TextAmplifier v1.0: removed {n_removed} prior hook(s)")

        if bypass or text_amplification == 1.0:
            reason = "bypass=True" if bypass else "text_amplification == 1.0"
            print(f"\u2192 [10S] TextAmplifier v1.0: {reason} \u2014 hooks cleared, "
                  f"model unmodified")
            return (m,)

        blocks = backbone.transformer_blocks
        n_blocks = len(blocks)
        idx_filter = _parse_index_filter(block_index_filter, n_blocks)

        state = {
            "latent_shape":   None,
            "shape_logged":   False,
            "hook_logged":    False,
            "calls":          0,
            "spatial_weight_cache": None,
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

        # ─── attn2 hook factory ───────────────────────────────────────────────
        def make_attn2_hook(block_idx):
            def hook(module, inputs, output):
                try:
                    if text_amplification == 1.0:
                        return None

                    tensor, wrap = _extract_attn_tensor(output)
                    if tensor is None or tensor.dim() != 3:
                        return None

                    B, seq, D = tensor.shape

                    # Uniform amplification path (no spatial weighting needed)
                    if spatial_focus <= 0.0 or state["latent_shape"] is None:
                        if not state["hook_logged"]:
                            print(f"\u2192 [10S] TextAmplifier v1.0: HOOK ACTIVE | "
                                  f"first fire on blk{block_idx} | "
                                  f"seq={seq} D={D} amp={text_amplification} "
                                  f"mode=uniform")
                            state["hook_logged"] = True
                        modified = tensor * text_amplification
                        state["calls"] += 1
                        return wrap(modified)

                    # Spatial-focus path
                    _, _, F_lat, H_lat, W_lat = state["latent_shape"]
                    F_tok = max(1, F_lat // TEMPORAL_PATCH)
                    H_tok = max(1, H_lat // SPATIAL_PATCH)
                    W_tok = max(1, W_lat // SPATIAL_PATCH)
                    if F_tok * H_tok * W_tok != seq:
                        # Shape mismatch — fall back to uniform safely
                        modified = tensor * text_amplification
                        state["calls"] += 1
                        return wrap(modified)

                    # Build spatial weight grid (cached)
                    if state["spatial_weight_cache"] is None:
                        sw = _build_spatial_weight(
                            H_tok, W_tok, spatial_focus,
                            dtype=tensor.dtype, device=tensor.device,
                        )
                        if sw is None:
                            modified = tensor * text_amplification
                            state["calls"] += 1
                            return wrap(modified)
                        state["spatial_weight_cache"] = sw

                    spatial_weight = state["spatial_weight_cache"]   # (H, W)
                    if spatial_weight.shape != (H_tok, W_tok):
                        # Latent shape changed — rebuild
                        spatial_weight = _build_spatial_weight(
                            H_tok, W_tok, spatial_focus,
                            dtype=tensor.dtype, device=tensor.device,
                        )
                        state["spatial_weight_cache"] = spatial_weight

                    # Per-token amp factor: 1 + (amp - 1) * spatial_weight
                    # Center: full amp. Edges: no amp.
                    amp_grid = 1.0 + (text_amplification - 1.0) * spatial_weight  # (H, W)
                    amp_full = amp_grid.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                        B, F_tok, H_tok, W_tok, 1
                    )                                                              # (B,F,H,W,1)

                    grid = tensor.reshape(B, F_tok, H_tok, W_tok, D)
                    modified_grid = grid * amp_full
                    modified = modified_grid.reshape(B, seq, D)

                    if not state["hook_logged"]:
                        max_amp = amp_grid.max().item()
                        min_amp = amp_grid.min().item()
                        print(f"\u2192 [10S] TextAmplifier v1.0: HOOK ACTIVE | "
                              f"first fire on blk{block_idx} | "
                              f"grid=(F={F_tok},H={H_tok},W={W_tok}) seq={seq} D={D} "
                              f"amp={text_amplification} mode=spatial "
                              f"focus={spatial_focus} "
                              f"per_token_amp_range=[{min_amp:.3f},{max_amp:.3f}]")
                        state["hook_logged"] = True

                    state["calls"] += 1
                    return wrap(modified)
                except Exception as e:
                    if debug:
                        print(f"\u2192 [10S] TextAmplifier v1.0: blk{block_idx} hook error: "
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
            if not hasattr(block, "attn2"):
                missing += 1
                continue
            try:
                h = block.attn2.register_forward_hook(make_attn2_hook(i))
                setattr(block.attn2, HOOK_ATTR_ATTN2, h)
                hooked += 1
            except Exception as e:
                missing += 1
                if debug:
                    print(f"\u2192 [10S] TextAmplifier v1.0: blk{i}.attn2 hook failed: "
                          f"{type(e).__name__}: {e}")

        mode_str = (f"uniform" if spatial_focus <= 0
                    else f"spatial_focus={spatial_focus}")
        print(f"\u2192 [10S] TextAmplifier v1.0: {hooked}/{n_blocks} blocks hooked "
              f"(skipped={skipped}, missing={missing}) | "
              f"backbone={type(backbone).__name__} pre_hook={pre_hook_mode} | "
              f"text_amplification={text_amplification} mode={mode_str}")

        if debug:
            attn2_count = sum(
                1 for b in blocks
                if getattr(b, "attn2", None) is not None
                and getattr(b.attn2, HOOK_ATTR_ATTN2, None) is not None
            )
            print(f"  \u00b7 hook census: text_amp_attn2={attn2_count}/{n_blocks}")

        if idx_filter is not None:
            sample = sorted(idx_filter)
            preview = sample[:8] + (["..."] + [sample[-1]] if len(sample) > 8 else [])
            print(f"  \u00b7 block filter active: {preview}")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXTextAttentionAmplifier": LTXTextAttentionAmplifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXTextAttentionAmplifier": "\U0001f50a LTX Text Attention Amplifier",
}
