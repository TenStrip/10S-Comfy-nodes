"""
LTX Face Attention Anchor v4.0 — face-targeted identity intervention.

Cache feature was split into its own node (LTXLatentAnchor / latent_anchor.py)
because investigation showed it acts on whole-scene representation rather
than face-specific features. This node retains face-bbox-targeted
intervention only.

================================================================================
CORE MECHANISM
================================================================================
Forward-hook on each BasicAVTransformerBlock.attn1 (video self-attention).
On every block forward, the hook:

  1. Extracts anchor face features from bbox region of anchor frame
     (optionally bilinear-upsampled for finer correspondence resolution).
  2. Computes per-frame-centered cosine similarity between every token
     in the volume and every anchor token.
  3. For each token, gathers its best-matching anchor token.
  4. Builds a soft mask from similarity scores (single-sigmoid for
     'tracked', dual-sigmoid for 'tracked_correction').
  5. Multiplies mask by spatial Gaussian prior centered on bbox to
     confine corrections to the face vicinity.
  6. Adds (per-frame-scaled) residual = strength × mask × (target − current).

================================================================================
INJECTION MODES
================================================================================
  tracked            : single-mask correspondence blend
                       Mask = sigmoid((sim - track_threshold) * 8.0)
                       Best general-purpose identity preservation.

  tracked_correction : dual-mask drift-targeted blend
                       Mask = sigmoid((sim - face_threshold) * 8.0)
                            * sigmoid((identity_threshold - sim) * 8.0)
                       Concentrates correction on tokens that are
                       face-like AND drifted from anchor. Best for
                       hard-cut / scene-transition recovery.

================================================================================
RECOMMENDED CONFIGURATIONS
================================================================================

General identity preservation:
  strength            = 0.10
  inject_mode         = tracked
  depth_curve         = flat
  track_threshold     = 0.50
  spatial_prior       = 0.50
  anchor_upsample     = 2

Stubborn drift / hard cuts:
  strength            = 0.15
  inject_mode         = tracked_correction
  depth_curve         = late_focus
  face_threshold      = 0.30
  identity_threshold  = 0.75
  spatial_prior       = 0.50
  anchor_upsample     = 2

Subtle high-strength variant (per testing):
  strength            = 1.0
  inject_mode         = tracked
  depth_curve         = ramp_up
  block_index_filter  = 10-27
  anchor_upsample     = 3
"""

import torch
import torch.nn.functional as F


# ─── Hardcoded constants (LTX2 architecture) ────────────────────────────────
TOKEN_ORDER     = "fhw"
SIM_CENTERING   = "per_frame"
TRACK_SHARPNESS = 8.0
SPATIAL_PATCH   = 1
TEMPORAL_PATCH  = 1


HOOK_ATTR_BACKBONE = "_10s_face_anchor_pre_hook"
HOOK_ATTR_ATTN1    = "_10s_face_anchor_attn1_hook"
HOOK_ATTR_BLOCK    = "_10s_face_anchor_block_hook"  # cleanup of legacy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bbox(s, default=(0.35, 0.10, 0.65, 0.50)):
    try:
        parts = [float(v.strip()) for v in s.split(",")]
        if len(parts) != 4:
            return default
        x1, y1, x2, y2 = parts
        if x2 <= x1 or y2 <= y1:
            return default
        return (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)))
    except Exception:
        return default


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
            h = getattr(block, HOOK_ATTR_BLOCK, None)
            if h is not None:
                try:
                    h.remove(); removed += 1
                except Exception:
                    pass
                try:
                    delattr(block, HOOK_ATTR_BLOCK)
                except Exception:
                    pass
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

class LTXFaceAttentionAnchor:
    """
    Forward-hook face-region identity anchor for LTX2.x video DiT (v4.0).

    Targets: bbox region of anchor frame as identity source.
    See module docstring for full reference and tuning entry points.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":          ("MODEL",),
                "face_bbox_norm": ("STRING", {"default": "0.35,0.10,0.65,0.50"}),
            },
            "optional": {
                "strength":           ("FLOAT",   {"default": 0.10, "min": 0.0, "max": 5.0, "step": 0.01}),
                "inject_mode":        (["tracked", "tracked_correction"], {"default": "tracked"}),
                "anchor_frame":       ("INT",     {"default": 0,    "min": 0, "max": 256, "step": 1}),
                "anchor_upsample":    ("INT",     {"default": 2,    "min": 1, "max": 4,   "step": 1}),
                "track_threshold":    ("FLOAT",   {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_threshold":     ("FLOAT",   {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "identity_threshold": ("FLOAT",   {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_curve":        (["flat", "ramp_up", "ramp_down", "late_focus", "middle"],
                                       {"default": "flat"}),
                "spatial_prior":      ("FLOAT",   {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
                "block_index_filter": ("STRING",  {"default": ""}),
                "bypass":             ("BOOLEAN", {"default": False}),
                "debug":              ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "v4.0: face-region identity anchor. Cache feature moved to LTXLatentAnchor node "
        "(scene-wide stabilisation). This node remains face-targeted."
    )

    def patch(self, model, face_bbox_norm,
              strength=0.10, inject_mode="tracked",
              anchor_frame=0, anchor_upsample=2,
              track_threshold=0.50,
              face_threshold=0.30, identity_threshold=0.75,
              depth_curve="flat", spatial_prior=0.50,
              block_index_filter="",
              bypass=False, debug=False):

        x1, y1, x2, y2 = _parse_bbox(face_bbox_norm)

        m = model.clone()
        backbone, _ = _resolve_diffusion_model(m)
        if backbone is None:
            print("\u2192 [10S] FaceAnchor v4.0: could not locate diffusion backbone.")
            return (m,)
        if not hasattr(backbone, "transformer_blocks"):
            print(f"\u2192 [10S] FaceAnchor v4.0: backbone {type(backbone).__name__} has no "
                  f"'transformer_blocks' attribute.")
            return (m,)

        n_removed = _remove_prior_hooks(backbone)
        if n_removed > 0:
            print(f"\u2192 [10S] FaceAnchor v4.0: removed {n_removed} prior hook(s)")

        if bypass or strength <= 0.0:
            reason = "bypass=True" if bypass else "strength == 0"
            print(f"\u2192 [10S] FaceAnchor v4.0: {reason} \u2014 hooks cleared")
            return (m,)

        blocks = backbone.transformer_blocks
        n_blocks = len(blocks)
        idx_filter = _parse_index_filter(block_index_filter, n_blocks)

        state = {
            "latent_shape":  None,
            "shape_logged":  False,
            "hook_logged":   False,
            "calls":         0,
        }

        def _capture_5d_from_iterable(it, source_label):
            for v in it:
                if torch.is_tensor(v) and v.dim() == 5:
                    state["latent_shape"] = tuple(v.shape)
                    if debug and not state["shape_logged"]:
                        print(f"  \u00b7 captured 5D latent from {source_label}: "
                              f"{state['latent_shape']}")
                        state["shape_logged"] = True
                    return True
            return False

        def backbone_pre_hook_kw(module, args, kwargs):
            if state["latent_shape"] is not None:
                return None
            if args and _capture_5d_from_iterable(args, "args"):
                return None
            if kwargs and _capture_5d_from_iterable(kwargs.values(), "kwargs"):
                return None
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

        def _to_grid(t, B, F_tok, H_tok, W_tok, D):
            return t.reshape(B, F_tok, H_tok, W_tok, D)

        def _from_grid(grid, B, F_tok, H_tok, W_tok, D, seq):
            return grid.reshape(B, seq, D)

        def _apply_blend(tensor, block_idx, depth_mult):
            B, seq, D = tensor.shape
            _, _, F_lat, H_lat, W_lat = state["latent_shape"]
            F_tok = max(1, F_lat // TEMPORAL_PATCH)
            H_tok = max(1, H_lat // SPATIAL_PATCH)
            W_tok = max(1, W_lat // SPATIAL_PATCH)
            if F_tok * H_tok * W_tok != seq:
                return None

            anchor_idx = max(0, min(anchor_frame, F_tok - 1))
            ay1 = max(0, int(y1 * H_tok))
            ay2 = max(ay1 + 1, min(H_tok, int(round(y2 * H_tok))))
            ax1 = max(0, int(x1 * W_tok))
            ax2 = max(ax1 + 1, min(W_tok, int(round(x2 * W_tok))))
            if (ay2 - ay1) < 1 or (ax2 - ax1) < 1:
                return None

            fh = ay2 - ay1
            fw = ax2 - ax1
            K_native = fh * fw

            if not state["hook_logged"]:
                extra = (f" thr={track_threshold}" if inject_mode == "tracked"
                         else f" face_thr={face_threshold} id_thr={identity_threshold}")
                K_used = K_native * (anchor_upsample ** 2)
                print(f"\u2192 [10S] FaceAnchor v4.0: HOOK ACTIVE | first fire on blk{block_idx} | "
                      f"grid=(F={F_tok},H={H_tok},W={W_tok}) seq={seq} D={D} "
                      f"mode={inject_mode} curve={depth_curve} "
                      f"bbox=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) "
                      f"strength={strength}{extra} "
                      f"upsample={anchor_upsample}x ({K_native}\u2192{K_used} anchor tokens)")
                state["hook_logged"] = True

            grid = _to_grid(tensor, B, F_tok, H_tok, W_tok, D)

            fs = torch.full((F_tok,), strength * depth_mult,
                            dtype=tensor.dtype, device=tensor.device)
            fs[anchor_idx] = 0.0

            anchor_face = grid[:, anchor_idx, ay1:ay2, ax1:ax2, :]

            if anchor_upsample > 1:
                anchor_face_chw = anchor_face.permute(0, 3, 1, 2)
                orig_dtype = anchor_face_chw.dtype
                upsampled = F.interpolate(
                    anchor_face_chw.float(),
                    scale_factor=anchor_upsample,
                    mode="bilinear",
                    align_corners=False,
                ).to(orig_dtype)
                up_h = upsampled.shape[2]
                up_w = upsampled.shape[3]
                anchor_face_full = upsampled.permute(0, 2, 3, 1)
                K = up_h * up_w
                anchor_face_flat = anchor_face_full.reshape(B, K, D)
            else:
                K = K_native
                anchor_face_flat = anchor_face.reshape(B, K, D)

            N = F_tok * H_tok * W_tok
            grid_for_sim = grid.reshape(B, F_tok, H_tok * W_tok, D)
            frame_mean = grid_for_sim.mean(dim=2, keepdim=True)
            centered_grid = grid_for_sim - frame_mean
            all_for_sim = centered_grid.reshape(B, N, D)

            anchor_frame_mean = frame_mean[:, anchor_idx, :, :]
            anchor_for_sim = anchor_face_flat - anchor_frame_mean.expand(B, K, D)

            all_norm = F.normalize(all_for_sim, dim=-1, eps=1e-6)
            anchor_norm = F.normalize(anchor_for_sim, dim=-1, eps=1e-6)
            sim = torch.bmm(all_norm, anchor_norm.transpose(1, 2))

            best_sim, best_idx = sim.max(dim=-1)

            expanded_idx = best_idx.unsqueeze(-1).expand(-1, -1, D)
            gathered = torch.gather(anchor_face_flat, 1, expanded_idx)

            if inject_mode == "tracked":
                mask = torch.sigmoid((best_sim - track_threshold) * TRACK_SHARPNESS)
            else:
                mask_face = torch.sigmoid((best_sim - face_threshold) * TRACK_SHARPNESS)
                mask_drift = torch.sigmoid((identity_threshold - best_sim) * TRACK_SHARPNESS)
                mask = mask_face * mask_drift

            mask_grid = mask.reshape(B, F_tok, H_tok, W_tok, 1)

            if spatial_prior > 0.0:
                cy = (ay1 + ay2) / 2.0
                cx = (ax1 + ax2) / 2.0
                hy = max((ay2 - ay1) / 2.0, 1.0)
                hx = max((ax2 - ax1) / 2.0, 1.0)
                sigma = max(0.5, 2.0 - 1.5 * spatial_prior)
                y_idx = torch.arange(H_tok, dtype=tensor.dtype, device=tensor.device)
                x_idx = torch.arange(W_tok, dtype=tensor.dtype, device=tensor.device)
                dy = (y_idx - cy) / hy
                dx = (x_idx - cx) / hx
                dist_sq = dy.unsqueeze(1).pow(2) + dx.unsqueeze(0).pow(2)
                spatial = torch.exp(-dist_sq / (2.0 * sigma * sigma))
                spatial = spatial.reshape(1, 1, H_tok, W_tok, 1)
                mask_grid = mask_grid * spatial

            diff_grid = gathered.reshape(B, F_tok, H_tok, W_tok, D) - grid
            residual = fs.view(1, F_tok, 1, 1, 1) * mask_grid * diff_grid
            grid_modified = grid + residual

            if debug and state["calls"] < 3:
                try:
                    mpf = mask_grid.squeeze(-1).sum(dim=(2, 3))[0].tolist()
                    preview = [f"{v:.1f}" for v in mpf[:8]]
                    tail = (f" ...({len(mpf)-8} more)" if len(mpf) > 8 else "")
                    sim_f = best_sim.float()
                    print(f"  \u00b7 blk{block_idx} call {state['calls']}: "
                          f"depth_mult={depth_mult:.3f} K={K} "
                          f"sim_mean={sim_f.mean().item():.3f} "
                          f"sim_p90={sim_f.quantile(0.9).item():.3f} "
                          f"|residual|max={residual.float().abs().max().item():.4f}")
                    print(f"    tokens_in_mask per_frame: [{', '.join(preview)}]{tail}")
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
                        print(f"\u2192 [10S] FaceAnchor v4.0: blk{block_idx} hook error: "
                              f"{type(e).__name__}: {e}")
                    return None
            return hook

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
                    print(f"\u2192 [10S] FaceAnchor v4.0: blk{i}.attn1 hook failed: "
                          f"{type(e).__name__}: {e}")

        extra = (f" track_thr={track_threshold}" if inject_mode == "tracked"
                 else f" face_thr={face_threshold} id_thr={identity_threshold}")

        print(f"\u2192 [10S] FaceAnchor v4.0: {hooked}/{n_blocks} blocks hooked "
              f"(skipped={skipped}, missing={missing}) | "
              f"backbone={type(backbone).__name__} pre_hook={pre_hook_mode} "
              f"mode={inject_mode} curve={depth_curve} prior={spatial_prior} "
              f"upsample={anchor_upsample}x | "
              f"bbox=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) "
              f"anchor_frame={anchor_frame} strength={strength}{extra}")

        if debug:
            face_attn = sum(
                1 for b in blocks
                if getattr(b, "attn1", None) is not None
                and getattr(b.attn1, HOOK_ATTR_ATTN1, None) is not None
            )
            la_attn = sum(
                1 for b in blocks
                if getattr(b, "attn1", None) is not None
                and getattr(b.attn1, "_10s_latent_anchor_attn1_hook", None) is not None
            )
            print(f"  \u00b7 hook census on backbone: "
                  f"face_anchor_attn1={face_attn}/{n_blocks} | "
                  f"latent_anchor_attn1={la_attn}/{n_blocks}")

        if idx_filter is not None:
            sample = sorted(idx_filter)
            preview = sample[:8] + (["..."] + [sample[-1]] if len(sample) > 8 else [])
            print(f"  \u00b7 block filter active: {preview}")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXFaceAttentionAnchor": LTXFaceAttentionAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXFaceAttentionAnchor": "\U0001fa6a LTX Face Attention Anchor",
}
