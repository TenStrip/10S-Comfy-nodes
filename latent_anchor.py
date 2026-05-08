"""
LTX Latent Anchor v1.4 — consolidated surface, simple+advanced modes.

================================================================================
WHAT THIS NODE DOES (plain framing)
================================================================================
Inference-time regulariser for LTX2 video DiT. Snapshots the model's
internal representation of the conditioning frame at a chosen point
in sampling, and uses it as a gentle pull-target for all other frames'
internal representations on every subsequent block forward.

Empirically this stabilises:
  - prompt + image conditioning adherence over long sampling chains
  - scene composition consistency
  - physical sensibility (objects behave consistently with their initial
    interpretation; motion is more plausible)

Not what it does:
  - content-aware reading of the conditioning image (we never decode/VAE
    or analyse the conditioning input directly; we read what the model
    is computing about it internally)
  - replacement for face-targeted identity preservation (use the
    LTXFaceAttentionAnchor node alongside this for that)

================================================================================
WHAT CHANGED IN v1.4 (over v1.3)
================================================================================
- decay_with_distance restored (per-frame strength scaling by temporal
  distance from anchor frame). Useful for the sensibility/physics use
  case: keep early frames anchored hard to preserve initial interpretation,
  let distant frames be freer to express motion.
- Parameter surface split: simple_mode (default) shows 5 essential knobs;
  advanced_mode toggle exposes the full research surface.
- cache_warmup deprecated in favor of cache_at_step (kept as fallback
  only when sigmas is not connected).
- Consolidated cache mode logic: cache_at_sigma removed (redundant with
  cache_at_step); auto_sigma renamed to schedule for clarity.

================================================================================
PARAMETERS — SIMPLE MODE (default)
================================================================================
  strength             : Pull magnitude. 0.05-0.15 typical for
                         sensibility; higher freezes motion. Default 0.10.

  cache_at_step        : Sampling step at which to lock the anchor.
                         Range typically 3-9 on a 13-step run (early to
                         mid sampling). Default 6.

  similarity_threshold : How broadly to pull. Lower = more tokens
                         affected. Default 0.50.

  decay_with_distance  : 0 = uniform pull strength across frames.
                         1 = full linear decay (anchor frame gets full
                         strength, far frames get near-zero). For
                         sensibility/physics use, 0.3-0.6 is useful —
                         preserves early-frame interpretation while
                         allowing far frames to move freely.
                         Default 0.0.

  bypass               : Clear hooks, return model unmodified.

================================================================================
PARAMETERS — ADVANCED MODE (advanced_mode=True)
================================================================================
Adds:
  cache_mode           : schedule | live_extraction | manual_calls.
                         schedule (default): cache at the step computed
                         from cache_at_step + connected sigmas.
                         live_extraction: never cache; recompute anchor
                         live every call. Subtler effect, more flexible.
                         manual_calls: cache after fixed call count
                         (cache_warmup). Fallback when sigmas not wired.

  forwards_per_step    : Forwards per sampling step. 1 for distilled
                         CFG=1, 2 for standard CFG, 3 for LTX2 modality-
                         CFG. Default 1.

  cache_warmup         : Call count for manual_calls mode. Only used
                         when cache_mode=manual_calls.

  anchor_frame         : Latent frame to use as scene reference.
                         Default 0 (conditioning frame in I2V).

  depth_curve          : Per-block strength scaling. flat | ramp_up |
                         ramp_down | late_focus | middle. Default flat.

  block_index_filter   : Block range filter. e.g. "10-30". Empty = all 48.

  debug                : Verbose logging.

================================================================================
FUTURE DIRECTION (noted but not implemented)
================================================================================
Currently the mechanism is structural and content-blind: it preserves
"whatever the model computed about the anchor frame at sampling step N"
without knowing what aspects of that representation matter. A content-
aware extension would read the VAE-encoded conditioning image directly
and modulate intervention based on its actual structural properties.
This would require running parallel forward passes on the conditioning
reference, which is a substantial rebuild beyond v1.x scope.
"""

import torch
import torch.nn.functional as F


# ─── Hardcoded constants (LTX2 architecture) ────────────────────────────────
TRACK_SHARPNESS = 8.0
SPATIAL_PATCH   = 1
TEMPORAL_PATCH  = 1


HOOK_ATTR_BACKBONE   = "_10s_latent_anchor_pre_hook"
HOOK_ATTR_PARENT_PRE = "_10s_latent_anchor_parent_pre_hook"
HOOK_ATTR_ATTN1      = "_10s_latent_anchor_attn1_hook"


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


def _looks_like_sigma(v):
    if not torch.is_tensor(v):
        return False, None
    if v.dim() > 1:
        return False, None
    if v.dim() == 1 and v.shape[0] > 16:
        return False, None
    try:
        val = float(v.flatten()[0]) if v.dim() == 1 else float(v)
    except Exception:
        return False, None
    if 0.0 <= val <= 1.05:
        return True, val
    if 0.0 <= val <= 1001.0:
        return True, val / 1000.0
    return False, None


def _scan_for_sigma(args, kwargs):
    SIGMA_NAMES = ("sigmas", "sigma", "timesteps", "timestep", "t", "ts",
                   "noise_level", "step_index", "step")
    if kwargs:
        for k in SIGMA_NAMES:
            if k in kwargs:
                ok, val = _looks_like_sigma(kwargs[k])
                if ok:
                    return val, f"kwargs['{k}']"
        to = kwargs.get("transformer_options")
        if isinstance(to, dict):
            for k in SIGMA_NAMES:
                if k in to:
                    ok, val = _looks_like_sigma(to[k])
                    if ok:
                        return val, f"transformer_options['{k}']"
            for k, v in to.items():
                if k in SIGMA_NAMES:
                    continue
                ok, val = _looks_like_sigma(v)
                if ok:
                    return val, f"transformer_options['{k}']"
        for k, v in kwargs.items():
            if k == "transformer_options":
                continue
            ok, val = _looks_like_sigma(v)
            if ok:
                return val, f"kwargs['{k}']"
    if args:
        for i, v in enumerate(args):
            ok, val = _looks_like_sigma(v)
            if ok:
                return val, f"args[{i}]"
    return None, None


def _remove_prior_hooks(m, backbone):
    removed = 0
    parent = getattr(m, "model", None)
    if parent is not None:
        h = getattr(parent, HOOK_ATTR_PARENT_PRE, None)
        if h is not None:
            try:
                h.remove(); removed += 1
            except Exception:
                pass
            try:
                delattr(parent, HOOK_ATTR_PARENT_PRE)
            except Exception:
                pass
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

class LTXLatentAnchor:
    """
    Whole-scene latent anchor with mid-sampling cache.
    Coexists with LTXFaceAttentionAnchor on the same model.

    Simple mode: 5 essential knobs.
    Advanced mode: full research parameter surface.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                # ── Simple mode (always visible) ─────────────────────────────
                "sigmas":               ("SIGMAS",),
                "strength":             ("FLOAT",   {"default": 0.10, "min": 0.0,  "max": 5.0,  "step": 0.01}),
                "cache_at_step":        ("INT",     {"default": 6,    "min": 0,    "max": 100,  "step": 1}),
                "similarity_threshold": ("FLOAT",   {"default": 0.50, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "decay_with_distance":  ("FLOAT",   {"default": 0.0,  "min": 0.0,  "max": 1.0,  "step": 0.05}),
                "bypass":               ("BOOLEAN", {"default": False}),
                # ── Advanced mode (only used if advanced_mode=True) ──────────
                "advanced_mode":        ("BOOLEAN", {"default": False}),
                "cache_mode":           (["schedule", "live_extraction", "manual_calls"],
                                         {"default": "schedule"}),
                "forwards_per_step":    ("INT",     {"default": 1,    "min": 1,    "max": 8,    "step": 1}),
                "cache_warmup":         ("INT",     {"default": 144,  "min": 0,    "max": 5000, "step": 1}),
                "anchor_frame":         ("INT",     {"default": 0,    "min": 0,    "max": 256,  "step": 1}),
                "depth_curve":          (["flat", "ramp_up", "ramp_down", "late_focus", "middle"],
                                         {"default": "flat"}),
                "block_index_filter":   ("STRING",  {"default": ""}),
                "debug":                ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Whole-scene latent anchor for LTX2.x. Stabilises prompt + image conditioning "
        "and physical sensibility across long sampling. Connect SIGMAS for predictable "
        "cache timing. Toggle advanced_mode to access research parameters."
    )

    def patch(self, model,
              sigmas=None,
              # Simple mode
              strength=0.10,
              cache_at_step=6,
              similarity_threshold=0.50,
              decay_with_distance=0.0,
              bypass=False,
              # Advanced mode
              advanced_mode=False,
              cache_mode="schedule",
              forwards_per_step=1,
              cache_warmup=144,
              anchor_frame=0,
              depth_curve="flat",
              block_index_filter="",
              debug=False):

        # ── Force defaults for advanced params if not in advanced mode ──────
        if not advanced_mode:
            cache_mode         = "schedule"
            forwards_per_step  = 1
            cache_warmup       = 144
            anchor_frame       = 0
            depth_curve        = "flat"
            block_index_filter = ""
            debug              = False

        m = model.clone()
        backbone, _ = _resolve_diffusion_model(m)
        if backbone is None:
            print("\u2192 [10S] LatentAnchor v1.4: could not locate diffusion backbone.")
            return (m,)
        if not hasattr(backbone, "transformer_blocks"):
            print(f"\u2192 [10S] LatentAnchor v1.4: backbone {type(backbone).__name__} has no "
                  f"'transformer_blocks' attribute.")
            return (m,)

        n_removed = _remove_prior_hooks(m, backbone)
        if n_removed > 0:
            print(f"\u2192 [10S] LatentAnchor v1.4: removed {n_removed} prior hook(s)")

        if bypass or strength <= 0.0:
            reason = "bypass=True" if bypass else "strength == 0"
            print(f"\u2192 [10S] LatentAnchor v1.4: {reason} \u2014 hooks cleared")
            return (m,)

        blocks = backbone.transformer_blocks
        n_blocks = len(blocks)
        idx_filter = _parse_index_filter(block_index_filter, n_blocks)

        # ─── Schedule resolution ──────────────────────────────────────────────
        sigmas_schedule = None
        target_call_count_from_schedule = None
        target_idx_in_schedule = None
        target_sigma_at_idx = None
        if sigmas is not None and cache_mode == "schedule":
            try:
                sched = sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas)
                if len(sched) > 0:
                    sigmas_schedule = sched
                    target_idx = max(0, min(cache_at_step, len(sched) - 1))
                    target_idx_in_schedule = target_idx
                    target_sigma_at_idx = sched[target_idx]
                    target_call_count_from_schedule = target_idx * forwards_per_step
                    if debug:
                        print(f"  \u00b7 [schedule] sigmas connected: len={len(sched)} "
                              f"first={sched[0]:.4f} last={sched[-1]:.4f}")
                        print(f"  \u00b7 [schedule] cache_at_step={cache_at_step} \u2192 "
                              f"target_idx={target_idx} (sigma={target_sigma_at_idx:.4f}) | "
                              f"forwards_per_step={forwards_per_step} \u2192 "
                              f"target_call_count={target_call_count_from_schedule}")
            except Exception as e:
                if debug:
                    print(f"  \u00b7 [schedule] failed to parse sigmas input: "
                          f"{type(e).__name__}: {e}")
                sigmas_schedule = None

        state = {
            "latent_shape":     None,
            "shape_logged":     False,
            "hook_logged":      False,
            "calls":            0,
            "anchor_cache":     {},
            "block_calls":      {},
            "cache_logged":     False,
            "current_sigma":    None,
            "sigma_source":     None,
            "sigma_warned":     False,
            "sigma_seen_count": 0,
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
            if state["latent_shape"] is None:
                if args:
                    _capture_5d_from_iterable(args, "args")
                if state["latent_shape"] is None and kwargs:
                    _capture_5d_from_iterable(kwargs.values(), "kwargs")
            sigma_val, sigma_src = _scan_for_sigma(args, kwargs)
            if sigma_val is not None:
                state["current_sigma"] = sigma_val
                state["sigma_seen_count"] += 1
                if state["sigma_source"] is None:
                    state["sigma_source"] = sigma_src
                    if debug:
                        print(f"  \u00b7 sigma source identified: {sigma_src} "
                              f"(initial value={sigma_val:.4f})")
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

        def _should_cache_now(block_idx, call_count):
            if cache_mode == "live_extraction":
                return False
            if target_call_count_from_schedule is not None:
                return call_count >= target_call_count_from_schedule
            if cache_mode == "schedule":
                # No sigmas connected — fallback to runtime sigma if available
                cur_sigma = state["current_sigma"]
                if cur_sigma is None:
                    if call_count >= cache_warmup:
                        return True
                    return False
                # Use cache_at_step heuristically: assume each step is roughly
                # equal-spaced in sigma. Estimate sigma threshold.
                # This is a fallback approximation.
                approx_sigma_threshold = max(0.0, 1.0 - cache_at_step * 0.1)
                return cur_sigma <= approx_sigma_threshold
            # manual_calls mode
            return call_count >= cache_warmup

        def _apply_blend(tensor, block_idx, depth_mult):
            B, seq, D = tensor.shape
            _, _, F_lat, H_lat, W_lat = state["latent_shape"]
            F_tok = max(1, F_lat // TEMPORAL_PATCH)
            H_tok = max(1, H_lat // SPATIAL_PATCH)
            W_tok = max(1, W_lat // SPATIAL_PATCH)
            if F_tok * H_tok * W_tok != seq:
                return None

            anchor_idx = max(0, min(anchor_frame, F_tok - 1))

            if not state["hook_logged"]:
                K = H_tok * W_tok
                if cache_mode == "live_extraction":
                    cache_str = "live_extraction (no cache)"
                elif target_call_count_from_schedule is not None:
                    cache_str = (f"schedule@call>={target_call_count_from_schedule} "
                                 f"(step{cache_at_step}, sigma{target_sigma_at_idx:.3f})")
                elif cache_mode == "manual_calls":
                    cache_str = f"manual@{cache_warmup}calls"
                else:
                    cache_str = f"schedule (no SIGMAS, fallback)"
                print(f"\u2192 [10S] LatentAnchor v1.4: HOOK ACTIVE | first fire on blk{block_idx} | "
                      f"grid=(F={F_tok},H={H_tok},W={W_tok}) seq={seq} D={D} "
                      f"curve={depth_curve} strength={strength} decay={decay_with_distance} "
                      f"sim_thr={similarity_threshold} cache={cache_str} "
                      f"K={K} (whole anchor frame)")
                state["hook_logged"] = True

            grid = _to_grid(tensor, B, F_tok, H_tok, W_tok, D)

            # ─── Per-frame strength schedule with optional decay ─────────────
            if F_tok > 1 and decay_with_distance > 0.0:
                dist = torch.arange(F_tok, dtype=tensor.dtype, device=tensor.device)
                dist = (dist - anchor_idx).abs() / max(1, F_tok - 1)
                fs = strength * depth_mult * (1.0 - decay_with_distance * dist).clamp(min=0.0)
            else:
                fs = torch.full((F_tok,), strength * depth_mult,
                                dtype=tensor.dtype, device=tensor.device)
            fs[anchor_idx] = 0.0

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
                        sigma_info = (f"sigma={state['current_sigma']:.4f}"
                                      if state["current_sigma"] is not None
                                      else "sigma=None")
                        print(f"  \u00b7 [cache] populated whole-frame anchor for "
                              f"blk{block_idx} after {call_count} calls "
                              f"(K={K}, mode={cache_mode}, {sigma_info})")
                        state["cache_logged"] = True

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

            diff_grid = gathered.reshape(B, F_tok, H_tok, W_tok, D) - grid
            residual = fs.view(1, F_tok, 1, 1, 1) * mask_grid * diff_grid
            grid_modified = grid + residual

            if debug and state["calls"] < 3:
                try:
                    mpf = mask_grid.squeeze(-1).sum(dim=(2, 3))[0].tolist()
                    preview = [f"{v:.1f}" for v in mpf[:8]]
                    tail = (f" ...({len(mpf)-8} more)" if len(mpf) > 8 else "")
                    sim_f = best_sim.float()
                    sigma_str = (f" sigma={state['current_sigma']:.3f}"
                                 if state["current_sigma"] is not None else "")
                    print(f"  \u00b7 blk{block_idx} call {state['calls']}: "
                          f"depth_mult={depth_mult:.3f} K={K} "
                          f"cache={'hit' if cache_valid else 'miss'}{sigma_str} "
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
                        print(f"\u2192 [10S] LatentAnchor v1.4: blk{block_idx} hook error: "
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
                    print(f"\u2192 [10S] LatentAnchor v1.4: blk{i}.attn1 hook failed: "
                          f"{type(e).__name__}: {e}")

        if cache_mode == "live_extraction":
            cache_str = "live_extraction"
        elif target_call_count_from_schedule is not None:
            cache_str = (f"step{cache_at_step}/sigma{target_sigma_at_idx:.3f}/"
                         f"call{target_call_count_from_schedule}")
        elif cache_mode == "manual_calls":
            cache_str = f"manual@{cache_warmup}calls"
        else:
            cache_str = "schedule (no SIGMAS connected)"

        adv_str = " [advanced]" if advanced_mode else ""
        print(f"\u2192 [10S] LatentAnchor v1.4: {hooked}/{n_blocks} blocks hooked "
              f"(skipped={skipped}, missing={missing}){adv_str} | "
              f"backbone={type(backbone).__name__} pre_hook={pre_hook_mode} "
              f"curve={depth_curve} | "
              f"anchor_frame={anchor_frame} strength={strength} "
              f"decay={decay_with_distance} sim_thr={similarity_threshold} "
              f"cache={cache_str}")

        if debug:
            la_attn = sum(
                1 for b in blocks
                if getattr(b, "attn1", None) is not None
                and getattr(b.attn1, HOOK_ATTR_ATTN1, None) is not None
            )
            face_attn = sum(
                1 for b in blocks
                if getattr(b, "attn1", None) is not None
                and getattr(b.attn1, "_10s_face_anchor_attn1_hook", None) is not None
            )
            print(f"  \u00b7 hook census on backbone: "
                  f"latent_anchor_attn1={la_attn}/{n_blocks} | "
                  f"face_anchor_attn1={face_attn}/{n_blocks}")

        if idx_filter is not None:
            sample = sorted(idx_filter)
            preview = sample[:8] + (["..."] + [sample[-1]] if len(sample) > 8 else [])
            print(f"  \u00b7 block filter active: {preview}")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXLatentAnchor": LTXLatentAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLatentAnchor": "\U0001f3af LTX Latent Anchor",
}
