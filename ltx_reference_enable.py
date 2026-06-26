"""
ltx_reference_enable.py — Reference Token Injection for LTX2.3

A general-purpose mechanism for injecting reference frames into LTX2.3
(and Echo) DiT generation as a token prefix. Works by:

  1. Patchifying the reference latent through the same patchifier the
     target uses (SymmetricPatchifier, patch_size=1) — identical token
     layout and position math
  2. Prepending the resulting tokens to the front of the target's
     video token sequence inside _process_input
  3. Extending the modulation tensors from _prepare_timestep to match
     the new sequence length (memory tokens inherit target frame 0's
     adaLN values)
  4. Stripping the prefix from the output before unpatchify

Mechanism is architecture-general — the same forward operations that
process target tokens also process reference tokens. Attention attends
across the full sequence, so reference tokens provide identity influence
to the target via the standard self-attention pathway.

Discovered while implementing JoyAI-Echo's memory bank pattern, but the
underlying mechanism works on any LTX2.3 checkpoint (Echo, vanilla LTX2,
or merged variants). No JoyAI dependencies required.

Companion node: ltx_reference_conditioning.py (encodes IMAGE → latent,
attaches to MODEL).

Optional companion: ltx_reference_probe.py (diagnostic).

Position modes:
  - reference (default): memory positions overlap target's first frame.
    Provides uniform identity influence across all target frames.
  - prefix_continuous: memory positions precede target temporally.
    Equivalent to standard LTX2 i2v conditioning (image becomes prior
    context, target generates "after" it).

For most use cases, reference mode is what you want — it functions as
an attention-level identity reference without competing with frame_0
latent conditioning for spatial anchoring.
"""

from __future__ import annotations

import torch
from typing import Any, Dict, Optional


# ── Patch state ───────────────────────────────────────────────────────────
_PATCHES_APPLIED = False
_ORIGINAL_PROCESS_INPUT = None
_ORIGINAL_PREPARE_TIMESTEP = None
_PATCH_ERROR: Optional[str] = None
_CALL_COUNTER = 0
_VERBOSE = False  # Set True for debug logging


def _log(msg: str):
    if _VERBOSE:
        print(f"[LTX Ref] {msg}")


def _import_comfy():
    """Lazy import so this module loads even outside Comfy."""
    import comfy.ldm.lightricks.av_model as av_module
    import comfy.ldm.lightricks.model as model_module
    from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
    return av_module, model_module, latent_to_pixel_coords


def _patched_process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
    """Patched LTXAVModel._process_input — injects reference tokens."""
    global _CALL_COUNTER
    _CALL_COUNTER += 1
    call_id = _CALL_COUNTER

    transformer_options = kwargs.get("transformer_options", {}) or {}

    # Look for reference_latent in priority order:
    #   1. kwargs directly (Comfy unpacks model_options["transformer_options"]
    #      into kwargs at the top level — this is how it arrives in practice)
    #   2. nested transformer_options dict (for paths that pass it as-is)
    #   3. attribute side-channel (fallback only, leaks across model clones)
    reference_latent = kwargs.get("reference_latent")
    if reference_latent is None:
        reference_latent = kwargs.get("memory_video")  # legacy key
    if reference_latent is None and isinstance(transformer_options, dict):
        reference_latent = transformer_options.get("reference_latent")
        if reference_latent is None:
            reference_latent = transformer_options.get("memory_video")
    if reference_latent is None:
        reference_latent = getattr(self, "_ltx_reference_latent", None)
        if reference_latent is None:
            reference_latent = getattr(self, "_echo_memory_video", None)

    position_mode = kwargs.get("reference_position_mode") \
        or kwargs.get("memory_position_mode") \
        or "reference"
    if isinstance(transformer_options, dict) and position_mode == "reference":
        position_mode = transformer_options.get("reference_position_mode") \
            or transformer_options.get("memory_position_mode") \
            or "reference"

    # Always run the original first
    result = _ORIGINAL_PROCESS_INPUT(self, x, keyframe_idxs, denoise_mask, **kwargs)
    tokens_list, coords_list, additional_args = result

    self._pending_ref_seq_len = 0

    if reference_latent is None:
        return result

    if reference_latent.dim() != 5:
        _log(f"reference_latent has wrong dim: expected 5D [B,C,F,H,W], "
             f"got {reference_latent.dim()}D shape={tuple(reference_latent.shape)}")
        return result

    vx = tokens_list[0]
    reference_latent = reference_latent.to(device=vx.device, dtype=vx.dtype)

    # Spatial alignment: memory's H, W must match target's for the
    # per-frame compressed modulation extension to produce matching
    # token counts. When the user wires target_latent into the
    # Conditioning node, the image is resized in pixel space and memory
    # comes in pre-aligned. But at sampling time we may still see a
    # mismatch if the sampler is tiled — each tile has different
    # spatial dims than what was set up at conditioning time.
    #
    # Fallback: latent-space bilinear resize. Not as clean as pixel-
    # space (which the Conditioning node does when target_latent is
    # wired), but the only option here since the original pixel image
    # isn't accessible. Works well enough for the tile case.
    target_orig_shape = additional_args.get("orig_shape")
    if target_orig_shape is not None and len(target_orig_shape) >= 5:
        H_target = int(target_orig_shape[3])
        W_target = int(target_orig_shape[4])
        H_mem = int(reference_latent.shape[3])
        W_mem = int(reference_latent.shape[4])

        if (H_mem, W_mem) != (H_target, W_target):
            import torch.nn.functional as _F
            B, C, F_mem_dim, _, _ = reference_latent.shape
            flat = reference_latent.permute(0, 2, 1, 3, 4).reshape(
                B * F_mem_dim, C, H_mem, W_mem
            )
            flat = _F.interpolate(
                flat, size=(H_target, W_target),
                mode='bilinear', align_corners=False
            )
            reference_latent = flat.reshape(
                B, F_mem_dim, C, H_target, W_target
            ).permute(0, 2, 1, 3, 4).contiguous()

            # Log once per distinct mismatch pattern to avoid per-call
            # spam in tiled sampling
            if not hasattr(self, "_ltx_ref_seen_mismatches"):
                self._ltx_ref_seen_mismatches = set()
            key = (H_mem, W_mem, H_target, W_target)
            if key not in self._ltx_ref_seen_mismatches:
                self._ltx_ref_seen_mismatches.add(key)
                _log(f"  auto-resized memory latent {H_mem}x{W_mem} → "
                     f"{H_target}x{W_target} (latent-space fallback for "
                     f"tile or shape mismatch; explicit target_latent "
                     f"routing in Conditioning is preferred for the "
                     f"primary sampling pass)")

    # Patchify reference using same patchifier as target
    try:
        ref_tokens, ref_latent_coords = self.patchifier.patchify(reference_latent)
    except Exception as e:
        _log(f"patchify failed: {type(e).__name__}: {e}")
        return result

    # Same pixel coordinate math
    _, _, latent_to_pixel_coords = _import_comfy()
    try:
        ref_pixel_coords = latent_to_pixel_coords(
            latent_coords=ref_latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )
    except Exception as e:
        _log(f"pixel coords failed: {type(e).__name__}: {e}")
        return result

    # Optional: shift positions for prefix_continuous mode
    if position_mode == "prefix_continuous":
        try:
            ref_temporal_end = float(ref_pixel_coords[:, 0, :, 1].max().item())
            ref_pixel_coords = ref_pixel_coords.clone()
            ref_pixel_coords[:, 0, :, :] -= ref_temporal_end
        except Exception as e:
            _log(f"prefix_continuous offset failed: {type(e).__name__}: {e}")

    # Apply patchify_proj
    try:
        ref_tokens = self.patchify_proj(ref_tokens)
    except Exception as e:
        _log(f"patchify_proj failed: {type(e).__name__}: {e}")
        return result

    # Batch alignment: vx may have batch > 1 due to CFG batching (cond+
    # uncond stacked) or tiled sampling (multiple tiles processed in
    # parallel). Reference is encoded once at batch=1; broadcast it
    # along batch dim to match vx's batch count before concatenation,
    # otherwise torch.cat fails with "Expected size 1 but got size N
    # for tensor number 1 in the list".
    if ref_tokens.shape[0] != vx.shape[0]:
        if ref_tokens.shape[0] == 1:
            ref_tokens = ref_tokens.expand(vx.shape[0], -1, -1)
            ref_pixel_coords = ref_pixel_coords.expand(vx.shape[0], -1, -1, -1)
        else:
            _log(f"  ✗ batch mismatch: ref batch {ref_tokens.shape[0]}, "
                 f"vx batch {vx.shape[0]}, neither is 1 — cannot broadcast. "
                 f"Skipping memory injection for this call.")
            return result

    # Prepend
    vx_combined = torch.cat([ref_tokens, vx], dim=1)
    tokens_list[0] = vx_combined

    v_pixel_coords = coords_list[0]
    v_pixel_coords_combined = torch.cat([ref_pixel_coords, v_pixel_coords], dim=2)
    coords_list[0] = v_pixel_coords_combined

    ref_seq_len = ref_tokens.shape[1]
    ref_frames = int(reference_latent.shape[2])
    target_seq_len = int(vx.shape[1])
    spatial = max(1, ref_seq_len // max(1, ref_frames))
    target_frames = max(1, target_seq_len // spatial)

    additional_args["reference_seq_len"] = ref_seq_len
    additional_args["reference_frames"] = ref_frames
    additional_args["target_seq_len"] = target_seq_len
    additional_args["target_frames"] = target_frames
    self._pending_ref_seq_len = ref_seq_len
    self._pending_ref_frames = ref_frames

    _log(f"Prepending {ref_seq_len} ref tokens "
         f"(target was {target_seq_len}, now {vx_combined.shape[1]}, "
         f"F_ref={ref_frames}, F_tgt≈{target_frames}) [call #{call_id}]")

    return tokens_list, coords_list, additional_args


# ── Modulation tensor extension ────────────────────────────────────────────

def _extend_prefix_in_tensor(t: torch.Tensor, target_size: int, prefix_size: int) -> torch.Tensor:
    """Extend tensor's dim 1 by replicating row 0 prefix_size times at front."""
    if not isinstance(t, torch.Tensor) or t.dim() < 2 or t.shape[1] != target_size:
        return t
    prefix = t[:, 0:1, ...].expand(-1, prefix_size, *([t.shape[i] for i in range(2, t.dim())]))
    return torch.cat([prefix, t], dim=1)


def _walk_and_extend_item(obj, target_seq_len, ref_seq_len,
                            target_frames, ref_frames,
                            zero_ref_timesteps, depth=0):
    """Walk timestep object, extend tensors to include reference prefix.

    Handles three cases:
      1. CompressedTimestep with patches_per_frame > 1 (per-frame compressed
         storage) — extends `.data` along frame dim, increments `.num_frames`.
         The next call to `expand_for_computation()` will then produce a
         tensor sized for target + prefix.
      2. CompressedTimestep with patches_per_frame == 1, num_frames == 1
         (broadcast-only) — no extension needed, broadcasts naturally.
      3. Raw Tensor of shape (B, target_seq_len, dim) or (B, target_frames, dim) —
         direct extension with replication of row 0.
    """
    if depth > 5 or obj is None:
        return obj, 0, 0

    if isinstance(obj, list):
        ext, zer = 0, 0
        for i, item in enumerate(obj):
            new_item, e, z = _walk_and_extend_item(
                item, target_seq_len, ref_seq_len,
                target_frames, ref_frames, zero_ref_timesteps, depth + 1
            )
            obj[i] = new_item
            ext += e
            zer += z
        return obj, ext, zer

    if isinstance(obj, tuple):
        ext, zer = 0, 0
        for item in obj:
            _, e, z = _walk_and_extend_item(
                item, target_seq_len, ref_seq_len,
                target_frames, ref_frames, zero_ref_timesteps, depth + 1
            )
            ext += e
            zer += z
        return obj, ext, zer

    # CompressedTimestep — identified by having data/num_frames/patches_per_frame
    if (hasattr(obj, "data") and hasattr(obj, "num_frames")
            and hasattr(obj, "patches_per_frame")):
        try:
            data = obj.data
            num_frames = obj.num_frames
            patches_per_frame = obj.patches_per_frame

            if not isinstance(data, torch.Tensor) or data.dim() < 2:
                return obj, 0, 0

            # Broadcast case: (B, 1, dim) — expansion returns data unchanged
            # and broadcasts naturally. No extension needed.
            if patches_per_frame == 1 and num_frames == 1:
                return obj, 0, 0

            # Per-frame compressed: data is (B, num_frames, dim).
            # After expand_for_computation: (B, num_frames * patches_per_frame, dim).
            # We want that final size = target_seq_len + ref_seq_len.
            if patches_per_frame > 1 and num_frames * patches_per_frame == target_seq_len:
                # Add ref_frames extra frames, replicating row 0
                prefix = data[:, 0:1, :].expand(-1, ref_frames, -1).contiguous()
                if zero_ref_timesteps:
                    prefix = torch.zeros_like(prefix)
                new_data = torch.cat([prefix, data], dim=1).contiguous()
                obj.data = new_data
                obj.num_frames = num_frames + ref_frames
                _log(f"      extended CompressedTimestep: "
                     f"num_frames {num_frames} → {obj.num_frames}, "
                     f"data shape {tuple(data.shape)} → {tuple(new_data.shape)}, "
                     f"patches_per_frame={patches_per_frame}")
                return obj, 1, (1 if zero_ref_timesteps else 0)

            # Per-token uncompressed (patches_per_frame=1 but num_frames > 1).
            # data shape: (B, num_frames=target_seq_len, dim)
            if patches_per_frame == 1 and num_frames == target_seq_len:
                prefix = data[:, 0:1, :].expand(-1, ref_seq_len, -1).contiguous()
                if zero_ref_timesteps:
                    prefix = torch.zeros_like(prefix)
                new_data = torch.cat([prefix, data], dim=1).contiguous()
                obj.data = new_data
                obj.num_frames = num_frames + ref_seq_len
                _log(f"      extended CompressedTimestep (uncompressed): "
                     f"num_frames {num_frames} → {obj.num_frames}")
                return obj, 1, (1 if zero_ref_timesteps else 0)
        except Exception as e:
            _log(f"      ✗ couldn't extend CompressedTimestep: "
                 f"{type(e).__name__}: {e}")
        return obj, 0, 0

    # Raw tensor — same logic as before
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 2:
            size = obj.shape[1]
            if size == target_seq_len:
                new_obj = _extend_prefix_in_tensor(obj, target_seq_len, ref_seq_len)
                if zero_ref_timesteps:
                    new_obj = new_obj.clone()
                    new_obj[:, :ref_seq_len] = 0.0
                return new_obj, 1, (1 if zero_ref_timesteps else 0)
            elif size == target_frames:
                new_obj = _extend_prefix_in_tensor(obj, target_frames, ref_frames)
                if zero_ref_timesteps:
                    new_obj = new_obj.clone()
                    new_obj[:, :ref_frames] = 0.0
                return new_obj, 1, (1 if zero_ref_timesteps else 0)
        return obj, 0, 0

    return obj, 0, 0


def _patched_prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
    """Extend adaLN modulation tensors to match prepended reference tokens."""
    ref_seq_len = int(kwargs.get("reference_seq_len", 0) or 0)

    if ref_seq_len == 0:
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size, hidden_dtype, **kwargs)

    ref_frames = int(kwargs.get("reference_frames", 0) or 0)
    if ref_frames == 0:
        ref_frames = 1

    target_seq_len = int(kwargs.get("target_seq_len", 0) or 0)
    target_frames = int(kwargs.get("target_frames", 0) or 0)

    if target_seq_len == 0:
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size, hidden_dtype, **kwargs)

    zero_enabled = bool(getattr(self, "_ltx_zero_ref_timesteps", False))

    _log(f"_prepare_timestep ref_seq={ref_seq_len} ref_f={ref_frames} "
         f"tgt_seq={target_seq_len} tgt_f={target_frames}")

    result = _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size, hidden_dtype, **kwargs)

    if not isinstance(result, (tuple, list)):
        _log(f"  result is {type(result).__name__}, not iterable — skipping extension")
        return result

    # Diagnostic: log all tensor shapes encountered in result for visibility
    if _VERBOSE:
        _log(f"  result has {len(result)} top-level slots:")
        for slot_idx, slot in enumerate(result):
            _describe_slot(slot, slot_idx)

    was_tuple = isinstance(result, tuple)
    result_list = list(result) if was_tuple else result

    # Try extending ALL slots (not just first) — earlier assumption that
    # only result[0] held video modulation may have been wrong in cases
    # where conditioning changes the layout
    ext_total, zer_total = 0, 0
    for slot_idx in range(len(result_list)):
        slot = result_list[slot_idx]
        if isinstance(slot, list):
            for i, item in enumerate(slot):
                new_item, e, z = _walk_and_extend_item(
                    item, target_seq_len, ref_seq_len,
                    target_frames, ref_frames, zero_enabled, 0
                )
                slot[i] = new_item
                ext_total += e
                zer_total += z
        elif slot is not None:
            new_slot, e, z = _walk_and_extend_item(
                slot, target_seq_len, ref_seq_len,
                target_frames, ref_frames, zero_enabled, 0
            )
            result_list[slot_idx] = new_slot
            ext_total += e
            zer_total += z

    if ext_total > 0:
        _log(f"  ✓ extended {ext_total} modulation tensor(s)"
             + (f", zeroed {zer_total}" if zer_total > 0 else ""))
    else:
        _log(f"  ⚠ no modulation tensors matched target sizes "
             f"({target_seq_len} per-token or {target_frames} per-frame). "
             f"Block forward will likely fail at adaLN broadcast.")

    return tuple(result_list) if was_tuple else result_list


def _describe_slot(obj, idx, prefix="    "):
    """Diagnostic: print shapes of all tensors inside a result slot.

    Specifically tries hard to find tensors inside non-tensor wrapper
    objects like CompressedTimestep — checks many possible attribute
    names, then falls back to listing all public attributes.
    """
    if obj is None:
        _log(f"{prefix}slot[{idx}]: None")
        return
    if isinstance(obj, (list, tuple)):
        _log(f"{prefix}slot[{idx}]: {type(obj).__name__}[{len(obj)}]")
        for i, item in enumerate(obj):
            _describe_slot(item, f"{idx}.{i}", prefix + "  ")
        return
    if isinstance(obj, torch.Tensor):
        _log(f"{prefix}slot[{idx}]: Tensor{tuple(obj.shape)}")
        return

    # Unknown object — introspect to find tensor attribute(s)
    obj_type = type(obj).__name__

    # CompressedTimestep-shaped object: report num_frames and patches_per_frame
    if (hasattr(obj, "data") and hasattr(obj, "num_frames")
            and hasattr(obj, "patches_per_frame")):
        try:
            d = obj.data
            shape_str = tuple(d.shape) if isinstance(d, torch.Tensor) else f"{type(d).__name__}"
            _log(f"{prefix}slot[{idx}]: {obj_type}("
                 f"data={shape_str}, "
                 f"num_frames={obj.num_frames}, "
                 f"patches_per_frame={obj.patches_per_frame})")
            return
        except Exception:
            pass

    candidate_attrs = [
        "tensor", "data", "_tensor", "_data", "value", "t",
        "compressed", "timesteps", "values", "shift_scale",
        "scale_shift", "modulation", "x", "_x"
    ]
    found_tensors = []
    for attr in candidate_attrs:
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if isinstance(val, torch.Tensor):
                    per_frame = getattr(obj, "per_frame", "?")
                    found_tensors.append(
                        f"{attr}={tuple(val.shape)} per_frame={per_frame}"
                    )
            except Exception:
                pass

    if not found_tensors:
        # Walk all attributes (including private) looking for tensors
        try:
            all_attrs = [a for a in dir(obj) if not a.startswith('__')]
            tensor_attrs = []
            for attr in all_attrs:
                try:
                    val = getattr(obj, attr)
                    if isinstance(val, torch.Tensor):
                        per_frame = getattr(obj, "per_frame", "?")
                        tensor_attrs.append(f"{attr}={tuple(val.shape)}")
                except Exception:
                    pass
            if tensor_attrs:
                found_tensors = tensor_attrs[:3]  # cap to avoid spam
            else:
                found_tensors = [f"no tensor attrs found, all_attrs={all_attrs[:8]}"]
        except Exception as e:
            found_tensors = [f"introspection failed: {e}"]

    _log(f"{prefix}slot[{idx}]: {obj_type}({', '.join(found_tensors)})")


# ── Patchifier unpatchify wrap (instance-level) ────────────────────────────

class _UnpatchifyWrapper:
    """Strip reference prefix from output before reshape."""
    def __init__(self, original_unpatchify, model_ref):
        self._original_unpatchify = original_unpatchify
        self._model_ref = model_ref

    def __call__(self, latents, **kwargs):
        ref_seq_len = int(getattr(self._model_ref, "_pending_ref_seq_len", 0) or 0)
        if ref_seq_len > 0:
            latents = latents[:, ref_seq_len:, :]
            self._model_ref._pending_ref_seq_len = 0
        return self._original_unpatchify(latents, **kwargs)


def _apply_patchifier_wrap(model_instance):
    patchifier = model_instance.patchifier
    if getattr(patchifier, "_ltx_ref_wrapped", False):
        return False
    original_unpatchify = patchifier.unpatchify
    patchifier.unpatchify = _UnpatchifyWrapper(original_unpatchify, model_instance)
    patchifier._ltx_ref_wrapped = True
    patchifier._ltx_ref_original_unpatchify = original_unpatchify
    return True


def apply_global_patches():
    """Apply class-level patches to LTXAVModel. Idempotent."""
    global _PATCHES_APPLIED, _ORIGINAL_PROCESS_INPUT, _ORIGINAL_PREPARE_TIMESTEP, _PATCH_ERROR

    if _PATCHES_APPLIED:
        return True

    try:
        av_module, _model_module, _coords_fn = _import_comfy()
        LTXAVModel = av_module.LTXAVModel

        _ORIGINAL_PROCESS_INPUT = LTXAVModel._process_input
        _ORIGINAL_PREPARE_TIMESTEP = LTXAVModel._prepare_timestep

        LTXAVModel._process_input = _patched_process_input
        LTXAVModel._prepare_timestep = _patched_prepare_timestep

        _PATCHES_APPLIED = True
        _PATCH_ERROR = None
        print("[LTX Ref] Global patches applied to LTXAVModel.")
        return True

    except Exception as e:
        _PATCH_ERROR = f"{type(e).__name__}: {e}"
        print(f"[LTX Ref] ✗ Failed to apply patches: {_PATCH_ERROR}")
        return False


# ── Node class ────────────────────────────────────────────────────────────

class LTXReferenceEnable:
    """Patch the LTX2 model to accept a reference latent as token prefix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "zero_ref_timesteps": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Mark reference tokens as sigma=0 (clean "
                               "reference). Default OFF based on empirical "
                               "testing — most LTX2.3 checkpoints (including "
                               "Echo's released T2V) produce better output "
                               "when reference tokens share target's noise "
                               "sigma. Enable only if a checkpoint was "
                               "trained for clean-reference memory.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed per-call logging. Useful "
                               "for debugging the first time you wire up "
                               "this node. Disable for normal use to keep "
                               "the console clean.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "enable"
    CATEGORY = "10S Nodes/LTX2"
    DESCRIPTION = (
        "Patches LTX2.3-AV models to accept a reference latent that gets "
        "prepended to the video token sequence inside the transformer. "
        "Works as a complementary identity-injection mechanism — most "
        "useful combined with standard i2v frame_0 latent conditioning. "
        "Pair with LTX Reference Conditioning to attach an image. The "
        "patches activate only when a reference latent is provided; "
        "safe passthrough otherwise."
    )

    def enable(self, model, zero_ref_timesteps=False, verbose=False):
        global _VERBOSE
        _VERBOSE = bool(verbose)

        ok = apply_global_patches()
        if not ok:
            raise RuntimeError(
                f"[LTX Reference Enable] Couldn't patch LTXAVModel: {_PATCH_ERROR}"
            )

        try:
            diffusion_model = model.model.diffusion_model
            _apply_patchifier_wrap(diffusion_model)
            diffusion_model._ltx_zero_ref_timesteps = bool(zero_ref_timesteps)
        except AttributeError as e:
            raise RuntimeError(
                f"[LTX Reference Enable] Couldn't access diffusion_model: {e}"
            )

        return (model.clone(),)


NODE_CLASS_MAPPINGS = {
    "LTXReferenceEnable": LTXReferenceEnable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXReferenceEnable": "\U0001f517 LTX Reference Enable",
}
