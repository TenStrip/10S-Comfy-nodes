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
_ORIGINAL_PROCESS_OUTPUT = None
_PATCH_ERROR: Optional[str] = None
_CALL_COUNTER = 0
_VERBOSE = False  # Set True for debug logging

# ── Strata layout constants (backported from Best-Face-ID reference impl) ───
STRATA_SLOT_WIDTH = 1.5     # seconds between strata slots on T axis
                            # Changed 2026-07-18 from 0.5 to 1.5 (fixed slot0/1 collision)



def _log(msg: str):
    if _VERBOSE:
        print(f"[LTX Ref] {msg}")


def _import_comfy():
    """Lazy import so this module loads even outside Comfy."""
    import comfy.ldm.lightricks.av_model as av_module
    import comfy.ldm.lightricks.model as model_module
    from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
    return av_module, model_module, latent_to_pixel_coords


# ── v2: Best-Face-ID source phase rotation ─────────────────────────────────

_ORIGINAL_PREPARE_PE = None
_V2_DEBUG_DONE = False   # set to True after first hook run to reduce log spam
_V2_STRATA_DONE = False  # set to True after first multi-ref segment build
_V2_MULTIREF_DONE = False  # set to True after first multi-ref phase-active print


def _compose_source_phase(cos_orig, sin_orig, ref_len, source_id, phase_scale, theta=10000.0):
    """
    Apply source phase rotation to legacy 4-dim (B,H,T,D_head) cos/sin tensors.
    Kept for backward compatibility.  Newer ComfyUI packs cos+sin+split into
    one 6-dim tensor — see _rotate_packed_freq_tensor for that path.
    """
    if ref_len <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return cos_orig, sin_orig

    B, H, T, D = cos_orig.shape
    device = cos_orig.device
    dtype  = cos_orig.dtype
    if ref_len > T:
        ref_len = T

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    cos_extra = extra_angle.cos().to(dtype=dtype).view(1, 1, 1, D)
    sin_extra = extra_angle.sin().to(dtype=dtype).view(1, 1, 1, D)

    cos_ref = cos_orig[:, :, :ref_len, :]
    sin_ref = sin_orig[:, :, :ref_len, :]

    cos_new_ref = cos_ref * cos_extra - sin_ref * sin_extra
    sin_new_ref = cos_ref * sin_extra + sin_ref * cos_extra

    cos_out = cos_orig.clone()
    sin_out = sin_orig.clone()
    cos_out[:, :, :ref_len, :] = cos_new_ref
    sin_out[:, :, :ref_len, :] = sin_new_ref

    return cos_out, sin_out


def _rotate_packed_freq_tensor(freq_tensor, ref_len, source_id, phase_scale, theta=10000.0):
    """
    Rotate a packed frequency tensor of shape (B, T, H, D_head, 2, 2) where
    the last two dims form a 2x2 rotation matrix per (position, head, dim).

    Composition via element-wise cos/sin arithmetic on the last two dims
    treated as [cos/sin_idx, split_half_idx].  Mathematically equivalent to
    matrix multiplication of the 2x2 rotation but matches the pre-regression
    working form the LoRA responds strongly to.

    Uses theta^(-2d/L) frequency schedule (standard RoPE, adjacent dim pairs).

    Only the first ref_len positions along T (dim 1) get rotated.
    Target positions bit-identical to input (source_id=0 => identity matrix).
    """
    if ref_len <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return freq_tensor
    if freq_tensor.dim() != 6:
        return freq_tensor

    B, T, H, D, two_a, two_b = freq_tensor.shape
    if two_a != 2 or two_b != 2:
        return freq_tensor
    if ref_len > T:
        ref_len = T

    device = freq_tensor.device
    dtype  = freq_tensor.dtype

    # Frequency schedule per rotary dim — Best-Face-ID convention: theta^(-d/L)
    # Single d exponent (NOT 2d/L). Adjacent dim pairs (2i, 2i+1) share a rate.
    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)         # standard RoPE (was -d/D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    # Extra rotation angle per dim
    extra_angle = source_id * phase_scale * rate            # (D,)
    ce = extra_angle.cos().to(dtype=dtype)                  # (D,)
    se = extra_angle.sin().to(dtype=dtype)                  # (D,)

    # Broadcast for element-wise composition (working pre-regression form).
    # Interpret last-two dims as [cos/sin_idx, split_half_idx] and rotate.
    ce_b = ce.view(1, 1, 1, D, 1)
    se_b = se.view(1, 1, 1, D, 1)

    ref_slice = freq_tensor[:, :ref_len]                    # (B, ref_len, H, D, 2, 2)
    cos_ref = ref_slice[..., 0, :]                          # (B, ref_len, H, D, 2)
    sin_ref = ref_slice[..., 1, :]                          # (B, ref_len, H, D, 2)

    cos_new = cos_ref * ce_b - sin_ref * se_b
    sin_new = cos_ref * se_b + sin_ref * ce_b

    ref_rotated = torch.stack([cos_new, sin_new], dim=-2)   # (B, ref_len, H, D, 2, 2)

    result = freq_tensor.clone()
    result[:, :ref_len] = ref_rotated
    return result


def _rotate_freq_tuple(freq_tuple, ref_len, source_id, phase_scale, theta):
    """Dispatch to correct rotation based on tensor layout."""
    if not isinstance(freq_tuple, tuple) or len(freq_tuple) < 2:
        return freq_tuple

    first = freq_tuple[0]
    if not hasattr(first, "shape"):
        return freq_tuple


    if first.dim() == 6 and first.shape[-1] == 2 and first.shape[-2] == 2:
        rotated = _rotate_packed_freq_tensor(
            first, ref_len, source_id, phase_scale, theta
        )
        extras = freq_tuple[1:] if len(freq_tuple) > 1 else ()
        return (rotated, *extras)

    if first.dim() == 4 and len(freq_tuple) >= 2 and hasattr(freq_tuple[1], "shape"):
        cos_new, sin_new = _compose_source_phase(
            first, freq_tuple[1], ref_len, source_id, phase_scale, theta
        )
        extras = freq_tuple[2:] if len(freq_tuple) > 2 else ()
        return (cos_new, sin_new, *extras)

    return freq_tuple



def _apply_multi_source_phase_to_pe(pe, segments, phase_scale, theta=10000.0):
    """
    Apply per-segment phase rotation for stacked references.

    Args:
      segments: list of (start_pos, length, source_id) tuples
      phase_scale: shared magnitude scaler
      theta: RoPE base frequency

    Each segment gets its own source_id → distinct phase band. Prevents the
    "fading crossover" behavior when multiple references occupy overlap
    coordinates.
    """
    if not (isinstance(pe, (list, tuple)) and len(pe) >= 1):
        return pe

    # For each segment, apply _apply_source_phase_to_pe with that segment's
    # ref_len and source_id. Since our source-phase functions currently only
    # support "first N positions" slicing, we compose by working on a copy
    # for each segment individually.
    result_pe = pe
    for seg in segments:
        # Segment tuple may be (start, length, source_id) or (start, length, source_id, slot_idx)
        start_pos = seg[0]
        length = seg[1]
        seg_source_id = seg[2]
        if length <= 0 or seg_source_id == 0.0:
            continue
        # Apply rotation to positions [start_pos : start_pos+length]
        # We do this by temporarily working with a virtual "shifted" pe where
        # the segment appears at position 0, rotate first `length` positions,
        # then merge back. For simplicity we currently only handle segments
        # starting at 0 correctly; segments starting later would need slicing
        # support. Since our concatenation is [ref0, ref1, target...], only
        # ref0 starts at 0. For ref1+ we need offset-aware rotation.
        result_pe = _apply_source_phase_to_pe_ranged(
            result_pe, start_pos, length, seg_source_id, phase_scale, theta
        )
    return result_pe


def _apply_source_phase_to_pe_ranged(pe, start, length, source_id, phase_scale, theta):
    """Apply rotation to a specific position range [start, start+length]."""
    if length <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return pe
    if not (isinstance(pe, (list, tuple)) and len(pe) >= 1):
        return pe

    def rotate_video_selfattn(v_pe):
        if not isinstance(v_pe, tuple) or len(v_pe) < 2:
            return v_pe
        first = v_pe[0]
        if not hasattr(first, "shape"):
            new_halves = []
            for half in v_pe:
                new_halves.append(_rotate_freq_tuple_ranged(
                    half, start, length, source_id, phase_scale, theta))
            return tuple(new_halves)
        return _rotate_freq_tuple_ranged(v_pe, start, length, source_id, phase_scale, theta)

    pe_video_group = pe[0]
    if (isinstance(pe_video_group, tuple) and len(pe_video_group) == 2
            and (isinstance(pe_video_group[0], tuple) or
                 (isinstance(pe_video_group[0], tuple) and len(pe_video_group[0]) >= 2))):
        v_pe_orig, av_cross_v = pe_video_group[0], pe_video_group[1]
        v_pe_new = rotate_video_selfattn(v_pe_orig)
        new_pe0 = (v_pe_new, av_cross_v)
    else:
        new_pe0 = rotate_video_selfattn(pe_video_group)

    if isinstance(pe, list):
        return [new_pe0] + list(pe[1:])
    return (new_pe0,) + tuple(pe[1:])


def _rotate_freq_tuple_ranged(freq_tuple, start, length, source_id, phase_scale, theta):
    """Range-aware version of _rotate_freq_tuple for per-segment rotation."""
    if not isinstance(freq_tuple, tuple) or len(freq_tuple) < 2:
        return freq_tuple
    first = freq_tuple[0]
    if not hasattr(first, "shape"):
        return freq_tuple

    if first.dim() == 6 and first.shape[-1] == 2 and first.shape[-2] == 2:
        rotated = _rotate_packed_freq_tensor_ranged(
            first, start, length, source_id, phase_scale, theta
        )
        extras = freq_tuple[1:] if len(freq_tuple) > 1 else ()
        return (rotated, *extras)

    if first.dim() == 4 and len(freq_tuple) >= 2 and hasattr(freq_tuple[1], "shape"):
        # Legacy 4-dim path: only supports first-N; for arbitrary range we need
        # to slice differently. Simple approach: rotate the full range as-if.
        cos_new, sin_new = _compose_source_phase_ranged(
            first, freq_tuple[1], start, length, source_id, phase_scale, theta
        )
        extras = freq_tuple[2:] if len(freq_tuple) > 2 else ()
        return (cos_new, sin_new, *extras)
    return freq_tuple


def _rotate_packed_freq_tensor_ranged(freq_tensor, start, length, source_id, phase_scale, theta=10000.0):
    """Range-aware rotation of packed 6-dim freq tensor."""
    if length <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return freq_tensor
    if freq_tensor.dim() != 6:
        return freq_tensor

    B, T, H, D, two_a, two_b = freq_tensor.shape
    if two_a != 2 or two_b != 2:
        return freq_tensor
    end = min(T, start + length)
    if end <= start:
        return freq_tensor

    device = freq_tensor.device
    dtype  = freq_tensor.dtype

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    ce = extra_angle.cos().to(dtype=dtype)
    se = extra_angle.sin().to(dtype=dtype)

    ce_b = ce.view(1, 1, 1, D, 1)
    se_b = se.view(1, 1, 1, D, 1)

    ref_slice = freq_tensor[:, start:end]
    cos_ref = ref_slice[..., 0, :]
    sin_ref = ref_slice[..., 1, :]
    cos_new = cos_ref * ce_b - sin_ref * se_b
    sin_new = cos_ref * se_b + sin_ref * ce_b
    ref_rotated = torch.stack([cos_new, sin_new], dim=-2)

    result = freq_tensor.clone()
    result[:, start:end] = ref_rotated
    return result


def _compose_source_phase_ranged(cos_orig, sin_orig, start, length,
                                    source_id, phase_scale, theta=10000.0):
    """Legacy 4-dim path: range-aware rotation."""
    if length <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return cos_orig, sin_orig
    B, H, T, D = cos_orig.shape
    end = min(T, start + length)
    if end <= start:
        return cos_orig, sin_orig
    device = cos_orig.device
    dtype  = cos_orig.dtype

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    cos_extra = extra_angle.cos().to(dtype=dtype).view(1, 1, 1, D)
    sin_extra = extra_angle.sin().to(dtype=dtype).view(1, 1, 1, D)

    cos_ref = cos_orig[:, :, start:end, :]
    sin_ref = sin_orig[:, :, start:end, :]
    cos_new_ref = cos_ref * cos_extra - sin_ref * sin_extra
    sin_new_ref = cos_ref * sin_extra + sin_ref * cos_extra

    cos_out = cos_orig.clone()
    sin_out = sin_orig.clone()
    cos_out[:, :, start:end, :] = cos_new_ref
    sin_out[:, :, start:end, :] = sin_new_ref
    return cos_out, sin_out


def _apply_source_phase_to_pe(pe, ref_len, source_id, phase_scale, theta=10000.0):
    """
    Walk pe structure and apply phase rotation to video self-attn frequencies.

    Post-update AV pe structure (LTXAVModel._prepare_positional_embeddings):
      pe = [
          (v_pe, av_cross_video_freq_cis),   # pe[0]: video path
          (a_pe, av_cross_audio_freq_cis),   # pe[1]: audio path
      ]
    Where v_pe and a_pe themselves are outputs of _precompute_freqs_cis and
    can be either:
      - Nested tuple:  (freq_tuple_half_a, freq_tuple_half_b)  # split RoPE
      - Direct tuple:  (cos, sin, split_flag)                  # non-split

    We rotate the FIRST element of pe[0] (video self-attn = v_pe) only.
    Audio self-attn (pe[1][0]) and AV cross-attn tensors are left untouched
    since reference tokens are video-only.
    """
    if ref_len <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return pe
    if not (isinstance(pe, (list, tuple)) and len(pe) >= 1):
        return pe

    def rotate_video_selfattn(v_pe):
        """Apply rotation to video self-attention frequencies."""
        if not isinstance(v_pe, tuple) or len(v_pe) < 2:
            return v_pe

        first = v_pe[0]
        if not hasattr(first, "shape"):
            # v_pe is a tuple of halves — iterate
            new_halves = []
            for half in v_pe:
                new_halves.append(
                    _rotate_freq_tuple(half, ref_len, source_id, phase_scale, theta)
                )
            return tuple(new_halves)

        # v_pe[0] is a tensor — v_pe is directly a freq tuple, rotate as a whole
        return _rotate_freq_tuple(v_pe, ref_len, source_id, phase_scale, theta)

    # pe[0] is expected to be (v_pe, av_cross_video_freq_cis) tuple
    pe_video_group = pe[0]
    if not globals().get("_V2_STRUCTURE_LOGGED", False):
        globals()["_V2_STRUCTURE_LOGGED"] = True
        globals()["_V2_STRUCTURE_LOG_PRINTED"] = True

    # Handle both old and new layouts within pe[0]:
    if (isinstance(pe_video_group, tuple) and len(pe_video_group) == 2
            and (isinstance(pe_video_group[0], tuple) or
                 (isinstance(pe_video_group[0], tuple) and len(pe_video_group[0]) >= 2))):
        # New AV layout: (v_pe, av_cross_video_freq_cis)
        v_pe_orig, av_cross_v = pe_video_group[0], pe_video_group[1]
        v_pe_new = rotate_video_selfattn(v_pe_orig)
        new_pe0 = (v_pe_new, av_cross_v)
    else:
        # Fallback: treat pe[0] as v_pe directly (legacy structure)
        new_pe0 = rotate_video_selfattn(pe_video_group)

    if isinstance(pe, list):
        new_pe = [new_pe0] + list(pe[1:])
    else:
        new_pe = (new_pe0,) + tuple(pe[1:])
    return new_pe


def _patched_prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
    """v2 + strata: wrap standard pe construction with reference-token phase
    rotation and per-reference strata temporal offsets.

    Multi-ref: applies per-segment source_id and per-segment temporal shift
    if _pending_ref_segments is set. Each reference goes to its own strata
    slot on the T axis, giving both positional and phase-band separation."""
    global _ORIGINAL_PREPARE_PE

    ref_len     = int(getattr(self, "_pending_ref_seq_len", 0) or 0)
    source_id   = float(getattr(self, "_pending_source_id", 0.0) or 0.0)
    phase_scale = float(getattr(self, "_pending_phase_scale", 0.0) or 0.0)
    theta       = float(getattr(self, "positional_embedding_theta", 10000.0) or 10000.0)
    segments    = getattr(self, "_pending_ref_segments", None)

    # Strata coord shift disabled — was hurting rather than helping.
    # Overlap layout (all refs at same coords) with per-ref phase rotation
    # was the previous working state.
    pe = _ORIGINAL_PREPARE_PE(self, pixel_coords, frame_rate, x_dtype)

    global _V2_DEBUG_DONE
    _first_time = not _V2_DEBUG_DONE

    # Multi-ref confirmation uses its own one-shot flag so it fires once
    # per Python session regardless of prior single-ref runs.
    global _V2_MULTIREF_DONE
    _multi_ref_now = (segments is not None and len(segments) > 1
                     and ref_len > 0 and phase_scale != 0.0)

    if ref_len > 0 and phase_scale != 0.0:
        # If all segments share the same source_id, use single flat rotation
        # over the full ref_len range (matches pre-regression behavior).
        if segments is not None and len(segments) > 1:
            unique_source_ids = set(s[2] for s in segments)
            if len(unique_source_ids) == 1:
                # Uniform — collapse to single rotation over full ref range
                uniform_sid = next(iter(unique_source_ids))
                pe = _apply_source_phase_to_pe(pe, ref_len, uniform_sid, phase_scale, theta)
                if not _V2_MULTIREF_DONE:
                    _V2_MULTIREF_DONE = True
                    _V2_DEBUG_DONE = True
                    print(f"[LTX Ref] v2 phase active (multi-ref, uniform sid): "
                          f"{len(segments)} refs, source_id={uniform_sid}, "
                          f"ref_len={ref_len}, phase_scale={phase_scale}")
            else:
                # Distinct per-segment source_ids — use ranged multi-source path
                pe = _apply_multi_source_phase_to_pe(pe, segments, phase_scale, theta)
                if not _V2_MULTIREF_DONE:
                    _V2_MULTIREF_DONE = True
                    _V2_DEBUG_DONE = True
                    print(f"[LTX Ref] v2 multi-source phase active: "
                          f"{len(segments)} refs, source_ids="
                          f"{[s[2] for s in segments]}, "
                          f"phase_scale={phase_scale}")
        elif source_id != 0.0:
            # Single-ref: uniform rotation across all reference positions
            pe = _apply_source_phase_to_pe(pe, ref_len, source_id, phase_scale, theta)
            if _first_time:
                _V2_DEBUG_DONE = True
                print(f"[LTX Ref] v2 source phase active: ref_len={ref_len} "
                      f"source_id={source_id} phase_scale={phase_scale} theta={theta}")
    elif _first_time:
        _V2_DEBUG_DONE = True
        print(f"[LTX Ref] v2 idle (ref_len={ref_len}, source_id={source_id}, "
              f"phase_scale={phase_scale})")

    return pe


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

    # NEW (v1.1): source_id and phase_scale for RoPE source-phase tagging.
    # Best-Face-ID LoRA expects source_id=2, phase_scale=1.0.
    # Read from kwargs first, then nested transformer_options.
    source_id = kwargs.get("reference_source_id")
    if source_id is None and isinstance(transformer_options, dict):
        source_id = transformer_options.get("reference_source_id")
    if source_id is None:
        source_id = 0.0
    source_id = float(source_id)

    phase_scale = kwargs.get("reference_phase_scale")
    if phase_scale is None and isinstance(transformer_options, dict):
        phase_scale = transformer_options.get("reference_phase_scale")
    if phase_scale is None:
        phase_scale = 1.0
    phase_scale = float(phase_scale)

    # Always run the original first
    result = _ORIGINAL_PROCESS_INPUT(self, x, keyframe_idxs, denoise_mask, **kwargs)
    tokens_list, coords_list, additional_args = result

    self._pending_ref_seq_len = 0
    self._pending_source_id = 0.0
    self._pending_phase_scale = 0.0
    self._pending_ref_segments = None

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

    # Position handling — v1 uses pure "overlap" layout (Best-Face-ID default).
    #
    # Per Best-Face-ID creator: reference tokens sit at IDENTICAL raw T/H/W
    # coordinates as the target ("overlap" layout — what the trained LoRA
    # expects).  Disambiguation between reference and target comes from:
    #   1. denoise_mask=0 on reference tokens (always clean, never noised)
    #   2. Sequence position (reference concatenated before target)
    #   3. A per-dimension phase rotation added on top of RoPE (NOT YET
    #      IMPLEMENTED in v1 — this is the piece that requires hooking into
    #      the RoPE application inside attention modules)
    #
    # Without the phase rotation, when both reference and target are clean
    # at the same coordinate near the end of denoising, attention has no
    # positional signal to distinguish them.  v1 relies on sequence-position
    # and clean/noisy state as the sole disambiguation.  Works well early
    # in sampling; may show some identity blending late in sampling.
    #
    # Legacy 'prefix' mode retained for backward compatibility with earlier
    # workflows that expected additive positioning.
    if position_mode == "prefix_continuous" or position_mode == "prefix":
        try:
            ref_temporal_end = float(ref_pixel_coords[:, 0, :, 1].max().item())
            ref_pixel_coords = ref_pixel_coords.clone()
            ref_pixel_coords[:, 0, :, :] -= ref_temporal_end
            _log(f"prefix mode: shifted reference to precede target")
        except Exception as e:
            _log(f"prefix offset failed: {type(e).__name__}: {e}")
    else:
        # overlap / i2v_safe / t2v_overlap / reference — all use pure overlap.
        # No coordinate changes.  Reference reuses target's coord grid.
        _log(f"overlap layout: reference at same coords as target "
             f"(source_id={source_id}, phase_scale={phase_scale} — "
             f"phase rotation deferred to v2 attention-level hook)")

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
    self._pending_source_id = source_id
    self._pending_phase_scale = phase_scale
    self._pending_ref_frames = ref_frames
    # Per-reference segment tracking: each reference frame gets a distinct
    # source_id (source_id, source_id+1, source_id+2, ...) so multiple stacked
    # references don't collapse into an averaged blend.
    # Backported from Best-Face-ID phase overlap reference (per-ref seg_value).
    if ref_frames > 1:
        # UNIFORM source_id for all references — Best-Face-ID LoRA was trained
        # single-ref, so per-ref source_ids move ref2 into an untrained
        # response region.  Uniform makes the LoRA fire identically on both.
        self._pending_ref_segments = [(i * spatial, spatial, source_id, i)
                                      for i in range(ref_frames)]
        global _V2_STRATA_DONE
        if not _V2_STRATA_DONE:
            _V2_STRATA_DONE = True
            print(f"[LTX Ref] Multi-ref: {ref_frames} refs × {spatial} tokens each, "
                  f"all at source_id={source_id} (uniform)")
    else:
        self._pending_ref_segments = None    # single ref uses uniform source_id

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

def _patched_process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
    """Remove the reference prefix before ComfyUI restores keyframe tokens."""
    ref_seq_len = int(kwargs.get("reference_seq_len", 0) or 0)
    if ref_seq_len > 0:
        combined_seq_len = x.shape[1]

        if (
            isinstance(embedded_timestep, torch.Tensor)
            and embedded_timestep.dim() >= 2
            and embedded_timestep.shape[1] == combined_seq_len
        ):
            embedded_timestep = embedded_timestep[:, ref_seq_len:, ...]

        x = x[:, ref_seq_len:, ...]

        # The output prefix has already been removed, so prevent the legacy
        # unpatchify wrapper from stripping the target a second time.
        self._pending_ref_seq_len = 0

    return _ORIGINAL_PROCESS_OUTPUT(
        self, x, embedded_timestep, keyframe_idxs, **kwargs
    )



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
    global _PATCHES_APPLIED, _ORIGINAL_PROCESS_INPUT, _ORIGINAL_PREPARE_TIMESTEP
    global _ORIGINAL_PROCESS_OUTPUT, _PATCH_ERROR

    if _PATCHES_APPLIED:
        return True

    try:
        av_module, model_module, _coords_fn = _import_comfy()
        LTXAVModel = av_module.LTXAVModel
        LTXVModel = model_module.LTXVModel

        _ORIGINAL_PROCESS_INPUT = LTXAVModel._process_input
        _ORIGINAL_PREPARE_TIMESTEP = LTXAVModel._prepare_timestep
        _ORIGINAL_PROCESS_OUTPUT = LTXVModel._process_output

        LTXAVModel._process_input = _patched_process_input
        LTXAVModel._prepare_timestep = _patched_prepare_timestep
        LTXVModel._process_output = _patched_process_output

        # v2: patch _prepare_positional_embeddings on LTXAVModel itself.
        # Prior code walked MRO to find first owner, but if LTXAVModel overrides
        # the base class version, MRO-walk landed on a stale target that was
        # shadowed at runtime. Patch LTXAVModel directly to catch the actual
        # runtime call regardless of where the method is defined higher up.
        global _ORIGINAL_PREPARE_PE

        # Preserve the actual method that will be shadowed by our patch.
        # If LTXAVModel has its own override, capture that. Otherwise walk
        # MRO to find the inherited version.
        if '_prepare_positional_embeddings' in LTXAVModel.__dict__:
            _ORIGINAL_PREPARE_PE = LTXAVModel.__dict__['_prepare_positional_embeddings']
            _pe_source = LTXAVModel.__name__
        else:
            # Not overridden on LTXAVModel — find inherited version to preserve
            _pe_source = None
            for cls in LTXAVModel.__mro__[1:]:
                if '_prepare_positional_embeddings' in cls.__dict__:
                    _ORIGINAL_PREPARE_PE = cls.__dict__['_prepare_positional_embeddings']
                    _pe_source = cls.__name__
                    break

        if _ORIGINAL_PREPARE_PE is not None:
            # Always install patch on LTXAVModel itself so it wins the MRO race
            LTXAVModel._prepare_positional_embeddings = _patched_prepare_positional_embeddings
            print(f"[LTX Ref] v2 patched _prepare_positional_embeddings on LTXAVModel "
                  f"(preserving original from {_pe_source})")
        else:
            print("[LTX Ref] ⚠ could not find _prepare_positional_embeddings anywhere in MRO")

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
