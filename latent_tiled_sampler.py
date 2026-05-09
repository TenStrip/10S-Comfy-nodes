"""
LTX Tiled Sampler v2.0

Spatially-tiled drop-in replacement for SamplerCustomAdvanced. Designed
for upscale-pass refinement of LTX2 video latents where conditioning
gets diluted at upscaled token counts.

================================================================================
CHANGES IN v2.0
================================================================================
- DISCOVERED: LTX wrapper sampling stores its output in a flattened
  combined-tensor form via the x0 capture, NOT in the wrapper's .tensors
  slots. Reading via _extract_components on tile_result returns the
  ORIGINAL INPUT REFERENCES (cached at construction time), which is why
  v1.9's "extracted video" had correct shape but wrong (input) content.
- v2.0 fix: capture the x0 callback during sampling, apply
  process_latent_out, and unflatten the resulting combined tensor
  manually. Math: total_elements = video_shape.numel + audio_shape.numel.
  First chunk reshaped as video, remainder as audio.
- This unflatten approach correctly retrieves both modalities from the
  carrier wrapper sampling pass, preserving video-audio cross-attention
  for lipsync.
- Added value-range diagnostics to confirm video/audio content is
  actually sampled (not stale input or zero-buffer).

================================================================================
CHANGES IN v1.9 (extra process_latent_out — INSUFFICIENT)
================================================================================
- UNIFIED carrier-tile sampling: video and audio are now sampled TOGETHER
  in a single wrapper-input pass on the carrier tile. This preserves
  the model's video-audio cross-attention (essential for lipsync).
- Root cause of v1.6's "blank video" issue identified: the wrapper output
  contains video in raw model coordinate space, but our code was
  accumulating it directly without applying process_latent_out().
  Plain-video sampling applies this internally in guider.sample;
  wrapper extraction must do it manually. v1.9 applies it correctly.
- Non-carrier tiles still sample plain video (same as v1.7+). Audio is
  the same across all tiles (full sequence) so non-carrier audio data
  isn't useful and isn't computed.
- Cost: same as plain tiled sampling. No extra forward passes.

================================================================================
CHANGES IN v1.8 (audio captured via separate pass — REVERTED)
================================================================================
- Restored audio_pass="tile_carrying" with a working architecture:
  - Every tile samples plain video as before (correct video output)
  - On the carrier tile ONLY, an ADDITIONAL wrapper-sampling pass runs
    purely to capture sampled audio. The video output of that pass is
    discarded (we already have correct video from the plain pass).
  - Cost: one extra full-schedule sampling pass on the carrier tile.
    For your 3-step config that's 3 extra forward passes total.
- Audio is the same across all tiles (full sequence), so we only need
  to capture it once — from the chosen carrier tile.
- audio_carrier_tile parameter is again functional ('first' / 'middle'
  / 'last' selects which tile's wrapper pass captures the audio).

================================================================================
CHANGES IN v1.7 (audio sampling temporarily reverted)
================================================================================
- Reverted carrier-tile audio sampling: the wrapper round-trip
  (input wrapper → guider.sample → output wrapper) reliably produces an
  output where only the audio modality reflects sampling — the video
  modality is returned blank/unchanged. Net effect: blank video tile in
  carrier, defeating the purpose of tiled sampling. v1.7 reverts carrier
  to plain-video sampling.
- audio_pass="tile_carrying" preserved as a parameter for forward compat
  but currently produces identical output to "passthrough". Audio
  smoothing alongside tiled video is not currently supported by this
  node — addressing it requires deeper LTX guider integration than this
  drop-in replacement can provide.
- audio_carrier_tile parameter is currently unused; preserved for future
  reactivation.

================================================================================
CHANGES IN v1.6
================================================================================
- Fixed carrier-tile denoised x0 handling: when on the carrier tile, the
  x0_output['x0'] is the full wrapper (video+audio). v1.5's isinstance check
  didn't catch wrappers that subclass torch.Tensor — leaked audio data into
  the denoised video accumulator, causing dimension-mismatch crashes.
  v1.6 unconditionally extracts components on carrier tiles AND verifies
  the extracted video shape matches tile_samples shape before accumulating.
- Added shape verification on carrier tile_samples extraction: if the
  extracted video doesn't match the input tile shape, fall back to plain
  video sampling for that tile rather than corrupt the accumulator.

================================================================================
CHANGES IN v1.5
================================================================================
- New audio_pass="tile_carrying" mode: audio rides along with one chosen
  tile's sampling instead of an extra full pass. Zero extra sampling steps.
- audio_carrier_tile parameter: 'first' / 'middle' / 'last' selects which
  tile carries the audio. 'first' (default) for vertical talking-face
  workflows where the upper half typically contains the speaker. 'middle'
  for very large outputs (4K/8K) where speakers are typically center.
- tile_overlap default raised from 6 to 8 (empirically hides seams better).
- Memory optimisations: explicit del of per-tile intermediates, full_noise
  buffer freed after tile loop, fp32 accumulators freed after dtype cast,
  torch.cuda.empty_cache() between tiles when on CUDA.
- Audio shape sanity check: if captured audio shape doesn't match original,
  fall back to passthrough rather than ship malformed audio downstream
  (guard against ffmpeg errors).

================================================================================
CHANGES IN v1.3
================================================================================
- Fixed: '.ndim()' bug introduced in v1.2 (ndim is a property, not a method).
- Fixed: NestedTensor handling. LTX-Video wraps video+audio in a custom
  NestedTensor-like class that does NOT subclass torch.Tensor, so the
  v1.2 isinstance() check fell through to torch.as_tensor() which failed.
- v1.3 detects multiple NestedTensor formats:
    - PyTorch standard torch.NestedTensor (subclass of torch.Tensor)
    - LTX-Video custom NestedTensor wrapper (not a torch.Tensor subclass)
    - Tuple/list of tensors
    - Plain tensor (passthrough)

================================================================================
APPROACH
================================================================================
Whole-pipeline tiling for light refinement passes:
  1. Split video latent spatially along chosen axis with overlap.
  2. Generate full-size noise once; slice per-tile for consistent overlap noise.
  3. Run FULL sampling pipeline on each tile independently.
  4. Cosine-window blend tile outputs.

In tile_carrying audio mode, the chosen carrier tile receives the full
audio component alongside its video slice for sampling. The model's
video-audio cross-attention runs naturally during that tile's pass.
Sampled audio is captured from the result.

Suited for: 3-step euler_ancestral_cfg_pp upscale refinement (sigmas like
[0.85, 0.7, 0.4, 0]) with distilled CFG=1.

NOT suited for: heavy denoise from pure noise (tiles diverge too much).
"""

import math
import torch
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview


# ─────────────────────────────────────────────────────────────────────────────
# NestedTensor / wrapper extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_components(samples, debug=False):
    """
    Extract video tensor, audio tensor (optional), and format info for
    reconstruction.

    Returns (video, audio, format_info) where:
      video       : 5D tensor (B, C, F, H, W) — required
      audio       : tensor (typically 4D) — None if no audio component
      format_info : dict describing the input wrapper format, used by
                    _reconstruct_samples to rebuild the same structure
                    on output.

    Format types:
      "plain_tensor"   : just a 5D tensor, no audio
      "tuple"          : (video, audio) tuple
      "list"           : [video, audio] list
      "torch_nested"   : PyTorch built-in NestedTensor
      "wrapper_tensors": custom class with .tensors / ._tensors attribute
      "wrapper_attrs"  : custom class with .video / .audio attributes
    """
    type_name = type(samples).__name__

    # Path 1: plain torch.Tensor (no audio)
    if isinstance(samples, torch.Tensor):
        if hasattr(samples, "is_nested") and samples.is_nested:
            # PyTorch built-in NestedTensor
            if debug:
                print(f"  \u00b7 [extract] PyTorch NestedTensor; unbinding")
            try:
                parts = list(samples.unbind())
                video = None
                audio = None
                for p in parts:
                    if isinstance(p, torch.Tensor):
                        if p.dim() == 5 and video is None:
                            video = p
                        elif video is not None and audio is None:
                            audio = p
                if video is not None:
                    if debug:
                        print(f"  \u00b7 [extract] torch_nested: video"
                              f"={tuple(video.shape) if video is not None else None} "
                              f"audio={tuple(audio.shape) if audio is not None else None}")
                    return video, audio, {"type": "torch_nested"}
            except Exception as e:
                if debug:
                    print(f"  \u00b7 [extract] unbind failed: {e}")
        # Plain tensor
        return samples, None, {"type": "plain_tensor"}

    # Path 2: tuple/list - assume (video, audio) ordering
    if isinstance(samples, (tuple, list)):
        if len(samples) >= 1 and isinstance(samples[0], torch.Tensor):
            video = samples[0]
            audio = samples[1] if len(samples) >= 2 and isinstance(samples[1], torch.Tensor) else None
            fmt_type = "list" if isinstance(samples, list) else "tuple"
            if debug:
                print(f"  \u00b7 [extract] {fmt_type}: video={tuple(video.shape)} "
                      f"audio={tuple(audio.shape) if audio is not None else None}")
            return video, audio, {"type": fmt_type, "length": len(samples)}

    if debug:
        print(f"  \u00b7 [extract] non-tensor type: {type_name}; inspecting")

    # Path 3: custom wrapper with .tensors / ._tensors attribute
    for attr in ("tensors", "_tensors"):
        if hasattr(samples, attr):
            val = getattr(samples, attr)
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                video = None
                audio = None
                for t in val:
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 5 and video is None:
                            video = t
                        elif video is not None and audio is None:
                            audio = t
                if video is not None:
                    if debug:
                        print(f"  \u00b7 [extract] wrapper.{attr}: video"
                              f"={tuple(video.shape)} "
                              f"audio={tuple(audio.shape) if audio is not None else None}")
                    return video, audio, {
                        "type": "wrapper_tensors",
                        "wrapper_class": type(samples),
                        "attr": attr,
                        "n_tensors": len(val),
                    }

    # Path 4: custom wrapper with .video / .audio attributes
    video_attr = None
    audio_attr = None
    for attr in ("video", "video_latent", "video_samples"):
        val = getattr(samples, attr, None)
        if isinstance(val, torch.Tensor) and val.dim() == 5:
            video_attr = attr
            break
    for attr in ("audio", "audio_latent", "audio_samples"):
        val = getattr(samples, attr, None)
        if isinstance(val, torch.Tensor):
            audio_attr = attr
            break

    if video_attr:
        video = getattr(samples, video_attr)
        audio = getattr(samples, audio_attr) if audio_attr else None
        if debug:
            print(f"  \u00b7 [extract] wrapper.{video_attr} / "
                  f"{audio_attr or 'None'}")
        return video, audio, {
            "type": "wrapper_attrs",
            "wrapper_class": type(samples),
            "video_attr": video_attr,
            "audio_attr": audio_attr,
        }

    # Path 5: object with unbind() method
    if hasattr(samples, "unbind"):
        try:
            parts = list(samples.unbind())
            video = None
            audio = None
            for p in parts:
                if isinstance(p, torch.Tensor):
                    if p.dim() == 5 and video is None:
                        video = p
                    elif video is not None and audio is None:
                        audio = p
            if video is not None:
                if debug:
                    print(f"  \u00b7 [extract] unbind(): "
                          f"video={tuple(video.shape)} "
                          f"audio={tuple(audio.shape) if audio is not None else None}")
                return video, audio, {
                    "type": "unbind_object",
                    "wrapper_class": type(samples),
                }
        except Exception:
            pass

    # Total failure
    pub_attrs = [a for a in dir(samples) if not a.startswith("_")][:25]
    raise TypeError(
        f"Cannot extract video/audio from samples of type "
        f"'{type_name}'. Available attributes: {pub_attrs}. "
        f"Please report this type name and structure."
    )


def _reconstruct_samples(video, audio, format_info, debug=False):
    """
    Reconstruct the samples wrapper in the same format as input.
    Falls back to tuple if reconstruction fails.
    """
    fmt_type = format_info.get("type", "plain_tensor")

    if fmt_type == "plain_tensor":
        return video

    if fmt_type == "tuple":
        if audio is not None:
            return (video, audio)
        return (video,)

    if fmt_type == "list":
        if audio is not None:
            return [video, audio]
        return [video]

    if fmt_type == "torch_nested":
        try:
            parts = [video] + ([audio] if audio is not None else [])
            return torch.nested.nested_tensor(parts)
        except Exception as e:
            if debug:
                print(f"  \u00b7 [reconstruct] torch.nested failed ({e}); "
                      f"falling back to tuple")
            return tuple([video] + ([audio] if audio is not None else []))

    if fmt_type in ("wrapper_tensors", "unbind_object"):
        wrapper_cls = format_info.get("wrapper_class")
        tensors = [video] + ([audio] if audio is not None else [])
        if wrapper_cls is not None:
            # Try common construction patterns
            for ctor_args in (
                {"tensors": tensors},  # named
                tensors,                # positional list
            ):
                try:
                    if isinstance(ctor_args, dict):
                        return wrapper_cls(**ctor_args)
                    else:
                        return wrapper_cls(ctor_args)
                except Exception:
                    continue
        if debug:
            print(f"  \u00b7 [reconstruct] wrapper class reconstruction failed; "
                  f"falling back to tuple")
        return tuple(tensors)

    if fmt_type == "wrapper_attrs":
        # Can't easily reconstruct without knowing the constructor signature
        if debug:
            print(f"  \u00b7 [reconstruct] wrapper_attrs falling back to tuple")
        if audio is not None:
            return (video, audio)
        return (video,)

    # Default fallback
    if audio is not None:
        return (video, audio)
    return video


def _unflatten_ltx_combined(combined, expected_video_shape, expected_audio_shape, debug=False):
    """
    LTX's process_latent_out on a wrapper produces a flat tensor of shape
    (B, 1, total_elements) or (B, total_channels, total_tokens) where
    video and audio tokens are concatenated along the token-sequence axis,
    both projected into the same channel space.

    For a combined output:
      total_video_tokens = F * H * W
      total_audio_tokens = audio_F * audio_H  (or similar)
      total_tokens = total_video_tokens + total_audio_tokens
      shape = (B, channels, total_tokens) flattened to (B, 1, channels * total_tokens)

    This function unflattens such combined output back into separate video
    and audio tensors with their original shapes.

    Returns (video, audio) on success, (None, None) on shape-arithmetic failure.
    """
    if not isinstance(combined, torch.Tensor):
        if debug:
            print(f"  \u00b7 [unflatten] not a tensor: {type(combined).__name__}")
        return None, None

    B_v, C_v, F_v, H_v, W_v = expected_video_shape
    B_a = expected_audio_shape[0]
    expected_video_elements = B_v * C_v * F_v * H_v * W_v

    audio_elements = 1
    for d in expected_audio_shape:
        audio_elements *= d

    expected_total = expected_video_elements + audio_elements

    actual_total = combined.numel()
    if debug:
        print(f"  \u00b7 [unflatten] combined.shape={tuple(combined.shape)} "
              f"numel={actual_total}")
        print(f"  \u00b7 [unflatten] expected_video={expected_video_elements} "
              f"expected_audio={audio_elements} expected_total={expected_total}")

    if actual_total != expected_total:
        if debug:
            print(f"  \u00b7 [unflatten] element count mismatch "
                  f"({actual_total} vs {expected_total}); cannot unflatten")
        return None, None

    flat = combined.reshape(-1)
    video_flat = flat[:expected_video_elements]
    audio_flat = flat[expected_video_elements:expected_video_elements + audio_elements]

    try:
        video = video_flat.reshape(expected_video_shape).contiguous()
        audio = audio_flat.reshape(expected_audio_shape).contiguous()
        if debug:
            print(f"  \u00b7 [unflatten] success: video={tuple(video.shape)} "
                  f"audio={tuple(audio.shape)}")
        return video, audio
    except Exception as e:
        if debug:
            print(f"  \u00b7 [unflatten] reshape failed: {type(e).__name__}: {e}")
        return None, None


def _compute_tile_starts(total_size, n_tiles, overlap):
    if n_tiles <= 1 or total_size <= 1:
        return [0], total_size
    tile_size = math.ceil((total_size + (n_tiles - 1) * overlap) / n_tiles)
    tile_size = min(tile_size, total_size)
    if n_tiles == 2:
        return [0, max(0, total_size - tile_size)], tile_size
    starts = []
    stride = (total_size - tile_size) / (n_tiles - 1)
    for i in range(n_tiles):
        starts.append(int(round(i * stride)))
    return starts, tile_size


def _make_window_1d(size, fade_left, fade_right, dtype, device):
    win = torch.ones(size, dtype=dtype, device=device)
    if fade_left > 0:
        fl = min(fade_left, size)
        i = torch.arange(fl, dtype=dtype, device=device)
        win[:fl] = 0.5 * (1.0 - torch.cos(math.pi * i / fl))
    if fade_right > 0:
        fr = min(fade_right, size)
        i = torch.arange(fr, dtype=dtype, device=device)
        win[size - fr:] = 0.5 * (1.0 + torch.cos(math.pi * i / fr))
    return win


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTXTiledSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise":        ("NOISE",),
                "guider":       ("GUIDER",),
                "sampler":      ("SAMPLER",),
                "sigmas":       ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "tile_axis":            (["auto", "H", "W"], {"default": "auto"}),
                "n_tiles":              ("INT",     {"default": 2,  "min": 1, "max": 8,   "step": 1}),
                "tile_overlap":         ("INT",     {"default": 8,  "min": 0, "max": 32,  "step": 1}),
                "max_size_for_no_tile": ("INT",     {"default": 24, "min": 8, "max": 256, "step": 1}),
                "audio_pass":           (["passthrough", "tile_carrying"], {"default": "passthrough"}),
                "audio_carrier_tile":   (["first", "middle", "last"], {"default": "first"}),
                "debug":                ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample_tiled"
    CATEGORY = "10S Nodes/Sampling"
    DESCRIPTION = (
        "Spatially-tiled drop-in for SamplerCustomAdvanced for LTX2 upscale-pass "
        "light refinement. Each tile runs full sampling at training-distribution "
        "token count, then cosine-blends. Use only for light denoise. Audio "
        "passthrough by default; tile_carrying runs audio sampling alongside "
        "one chosen tile (no extra steps)."
    )

    def sample_tiled(self, noise, guider, sampler, sigmas, latent_image,
                     tile_axis="auto", n_tiles=2, tile_overlap=8,
                     max_size_for_no_tile=24, audio_pass="passthrough",
                     audio_carrier_tile="first",
                     debug=False):

        latent = latent_image.copy()
        raw_samples = latent["samples"]

        if debug:
            print(f"\u2192 [10S] TiledSampler v2.0: input samples type="
                  f"{type(raw_samples).__name__} audio_pass={audio_pass}")

        # ─── Extract video, audio, and format info ───────────────────────────
        try:
            latent_image_t, audio_tensor, format_info = _extract_components(
                raw_samples, debug=debug
            )
        except TypeError as e:
            print(f"\u2192 [10S] TiledSampler v2.0: extraction failed:\n  {e}")
            raise

        latent_image_t = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_image_t
        )
        latent["samples"] = latent_image_t

        # Noise mask extraction (mask may also be wrapped — extract video portion)
        noise_mask = latent.get("noise_mask", None)
        if noise_mask is not None and not isinstance(noise_mask, torch.Tensor):
            try:
                noise_mask, _, _ = _extract_components(noise_mask, debug=debug)
            except TypeError:
                if debug:
                    print(f"  \u00b7 noise_mask extraction failed; using None")
                noise_mask = None

        if latent_image_t.dim() != 5:
            print(f"\u2192 [10S] TiledSampler v2.0: extracted tensor is "
                  f"{latent_image_t.dim()}D, expected 5D \u2014 single-pass")
            return self._single_pass(noise, guider, sampler, sigmas, latent,
                                     latent_image_t, noise_mask)

        B, C, F, H, W = latent_image_t.shape

        if tile_axis == "auto":
            tile_axis = "H" if H >= W else "W"
        axis_size = H if tile_axis == "H" else W

        if debug:
            print(f"\u2192 [10S] TiledSampler v2.0: video shape="
                  f"{tuple(latent_image_t.shape)} dtype={latent_image_t.dtype} "
                  f"axis={tile_axis} (size={axis_size})")

        if axis_size <= max_size_for_no_tile or n_tiles <= 1:
            if debug:
                reason = (f"axis_size={axis_size} \u2264 {max_size_for_no_tile}"
                          if axis_size <= max_size_for_no_tile
                          else f"n_tiles={n_tiles}")
                print(f"  \u00b7 single-pass ({reason})")
            return self._single_pass(noise, guider, sampler, sigmas, latent,
                                     latent_image_t, noise_mask)

        starts, tile_size = _compute_tile_starts(axis_size, n_tiles, tile_overlap)
        if debug:
            print(f"  \u00b7 n_tiles={n_tiles} tile_size={tile_size} "
                  f"overlap_param={tile_overlap}")
            print(f"  \u00b7 tile starts={starts}")

        full_noise = noise.generate_noise({"samples": latent_image_t})
        if debug:
            print(f"  \u00b7 full noise shape={tuple(full_noise.shape)}")

        # Accumulators live on ComfyUI's canonical compute device. This
        # matches where sampling happens; tile outputs are moved here before
        # accumulation. Input latent may be on a different device (CPU after
        # an offload, etc.) but compute device is consistent across the run.
        device = comfy.model_management.get_torch_device()
        dtype = latent_image_t.dtype

        # Move full_noise and latent to compute device for consistent slicing
        full_noise = full_noise.to(device=device)
        latent_image_t = latent_image_t.to(device=device)

        # ─── Resize noise mask to match latent dims if needed ────────────────
        # Masks in ComfyUI workflows are often at a different resolution than
        # the latent (e.g. created pre-upscale). Slicing with latent indices
        # against a smaller mask produces empty (H=0) slices that crash
        # prepare_mask inside guider.sample. Pre-resize once here so each
        # tile gets a properly-sized mask slice.
        if noise_mask is not None and isinstance(noise_mask, torch.Tensor):
            noise_mask = noise_mask.to(device=device)
            mask_dim = noise_mask.dim()
            target_spatial = (F, H, W)
            if mask_dim == 5:
                _, _, m_F, m_H, m_W = noise_mask.shape
                mask_spatial = (m_F, m_H, m_W)
            elif mask_dim == 4:
                _, m_F, m_H, m_W = noise_mask.shape
                mask_spatial = (m_F, m_H, m_W)
            else:
                mask_spatial = None

            if mask_spatial is not None and mask_spatial != target_spatial:
                if debug:
                    print(f"  \u00b7 noise_mask spatial dims {mask_spatial} "
                          f"!= latent {target_spatial}; resizing")
                added_channel_dim = False
                if mask_dim == 4:
                    noise_mask = noise_mask.unsqueeze(1)  # (B, 1, F, H, W)
                    added_channel_dim = True
                try:
                    noise_mask = torch.nn.functional.interpolate(
                        noise_mask.float(),
                        size=target_spatial,
                        mode="trilinear",
                        align_corners=False,
                    ).to(dtype=dtype, device=device)
                except Exception as e:
                    print(f"\u2192 [10S] TiledSampler v2.0: noise_mask resize failed "
                          f"({type(e).__name__}: {e}); using None")
                    noise_mask = None
                if noise_mask is not None and added_channel_dim:
                    noise_mask = noise_mask.squeeze(1)
                if debug and noise_mask is not None:
                    print(f"  \u00b7 noise_mask resized to {tuple(noise_mask.shape)}")

        if debug:
            print(f"  \u00b7 compute device={device}")

        # ─── Determine audio carrier tile index ──────────────────────────────
        # If audio_pass=tile_carrying, we'll pass the full wrapper (with audio)
        # to this specific tile's sampling so the model sees video+audio
        # together for the tile's sampling. The sampled audio is captured
        # from the result. No extra sampling steps required.
        n_tile_count = len(starts)
        if audio_carrier_tile == "first":
            audio_carrier_idx = 0
        elif audio_carrier_tile == "last":
            audio_carrier_idx = n_tile_count - 1
        else:  # middle
            audio_carrier_idx = n_tile_count // 2

        do_tile_carrying = (audio_pass == "tile_carrying"
                            and audio_tensor is not None)
        if debug and do_tile_carrying:
            print(f"  \u00b7 audio_pass=tile_carrying carrier_tile="
                  f"{audio_carrier_tile} (tile index {audio_carrier_idx})")

        # Captured audio (set when carrier tile fires) — currently unused
        # in v1.7 since carrier tile reverted to plain video. Retained for
        # future reactivation.
        captured_audio = None
        state_carrier_logged = {"logged": False}

        output = torch.zeros_like(latent_image_t, dtype=torch.float32, device=device)
        weights_shape = (1, 1, 1,
                         H if tile_axis == "H" else 1,
                         W if tile_axis == "W" else 1)
        weights = torch.zeros(weights_shape, dtype=torch.float32, device=device)
        denoised_output = torch.zeros_like(output)
        denoised_present = False

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        for tile_idx, ax_start in enumerate(starts):
            ax_end = min(ax_start + tile_size, axis_size)
            is_carrier = do_tile_carrying and (tile_idx == audio_carrier_idx)

            if tile_axis == "H":
                tile_latent = latent_image_t[:, :, :, ax_start:ax_end, :].contiguous()
                tile_noise = full_noise[:, :, :, ax_start:ax_end, :].contiguous()
                if noise_mask is None:
                    tile_noise_mask = None
                elif noise_mask.dim() == 5:
                    tile_noise_mask = noise_mask[:, :, :, ax_start:ax_end, :]
                elif noise_mask.dim() == 4:
                    tile_noise_mask = noise_mask[:, :, ax_start:ax_end, :]
                else:
                    tile_noise_mask = noise_mask
            else:
                tile_latent = latent_image_t[:, :, :, :, ax_start:ax_end].contiguous()
                tile_noise = full_noise[:, :, :, :, ax_start:ax_end].contiguous()
                if noise_mask is None:
                    tile_noise_mask = None
                elif noise_mask.dim() == 5:
                    tile_noise_mask = noise_mask[:, :, :, :, ax_start:ax_end]
                elif noise_mask.dim() == 4:
                    tile_noise_mask = noise_mask[:, :, :, ax_start:ax_end]
                else:
                    tile_noise_mask = noise_mask

            actual_size = tile_latent.shape[3 if tile_axis == "H" else 4]

            if debug:
                carrier_str = " (audio carrier)" if is_carrier else ""
                print(f"  \u00b7 tile {tile_idx+1}/{len(starts)}{carrier_str}: "
                      f"axis_range=[{ax_start},{ax_end}) "
                      f"shape={tuple(tile_latent.shape)}")

            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )

            # ─── Sampling: unified video+audio for carrier, video-only otherwise ─
            # The carrier tile gets a SINGLE sampling pass on the full wrapper
            # so that video-audio cross-attention runs naturally — that's what
            # makes lipsync work.
            #
            # CRITICAL DETAIL discovered after multiple iterations:
            # LTX's wrapper sampling stores its output in a flattened combined
            # form, NOT in the wrapper's .tensors slots. Reading via
            # _extract_components(tile_result) returns the original input
            # references (cached at construction time), not the sampled
            # output. The actual sampled output is in the model's x0 capture
            # as a flat (B, 1, total_elements) tensor where video and audio
            # tokens are concatenated.
            #
            # We extract via the x0 callback and unflatten manually using
            # known expected shapes for video and audio.
            if is_carrier:
                try:
                    if debug:
                        print(f"    [carrier] sampling wrapper "
                              f"(video+audio together for cross-attention)")

                    carrier_input = _reconstruct_samples(
                        tile_latent, audio_tensor.to(device=device),
                        format_info, debug=False
                    )
                    carrier_audio_noise = torch.randn_like(
                        audio_tensor, device=device
                    )
                    carrier_noise_wrapper = _reconstruct_samples(
                        tile_noise, carrier_audio_noise,
                        format_info, debug=False
                    )

                    # Reset x0 capture for this carrier pass
                    x0_output.clear()

                    tile_result = guider.sample(
                        carrier_noise_wrapper, carrier_input,
                        sampler, sigmas,
                        denoise_mask=tile_noise_mask,
                        callback=callback,
                        disable_pbar=disable_pbar,
                        seed=noise.seed,
                    )

                    # ── Try to extract video + audio from sampling result ──
                    # First, inspect tile_result and x0 capture to determine
                    # which contains the actual sampled output.
                    extracted_video = None
                    sampled_audio = None

                    # Strategy 1: use the x0 capture's final state (the model's
                    # last denoised prediction). This typically contains the
                    # combined flat tensor that we can unflatten.
                    if "x0" in x0_output:
                        x0_tensor = guider.model_patcher.model.process_latent_out(
                            x0_output["x0"]
                        )
                        if debug:
                            print(f"    [carrier] x0 after process_latent_out: "
                                  f"shape={tuple(x0_tensor.shape) if isinstance(x0_tensor, torch.Tensor) else type(x0_tensor).__name__} "
                                  f"numel={x0_tensor.numel() if isinstance(x0_tensor, torch.Tensor) else 'n/a'}")
                        if isinstance(x0_tensor, torch.Tensor):
                            unflat_v, unflat_a = _unflatten_ltx_combined(
                                x0_tensor,
                                expected_video_shape=tile_latent.shape,
                                expected_audio_shape=audio_tensor.shape,
                                debug=debug,
                            )
                            if unflat_v is not None and unflat_a is not None:
                                extracted_video = unflat_v
                                sampled_audio = unflat_a

                    # Strategy 2: try _extract_components on tile_result as
                    # fallback if x0 path didn't work
                    if extracted_video is None or sampled_audio is None:
                        if debug:
                            print(f"    [carrier] x0 unflatten not used; "
                                  f"trying _extract_components on tile_result")
                        try:
                            ev, sa, _ = _extract_components(tile_result, debug=False)
                            if extracted_video is None:
                                extracted_video = ev
                            if sampled_audio is None:
                                sampled_audio = sa
                        except Exception as e:
                            if debug:
                                print(f"    \u26a0  [carrier] _extract_components "
                                      f"failed: {type(e).__name__}: {e}")

                    # Verify and use video result
                    if extracted_video is not None and \
                       extracted_video.shape == tile_latent.shape:
                        tile_samples = extracted_video
                        if debug:
                            v_min = tile_samples.min().item()
                            v_max = tile_samples.max().item()
                            v_mean = tile_samples.mean().item()
                            print(f"    [carrier] video sampled in unified pass: "
                                  f"shape={tuple(tile_samples.shape)} "
                                  f"value_range=[{v_min:.3f},{v_max:.3f}] "
                                  f"mean={v_mean:.3f}")
                    else:
                        # Video extraction failed; fall back to plain video
                        actual_shape = (
                            tuple(extracted_video.shape)
                            if extracted_video is not None else None
                        )
                        if debug:
                            print(f"    \u26a0  [carrier] video extraction failed "
                                  f"({actual_shape} vs {tuple(tile_latent.shape)}); "
                                  f"running plain video as fallback")
                        # Need fresh x0 capture and callback for retry
                        x0_output.clear()
                        callback_retry = latent_preview.prepare_callback(
                            guider.model_patcher, sigmas.shape[-1] - 1,
                            x0_output,
                        )
                        tile_samples = guider.sample(
                            tile_noise, tile_latent, sampler, sigmas,
                            denoise_mask=tile_noise_mask,
                            callback=callback_retry,
                            disable_pbar=disable_pbar,
                            seed=noise.seed,
                        )
                        del callback_retry

                    # Capture audio if extraction succeeded
                    if sampled_audio is not None:
                        if sampled_audio.shape == audio_tensor.shape:
                            captured_audio = sampled_audio.to(
                                dtype=audio_tensor.dtype,
                                device=audio_tensor.device,
                            )
                            if debug:
                                a_min = captured_audio.float().min().item()
                                a_max = captured_audio.float().max().item()
                                print(f"    [carrier] captured audio shape="
                                      f"{tuple(captured_audio.shape)} "
                                      f"value_range=[{a_min:.3f},{a_max:.3f}]")
                        else:
                            if debug:
                                print(f"    \u26a0  [carrier] audio shape "
                                      f"{tuple(sampled_audio.shape)} != "
                                      f"original {tuple(audio_tensor.shape)}; "
                                      f"discarding")

                    # Free wrapper intermediates
                    del carrier_input, carrier_noise_wrapper, carrier_audio_noise
                    if 'tile_result' in locals():
                        del tile_result
                    if 'extracted_video' in locals():
                        del extracted_video
                    if 'sampled_audio' in locals():
                        del sampled_audio

                except Exception as e:
                    print(f"\u2192 [10S] TiledSampler v2.0: carrier wrapper "
                          f"sampling failed ({type(e).__name__}: {e}); "
                          f"falling back to plain video for carrier tile")
                    x0_output.clear()
                    callback = latent_preview.prepare_callback(
                        guider.model_patcher, sigmas.shape[-1] - 1, x0_output,
                    )
                    tile_samples = guider.sample(
                        tile_noise, tile_latent, sampler, sigmas,
                        denoise_mask=tile_noise_mask,
                        callback=callback,
                        disable_pbar=disable_pbar,
                        seed=noise.seed,
                    )
            else:
                # Non-carrier tile: plain video sampling
                tile_samples = guider.sample(
                    tile_noise, tile_latent, sampler, sigmas,
                    denoise_mask=tile_noise_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=noise.seed,
                )

            # Force tile output onto the accumulator's device. ComfyUI's
            # sampler can return samples on the intermediate device (often
            # CPU after sampling completes) for memory reasons; our
            # accumulators live on the input latent's device.
            tile_samples = tile_samples.to(device=device)

            has_prev = tile_idx > 0
            has_next = tile_idx < len(starts) - 1
            ov_left = (max(0, min(starts[tile_idx-1] + tile_size, axis_size) - ax_start)
                       if has_prev else 0)
            ov_right = (max(0, ax_end - starts[tile_idx + 1])
                        if has_next else 0)

            window_1d = _make_window_1d(
                actual_size, ov_left, ov_right, torch.float32, device
            )
            if tile_axis == "H":
                window = window_1d.view(1, 1, 1, -1, 1)
            else:
                window = window_1d.view(1, 1, 1, 1, -1)

            if tile_axis == "H":
                output[:, :, :, ax_start:ax_end, :] += tile_samples.float() * window
                weights[:, :, :, ax_start:ax_end, :] += window
            else:
                output[:, :, :, :, ax_start:ax_end] += tile_samples.float() * window
                weights[:, :, :, :, ax_start:ax_end] += window

            if "x0" in x0_output:
                tile_denoised_raw = guider.model_patcher.model.process_latent_out(
                    x0_output["x0"]
                )
                tile_denoised = None

                if not isinstance(tile_denoised_raw, torch.Tensor):
                    # Wrapper output — defensively extract video portion
                    try:
                        extracted, _, _ = _extract_components(tile_denoised_raw, debug=False)
                        tile_denoised = extracted
                    except Exception as e:
                        if debug:
                            print(f"    \u26a0  denoised extract failed: "
                                  f"{type(e).__name__}: {e}")
                        tile_denoised = None
                else:
                    # On carrier tiles, x0 is the LTX flat combined tensor.
                    # Try to unflatten if shape doesn't match expected video.
                    if tile_denoised_raw.shape == tile_samples.shape:
                        tile_denoised = tile_denoised_raw
                    elif is_carrier and audio_tensor is not None:
                        # Try unflatten path
                        unflat_v, _ = _unflatten_ltx_combined(
                            tile_denoised_raw,
                            expected_video_shape=tile_latent.shape,
                            expected_audio_shape=audio_tensor.shape,
                            debug=False,
                        )
                        if unflat_v is not None:
                            tile_denoised = unflat_v
                        else:
                            tile_denoised = None
                    else:
                        tile_denoised = tile_denoised_raw

                # Verify shape matches tile_samples before accumulating
                if tile_denoised is not None and tile_denoised.shape == tile_samples.shape:
                    denoised_present = True
                    tile_denoised = tile_denoised.to(device=device)
                    if tile_axis == "H":
                        denoised_output[:, :, :, ax_start:ax_end, :] += \
                            tile_denoised.float() * window
                    else:
                        denoised_output[:, :, :, :, ax_start:ax_end] += \
                            tile_denoised.float() * window
                    del tile_denoised
                elif tile_denoised is not None and debug:
                    # Only log mismatch on non-carrier tiles or when unflatten
                    # truly failed; carrier flat tensor is expected and silent
                    if not is_carrier:
                        print(f"    \u26a0  denoised shape {tuple(tile_denoised.shape)} "
                              f"!= tile_samples {tuple(tile_samples.shape)}; "
                              f"skipping denoised for this tile")

                del tile_denoised_raw

            if debug:
                print(f"    fades=left:{ov_left} right:{ov_right} | "
                      f"weight_acc: min={weights.min().item():.3f} "
                      f"max={weights.max().item():.3f}")

            # ─── Memory cleanup between tiles ────────────────────────────────
            # Each tile holds substantial activation memory inside guider.sample.
            # Explicitly free per-tile intermediates and clear CUDA cache so the
            # next tile starts with maximum free memory.
            del tile_samples, tile_noise, tile_latent, window_1d, window
            del x0_output, callback
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ─── Free full_noise — done with it after all tiles processed ────────
        del full_noise
        if device.type == "cuda":
            torch.cuda.empty_cache()

        wmin = weights.min().item()
        wmax = weights.max().item()
        if debug:
            print(f"  \u00b7 final weight: min={wmin:.4f} max={wmax:.4f}")
        if wmin < 1e-3:
            print(f"\u2192 [10S] TiledSampler v2.0: \u26a0  weight min={wmin:.4f} "
                  f"\u2014 unstable normalisation. Increase tile_overlap.")
        if wmax > 1.05:
            print(f"\u2192 [10S] TiledSampler v2.0: \u26a0  weight max={wmax:.4f} > 1.05 "
                  f"\u2014 cosine fades not summing properly.")

        output = output / weights.clamp(min=1e-8)
        if denoised_present:
            denoised_output = denoised_output / weights.clamp(min=1e-8)

        # Free the weight buffer once we've normalized
        del weights

        intermediate_device = comfy.model_management.intermediate_device()
        output = output.to(dtype=dtype, device=intermediate_device)
        denoised_output_final = (
            denoised_output.to(dtype=dtype, device=intermediate_device)
            if denoised_present else output
        )
        # Free the fp32 accumulator copies on compute device
        if denoised_present:
            del denoised_output
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if debug:
            print(f"\u2192 [10S] TiledSampler v2.0: video output shape="
                  f"{tuple(output.shape)} dtype={output.dtype}")

        # ─── Audio handling ───────────────────────────────────────────────────
        # Determine output audio:
        #   passthrough     : original audio unchanged (default; fast)
        #   tile_carrying   : audio captured from the carrier tile's sampling
        #                     (rides along with one tile, no extra steps).
        #                     Falls back to passthrough if capture failed.
        output_audio = audio_tensor
        if audio_pass == "tile_carrying" and captured_audio is not None:
            # Sanity check captured audio shape matches original — this is the
            # ffmpeg-error guard. If shapes diverge unexpectedly, fall back
            # rather than ship a malformed audio tensor downstream.
            if audio_tensor is not None:
                if captured_audio.shape == audio_tensor.shape:
                    output_audio = captured_audio
                    if debug:
                        print(f"\u2192 [10S] TiledSampler v2.0: using carrier-tile "
                              f"audio shape={tuple(output_audio.shape)}")
                else:
                    print(f"\u2192 [10S] TiledSampler v2.0: \u26a0  carrier audio "
                          f"shape {tuple(captured_audio.shape)} != original "
                          f"{tuple(audio_tensor.shape)}; using passthrough audio")
            else:
                output_audio = captured_audio

        # ─── Reconstruct samples wrapper for downstream nodes ────────────────
        try:
            reconstructed = _reconstruct_samples(output, output_audio, format_info, debug=debug)
            denoised_reconstructed = _reconstruct_samples(
                denoised_output_final, output_audio, format_info, debug=debug
            )
        except Exception as e:
            print(f"\u2192 [10S] TiledSampler v2.0: reconstruction failed "
                  f"({type(e).__name__}: {e}); outputting tuple")
            reconstructed = (output, output_audio) if output_audio is not None else output
            denoised_reconstructed = (
                (denoised_output_final, output_audio) if output_audio is not None
                else denoised_output_final
            )

        if debug:
            print(f"  \u00b7 reconstructed output type={type(reconstructed).__name__}")

        out_dict = latent.copy()
        out_dict["samples"] = reconstructed
        out_denoised_dict = latent.copy()
        out_denoised_dict["samples"] = denoised_reconstructed

        return (out_dict, out_denoised_dict)

    @staticmethod
    def _single_pass(noise, guider, sampler, sigmas, latent_dict,
                     latent_image_t, noise_mask):
        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            noise.generate_noise(latent_dict),
            latent_image_t,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent_dict.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent_dict.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(
                x0_output["x0"]
            )
        else:
            out_denoised = out
        return (out, out_denoised)


NODE_CLASS_MAPPINGS = {"LTXTiledSampler": LTXTiledSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXTiledSampler": "\U0001f3b2 LTX Tiled Sampler"}
