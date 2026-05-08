"""
LTX Latent Upsampler Tiled v1.0

Drop-in replacement for ComfyUI's LTXVLatentUpsampler that tiles the
spatial dimension when input is large enough to trigger the upscale
model's aspect-ratio failure modes (color shifts, distortion at
1300+ pixel vertical extent).

================================================================================
APPROACH
================================================================================
The stock LTXV upsampler runs the entire latent through the upscale model
in one forward pass. For tall aspect ratios this causes color shifts and
distortion because:
  - The upscale CNN's effective receptive field doesn't cover the full
    spatial extent
  - Normalization layers (GroupNorm/BatchNorm) compute statistics over
    highly non-square spatial extents, dominated by the long axis
  - Aspect ratios past training distribution produce out-of-distribution
    behavior

This node tiles the latent spatially with cosine-windowed overlap blending.
Each tile is small enough to stay within the model's training distribution.
Tiles are processed independently then stitched with smooth blending.

WHY THIS IS SAFE
================================================================================
1. Cosine windows sum to exactly 1.0 in overlap zones (Hann window
   property). Perfect reconstruction within tiles' overlap regions.
2. Edge-aware windowing: tiles at the image boundary don't fade outward,
   so corners and edges are full-weight from a single tile.
3. Un-normalize and re-normalize happen ONCE on the full latent, not per
   tile. Per-channel statistics are global; per-tile normalization would
   shift colors systematically.
4. Float32 accumulators preserve precision during blending; cast back to
   input dtype at the end.
5. Auto-skip for small inputs: latents <= max_size_for_no_tile in both
   dims go through the original non-tiled path. Drop-in compatible.

================================================================================
PARAMETERS
================================================================================
  samples         : LATENT input
  upscale_model   : LATENT_UPSCALE_MODEL (LTX upscale model)
  vae             : VAE (for per-channel statistics normalization)

  tile_size              : Spatial tile size in latent tokens. Default 24
                           (~768 pixels at 32x VAE compression). Should be
                           well within the upscale model's training
                           distribution.
  overlap                : Overlap in latent tokens between adjacent tiles.
                           Default 8. Must be > 0 to avoid seams; larger
                           overlap = smoother blending but more compute.
  max_size_for_no_tile   : If both spatial dims of input latent are <=
                           this size, skip tiling and use single-pass
                           upscale. Default 32. Set lower to force tiling
                           on smaller inputs (rarely needed).
  debug                  : Print tile counts and per-tile diagnostics.
"""

import math
import torch
from comfy import model_management


class LTXVLatentUpsamplerTiled:
    """
    Spatially-tiled drop-in replacement for LTXVLatentUpsampler.
    Solves color/distortion artifacts at extreme aspect ratios.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples":       ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae":           ("VAE",),
            },
            "optional": {
                "tile_size":             ("INT",     {"default": 24, "min": 8,  "max": 128, "step": 1}),
                "overlap":               ("INT",     {"default": 8,  "min": 2,  "max": 32,  "step": 1}),
                "max_size_for_no_tile":  ("INT",     {"default": 32, "min": 8,  "max": 256, "step": 1}),
                "rotate_for_landscape":  ("BOOLEAN", {"default": False}),
                "debug":                 ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upsample_latent_tiled"
    CATEGORY = "10S Nodes/Latent"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Tiled drop-in replacement for LTXVLatentUpsampler. Solves color shifts and "
        "distortion at extreme aspect ratios by processing the upscale in spatial "
        "tiles with cosine-windowed overlap blending. Output is virtually identical "
        "to non-tiled in safe-aspect cases, free of upscale-model instability in "
        "extreme cases."
    )

    def upsample_latent_tiled(self, samples, upscale_model, vae,
                              tile_size=24, overlap=8,
                              max_size_for_no_tile=32,
                              rotate_for_landscape=False,
                              debug=False):
        device = model_management.get_torch_device()
        model_dtype = next(upscale_model.parameters()).dtype
        latents = samples["samples"]
        input_dtype = latents.dtype
        B, C, F, H, W = latents.shape

        if debug:
            print(f"\u2192 [10S] LatentUpsamplerTiled: input shape={tuple(latents.shape)} "
                  f"dtype={input_dtype}")

        if overlap >= tile_size:
            print(f"\u2192 [10S] LatentUpsamplerTiled: overlap={overlap} >= tile_size={tile_size}; "
                  f"clamping overlap to {tile_size - 1}")
            overlap = max(1, tile_size - 1)

        # Memory estimate — only one tile in memory at a time, plus accumulators
        memory_required = model_management.module_size(upscale_model)
        tile_volume = B * C * F * (tile_size * 2) ** 2
        output_volume = B * C * F * (H * 2) * (W * 2)
        memory_required += tile_volume * 3000.0
        memory_required += output_volume * 4.0  # fp32 accumulator
        model_management.free_memory(memory_required, device)

        try:
            upscale_model.to(device)

            # Un-normalize ONCE on full latent (global per-channel statistics)
            latents_dev = latents.to(dtype=model_dtype, device=device)
            latents_un = vae.first_stage_model.per_channel_statistics.un_normalize(latents_dev)

            # ─── Optional rotation for landscape orientation ─────────────────
            # Hypothesis: upscale model's training distribution is biased
            # toward landscape (most video data is wider than tall). Tall
            # portrait inputs produce out-of-distribution behaviour (color
            # drift). Transposing H, W puts content in landscape orientation
            # for the upscale forward, then rotates back. This is a lossless
            # geometric transform (transpose is bijective).
            rotated = False
            if rotate_for_landscape and latents_un.shape[-2] > latents_un.shape[-1]:
                latents_un = latents_un.transpose(-1, -2).contiguous()
                rotated = True
                if debug:
                    print(f"  \u00b7 rotated for landscape: H={H}>W={W} \u2192 "
                          f"shape={tuple(latents_un.shape)}")
                # Update local H, W for tiling decisions
                H, W = W, H

            # Decide tiling
            should_tile = (H > max_size_for_no_tile) or (W > max_size_for_no_tile)
            if not should_tile:
                if debug:
                    print(f"  \u00b7 H={H} W={W} both \u2264 max_size_for_no_tile="
                          f"{max_size_for_no_tile}; using non-tiled path")
                upsampled = upscale_model(latents_un)
            else:
                if debug:
                    print(f"  \u00b7 tiling triggered: H={H} > {max_size_for_no_tile} "
                          f"or W={W} > {max_size_for_no_tile}")
                upsampled = self._upsample_tiled(
                    latents_un, upscale_model, tile_size, overlap, debug
                )

            # Rotate back if we rotated
            if rotated:
                upsampled = upsampled.transpose(-1, -2).contiguous()
                if debug:
                    print(f"  \u00b7 rotated back: shape={tuple(upsampled.shape)}")

            # Re-normalize ONCE on full output
            upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)
        finally:
            upscale_model.cpu()

        upsampled = upsampled.to(
            dtype=input_dtype,
            device=model_management.intermediate_device(),
        )

        if debug:
            print(f"\u2192 [10S] LatentUpsamplerTiled: output shape={tuple(upsampled.shape)}")

        return_dict = samples.copy()
        return_dict["samples"] = upsampled
        return_dict.pop("noise_mask", None)
        return (return_dict,)

    # ─── Core tiled upscale ─────────────────────────────────────────────────

    def _upsample_tiled(self, latents, upscale_model, tile_size, overlap, debug):
        """
        Spatial tiling with cosine-windowed overlap blending.
        Frames processed together per tile (preserves temporal consistency).

        v1.1 fixes:
          - Auto-detect upscale ratio from first tile (supports x1.5, x2, x4 etc.)
          - Compute fade sizes from ACTUAL per-tile overlaps, not configured
            overlap parameter. This is critical because last-tile alignment
            forces some overlaps to be larger than configured, and using
            configured overlap for window fades produces cosines that don't
            sum to 1.0, causing weight accumulation > 1 and incorrect blending.
        """
        device = latents.device
        dtype = latents.dtype
        B, C, F, H, W = latents.shape

        h_starts = self._compute_tile_starts(H, tile_size, overlap)
        w_starts = self._compute_tile_starts(W, tile_size, overlap)

        # Helper: compute actual overlap with neighbor tiles (in input coords)
        def actual_overlap_with_prev(starts, idx, tile_sz, total):
            if idx <= 0:
                return 0
            prev_start = starts[idx - 1]
            prev_end = min(prev_start + tile_sz, total)
            this_start = starts[idx]
            return max(0, prev_end - this_start)

        def actual_overlap_with_next(starts, idx, tile_sz, total):
            if idx >= len(starts) - 1:
                return 0
            this_start = starts[idx]
            this_end = min(this_start + tile_sz, total)
            next_start = starts[idx + 1]
            return max(0, this_end - next_start)

        # ─── Process first tile to determine upscale ratio ───────────────────
        h0_end = min(h_starts[0] + tile_size, H)
        w0_end = min(w_starts[0] + tile_size, W)
        first_tile_in = latents[:, :, :, h_starts[0]:h0_end, w_starts[0]:w0_end].contiguous()
        first_tile_out = upscale_model(first_tile_in)

        # Detect actual scale (could be 2.0, 1.5, etc.)
        scale_h = first_tile_out.shape[3] / first_tile_in.shape[3]
        scale_w = first_tile_out.shape[4] / first_tile_in.shape[4]

        # Sanity: most LTX upscalers are uniform scale
        scale = (scale_h + scale_w) / 2.0
        if abs(scale_h - scale_w) > 0.01:
            print(f"\u2192 [10S] LatentUpsamplerTiled: WARN non-uniform scale detected "
                  f"({scale_h:.3f} vs {scale_w:.3f}); using average={scale:.3f}")

        # Output dimensions
        out_H = int(round(H * scale))
        out_W = int(round(W * scale))

        if debug:
            print(f"  \u00b7 detected upscale ratio: {scale:.3f}x "
                  f"\u2192 output {out_H}x{out_W} (from {H}x{W})")
            print(f"  \u00b7 tile_size={tile_size} overlap={overlap} "
                  f"\u2192 h_starts={h_starts} ({len(h_starts)}) "
                  f"w_starts={w_starts} ({len(w_starts)}) "
                  f"total_tiles={len(h_starts)*len(w_starts)}")

        # fp32 accumulators on device for blending precision
        output = torch.zeros((B, C, F, out_H, out_W),
                             dtype=torch.float32, device=device)
        weights = torch.zeros((1, 1, 1, out_H, out_W),
                              dtype=torch.float32, device=device)

        single_h = len(h_starts) == 1
        single_w = len(w_starts) == 1
        first_tile_logged = False

        for h_idx, h_start in enumerate(h_starts):
            h_end = min(h_start + tile_size, H)
            for w_idx, w_start in enumerate(w_starts):
                w_end = min(w_start + tile_size, W)

                # Reuse first tile output, otherwise compute
                if h_idx == 0 and w_idx == 0:
                    tile_out = first_tile_out
                    tile_in_shape = first_tile_in.shape
                else:
                    tile_in = latents[:, :, :, h_start:h_end, w_start:w_end].contiguous()
                    tile_out = upscale_model(tile_in)
                    tile_in_shape = tile_in.shape

                # Output positions — use actual tile_out shape, not assumed
                out_h_start = int(round(h_start * scale))
                out_h_end = out_h_start + tile_out.shape[3]
                out_w_start = int(round(w_start * scale))
                out_w_end = out_w_start + tile_out.shape[4]

                # Clamp to output buffer (defensive — shouldn't need this but
                # rounding can produce off-by-one at last tile)
                out_h_end = min(out_h_end, out_H)
                out_w_end = min(out_w_end, out_W)
                actual_out_h = out_h_end - out_h_start
                actual_out_w = out_w_end - out_w_start
                if actual_out_h <= 0 or actual_out_w <= 0:
                    continue

                # ─── Compute ACTUAL overlaps with neighbours (output coords) ──
                ov_top_in    = actual_overlap_with_prev(h_starts, h_idx, tile_size, H)
                ov_bot_in    = actual_overlap_with_next(h_starts, h_idx, tile_size, H)
                ov_left_in   = actual_overlap_with_prev(w_starts, w_idx, tile_size, W)
                ov_right_in  = actual_overlap_with_next(w_starts, w_idx, tile_size, W)

                fade_top    = int(round(ov_top_in   * scale)) if not single_h else 0
                fade_bot    = int(round(ov_bot_in   * scale)) if not single_h else 0
                fade_left   = int(round(ov_left_in  * scale)) if not single_w else 0
                fade_right  = int(round(ov_right_in * scale)) if not single_w else 0

                # Guard against fade > tile output size
                fade_top   = min(fade_top,   actual_out_h)
                fade_bot   = min(fade_bot,   actual_out_h)
                fade_left  = min(fade_left,  actual_out_w)
                fade_right = min(fade_right, actual_out_w)

                window = self._make_window_2d(
                    actual_out_h, actual_out_w,
                    fade_top   = fade_top,
                    fade_bot   = fade_bot,
                    fade_left  = fade_left,
                    fade_right = fade_right,
                    device=device,
                )                                                       # (h, w)
                window = window.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,h,w)

                # If tile_out is slightly larger than output region (rounding),
                # crop to fit
                tile_out_cropped = tile_out[:, :, :, :actual_out_h, :actual_out_w]

                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += \
                    tile_out_cropped.float() * window
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += window

                if debug and not first_tile_logged:
                    print(f"  \u00b7 first tile: in={tuple(tile_in_shape)} "
                          f"\u2192 out={tuple(tile_out.shape)} "
                          f"window=({actual_out_h},{actual_out_w}) "
                          f"actual_overlaps=top:{ov_top_in},bot:{ov_bot_in},"
                          f"left:{ov_left_in},right:{ov_right_in} (input coords) "
                          f"\u2192 fades=top:{fade_top},bot:{fade_bot},"
                          f"left:{fade_left},right:{fade_right}")
                    first_tile_logged = True

        # Health check: weights should be ~1.0 everywhere, not > 1
        if debug:
            w_min = weights.min().item()
            w_max = weights.max().item()
            print(f"  \u00b7 weight accumulator: min={w_min:.4f} max={w_max:.4f} "
                  f"(should be \u22481.0 everywhere; max>1 indicates blending bug)")
            if w_min < 1e-3:
                print(f"  \u26a0  weight min very low \u2014 some output positions "
                      f"have unstable normalization. Increase overlap.")
            if w_max > 1.05:
                print(f"  \u26a0  weight max > 1.05 \u2014 cosine fades not summing "
                      f"to 1.0 in overlap zones. Possible window construction issue.")

        # Normalize by accumulated weights
        output = output / (weights + 1e-8)

        return output.to(dtype=dtype)

    # ─── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_tile_starts(total_size, tile_size, overlap):
        """
        Tile start positions covering [0, total_size) with given overlap.
        Last tile is aligned to total_size - tile_size to ensure full coverage
        (may have larger overlap with the second-to-last tile).
        """
        if total_size <= tile_size:
            return [0]

        starts = []
        stride = tile_size - overlap
        pos = 0
        while pos + tile_size < total_size:
            starts.append(pos)
            pos += stride

        # Always include a tile aligned to the end
        last_start = total_size - tile_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        return starts

    @staticmethod
    def _make_window_1d(size, fade_left_size, fade_right_size, device):
        """
        1D weight window: 1.0 in middle, cosine fade at edges where
        fade_X_size > 0. Two adjacent tiles' fades sum to exactly 1.0 in the
        overlap zone (Hann window property).
        """
        win = torch.ones(size, dtype=torch.float32, device=device)

        if fade_left_size > 0:
            fl = min(fade_left_size, size)
            i = torch.arange(fl, dtype=torch.float32, device=device)
            # Rises from 0 at i=0 to ~1 at i=fl-1
            win[:fl] = 0.5 * (1.0 - torch.cos(math.pi * i / fl))

        if fade_right_size > 0:
            fr = min(fade_right_size, size)
            i = torch.arange(fr, dtype=torch.float32, device=device)
            # Falls from ~1 at start of fade to 0 at end
            win[size - fr:] = 0.5 * (1.0 + torch.cos(math.pi * i / fr))

        return win

    @staticmethod
    def _make_window_2d(h, w, fade_top, fade_bot, fade_left, fade_right, device):
        """2D window via outer product of 1D windows."""
        win_h = LTXVLatentUpsamplerTiled._make_window_1d(h, fade_top, fade_bot, device)
        win_w = LTXVLatentUpsamplerTiled._make_window_1d(w, fade_left, fade_right, device)
        return win_h.unsqueeze(1) * win_w.unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "LTXVLatentUpsamplerTiled": LTXVLatentUpsamplerTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVLatentUpsamplerTiled": "\U0001f50d LTX Latent Upsampler (Tiled)",
}
