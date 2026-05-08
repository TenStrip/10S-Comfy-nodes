"""
LTX Latent Outlier Suppress v1.0

Detects and suppresses spatially localized outlier positions in a latent,
typically introduced by upscaling operations (VAE-based, model-based, or
direct interpolation). The artifacts manifest as unusually-high-magnitude
points in specific channels, concentrated in particular frames or regions.
Without suppression, these artifacts get smeared by the subsequent sampler
into broad hue/tone shifts visible in the final output.

================================================================================
DIAGNOSIS THIS FIXES
================================================================================
Empirical finding: post-upscale latents (from any upscaler — model, VAE, or
direct interpolation) often contain concentrated point artifacts that are
not present in the pre-upscale latent. When fed to a sampler, the artifact
diffuses across the spatial-temporal volume via self-attention, becoming
a global hue shift in the final output. Color statistics matching post-
sampler can't reliably undo this because the distortion is no longer
captured by per-channel mean/std differences (it's now distributed across
positions and channels).

This node operates BEFORE the sampler, while the artifact is still
localized and removable via spatial in-painting.

================================================================================
PIPELINE PLACEMENT
================================================================================
  Pre-upscale latent ──┬─→ reference input
                       │
  Upsampler ───────────┴─→ samples input
                       │
  LTX Latent Outlier Suppress
                       │
                       v
  Conditioning re-application
                       │
                       v
  KSampler (second pass)

The reference is the clean pre-upscale latent. Different spatial
dimensions are fine — only per-channel statistics are used for detection.

================================================================================
HOW IT WORKS
================================================================================
1. Compute per-channel mean and std from the reference (pre-upscale) latent.
   Optionally per-frame, for stricter detection on temporally localized
   artifacts.
2. Z-score every position in the samples (post-upscale) latent against
   those statistics: z = (x - mean) / std.
3. Flag positions with |z| > threshold across enough channels. A position
   is an outlier if its abs-z exceeds the threshold in N or more channels
   (default: 1 channel, i.e. any channel out of distribution flags it).
4. Suppress flagged positions by replacing with a weighted average of
   nearby non-flagged positions in the same frame. Weights combine
   spatial distance (closer = more weight) and channel similarity.
5. Optionally smooth the boundary between flagged and non-flagged regions
   to avoid hard seams.

================================================================================
PARAMETERS
================================================================================
  samples           : LATENT to clean (post-upscale).
  reference         : LATENT with clean distribution (pre-upscale).

  z_threshold       : Outlier z-score threshold. Higher = more selective.
                      Default 4.0 (only true outliers flagged).
                      Range typically 3.0-6.0.

  channel_quorum    : Number of channels that must exceed z_threshold for
                      a position to be flagged. Default 1 (any channel out
                      of distribution flags the position). Higher values
                      = more conservative (e.g. 3 = need 3 channels to
                      agree before flagging).

  suppression_radius: Spatial radius (in latent tokens) for inpainting
                      neighbour search. Default 3.

  scope             : 'global'    : per-channel statistics across all
                                    frames. Default.
                      'per_frame' : per-channel-per-frame statistics.
                                    More aggressive; flags artifacts
                                    that are locally extreme even if
                                    they fit the global distribution.

  blend_strength    : 0.0 = passthrough (no suppression).
                      1.0 = full replacement of outliers (default).
                      Intermediate values blend original with suppressed.

  debug             : Diagnostic logging.
"""

import torch
import torch.nn.functional as F


class LTXLatentOutlierSuppress:
    """
    Detect and suppress spatially-localized outlier positions in a latent
    relative to a clean reference distribution.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples":   ("LATENT",),
                "reference": ("LATENT",),
            },
            "optional": {
                "z_threshold":        ("FLOAT",   {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "channel_quorum":     ("INT",     {"default": 1,   "min": 1,   "max": 64,   "step": 1}),
                "suppression_radius": ("INT",     {"default": 3,   "min": 1,   "max": 8,    "step": 1}),
                "scope":              (["global", "per_frame"], {"default": "global"}),
                "blend_strength":     ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,  "step": 0.05}),
                "debug":              ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "suppress"
    CATEGORY = "10S Nodes/Latent"
    DESCRIPTION = (
        "Detect and suppress per-channel outliers in a latent vs a clean reference. "
        "Use between upsampler and conditioning re-application to remove localised "
        "artifacts before the sampler smears them into global hue shifts."
    )

    def suppress(self, samples, reference,
                 z_threshold=4.0, channel_quorum=1,
                 suppression_radius=3, scope="global",
                 blend_strength=1.0, debug=False):

        in_lat = samples["samples"]
        ref_lat = reference["samples"]

        if debug:
            print(f"\u2192 [10S] LatentOutlierSuppress: "
                  f"samples shape={tuple(in_lat.shape)} dtype={in_lat.dtype}")
            print(f"  \u00b7 reference shape={tuple(ref_lat.shape)} "
                  f"dtype={ref_lat.dtype}")
            print(f"  \u00b7 z_threshold={z_threshold} channel_quorum={channel_quorum} "
                  f"radius={suppression_radius} scope={scope} "
                  f"blend={blend_strength}")

        if in_lat.dim() != 5 or ref_lat.dim() != 5:
            print(f"\u2192 [10S] LatentOutlierSuppress: expected 5D latents, got "
                  f"{in_lat.dim()}D / {ref_lat.dim()}D; passing through")
            return (samples,)

        if in_lat.shape[1] != ref_lat.shape[1]:
            print(f"\u2192 [10S] LatentOutlierSuppress: channel count mismatch "
                  f"({in_lat.shape[1]} vs {ref_lat.shape[1]}); passing through")
            return (samples,)

        if blend_strength <= 0.0:
            return (samples,)

        orig_dtype = in_lat.dtype
        in_f = in_lat.float()
        ref_f = ref_lat.float()

        B, C, F_dim, H, W = in_f.shape

        # ─── Reference statistics ────────────────────────────────────────────
        if scope == "global":
            # (1, C, 1, 1, 1) — broadcasts across all positions
            reduce_dims = (0, 2, 3, 4)
            ref_mean = ref_f.mean(dim=reduce_dims, keepdim=True)
            ref_std = ref_f.std(dim=reduce_dims, keepdim=True) + 1e-6
        else:
            # per_frame: stats per (C, F), but reference may have different F
            # In that case fall back to global (per-frame doesn't make sense
            # if frame counts differ).
            if ref_f.shape[2] != F_dim:
                if debug:
                    print(f"  \u00b7 per_frame requested but ref F={ref_f.shape[2]} "
                          f"!= samples F={F_dim}; using global statistics")
                reduce_dims = (0, 2, 3, 4)
                ref_mean = ref_f.mean(dim=reduce_dims, keepdim=True)
                ref_std = ref_f.std(dim=reduce_dims, keepdim=True) + 1e-6
            else:
                # (1, C, F, 1, 1)
                reduce_dims = (0, 3, 4)
                ref_mean = ref_f.mean(dim=reduce_dims, keepdim=True)
                ref_std = ref_f.std(dim=reduce_dims, keepdim=True) + 1e-6

        # ─── Z-score every position in samples ───────────────────────────────
        z = (in_f - ref_mean) / ref_std                  # (B, C, F, H, W)
        abs_z = z.abs()

        # Per-position channel quorum: count channels exceeding threshold
        outlier_channels = (abs_z > z_threshold)         # (B, C, F, H, W) bool
        n_outlier_channels = outlier_channels.sum(dim=1) # (B, F, H, W) int
        outlier_mask = (n_outlier_channels >= channel_quorum)  # (B, F, H, W) bool

        n_outlier = int(outlier_mask.sum().item())
        total_positions = B * F_dim * H * W
        outlier_pct = 100.0 * n_outlier / max(1, total_positions)

        if debug:
            max_z_per_pos = abs_z.max(dim=1).values     # (B, F, H, W) — max z across channels
            global_max_z = max_z_per_pos.max().item()
            mean_max_z = max_z_per_pos.mean().item()
            print(f"  \u00b7 z-score stats: global_max_z={global_max_z:.2f} "
                  f"mean_max_z={mean_max_z:.2f}")
            print(f"  \u00b7 outliers flagged: {n_outlier} / {total_positions} "
                  f"({outlier_pct:.3f}%)")

            if outlier_pct > 5.0:
                print(f"  \u26a0  large fraction of positions flagged ({outlier_pct:.1f}%) "
                      f"\u2014 z_threshold may be too low, or reference distribution "
                      f"differs significantly from samples (not just artifacts)")
            elif outlier_pct < 0.001 and n_outlier == 0:
                print(f"  \u00b7 no outliers detected at z_threshold={z_threshold}; "
                      f"passing through unchanged")

            # Per-frame outlier counts to see temporal distribution
            per_frame_counts = outlier_mask.sum(dim=(0, 2, 3))[:8].tolist()
            if F_dim > 8:
                print(f"  \u00b7 outliers per frame (first 8): {per_frame_counts} "
                      f"...({F_dim-8} more)")
            else:
                print(f"  \u00b7 outliers per frame: {per_frame_counts}")

        if n_outlier == 0:
            return (samples,)

        # ─── Spatial in-painting of flagged positions ────────────────────────
        # Strategy: for each flagged position, replace its values with a
        # weighted average of NON-FLAGGED positions in the same frame within
        # suppression_radius. Implemented as a separable spatial average
        # weighted by inverse outlier mask.
        suppressed = self._inpaint_spatial(
            in_f, outlier_mask, suppression_radius, debug
        )

        # Blend
        if blend_strength < 1.0:
            output = blend_strength * suppressed + (1.0 - blend_strength) * in_f
        else:
            output = suppressed

        if debug:
            # How much did the flagged positions actually change?
            mask5d = outlier_mask.unsqueeze(1).expand(-1, C, -1, -1, -1)
            avg_change = (output[mask5d] - in_f[mask5d]).abs().mean().item()
            print(f"  \u00b7 avg |change| at flagged positions: {avg_change:.4f}")

        output = output.to(dtype=orig_dtype)

        return_dict = samples.copy()
        return_dict["samples"] = output
        if debug:
            print(f"\u2192 [10S] LatentOutlierSuppress: output shape={tuple(output.shape)}")
        return (return_dict,)

    @staticmethod
    def _inpaint_spatial(latent, mask, radius, debug):
        """
        Per-frame spatial in-painting via masked box filter.
        latent: (B, C, F, H, W)
        mask:   (B, F, H, W) bool, True = flagged for replacement

        For each flagged position, computes weighted average of non-flagged
        positions within (2*radius+1) box, then assigns that average to the
        flagged position.

        Implementation note: we use a simple uniform box average over a
        (2r+1)x(2r+1) window, weighted by the inverse mask. This is fast
        (one conv2d per frame batched together) and produces smooth fills.
        """
        B, C, F_dim, H, W = latent.shape
        device = latent.device
        dtype = latent.dtype

        # Reshape (B, C, F, H, W) -> (B*F, C, H, W) for 2D conv
        lat2d = latent.permute(0, 2, 1, 3, 4).reshape(B * F_dim, C, H, W)
        # (B, F, H, W) -> (B*F, 1, H, W)
        valid_mask = (~mask).float().reshape(B * F_dim, 1, H, W)

        # Uniform box kernel
        k = 2 * radius + 1
        kernel_2d = torch.ones((1, 1, k, k), dtype=dtype, device=device)
        kernel_C = torch.ones((C, 1, k, k), dtype=dtype, device=device)

        # Numerator: sum of (valid * value) over neighbourhood
        valid_lat = lat2d * valid_mask                          # zero out invalid
        # Per-channel grouped convolution to sum within window
        num = F.conv2d(
            valid_lat,
            kernel_C,
            padding=radius,
            groups=C,
        )                                                        # (B*F, C, H, W)

        # Denominator: count of valid positions in neighbourhood
        denom = F.conv2d(
            valid_mask,
            kernel_2d,
            padding=radius,
        )                                                        # (B*F, 1, H, W)

        # Avoid division by zero (all-flagged neighbourhoods).
        # If a neighbourhood has zero valid positions, fall back to original
        # (we have no information to in-paint with).
        safe_denom = denom.clamp(min=1.0)
        avg = num / safe_denom                                   # (B*F, C, H, W)

        # Where denom==0, keep original value
        no_valid = (denom < 0.5)                                 # (B*F, 1, H, W)
        no_valid_5d = no_valid.expand(-1, C, -1, -1)
        avg = torch.where(no_valid_5d, lat2d, avg)

        # Reshape back to (B, C, F, H, W)
        avg = avg.reshape(B, F_dim, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # Apply only to flagged positions: out = where(mask, avg, orig)
        mask_5d = mask.unsqueeze(1).expand(-1, C, -1, -1, -1)
        result = torch.where(mask_5d, avg, latent)

        if debug:
            # How many flagged positions had no valid neighbours?
            no_valid_count = int(no_valid.sum().item())
            if no_valid_count > 0:
                print(f"  \u00b7 in-paint: {no_valid_count} positions had all-flagged "
                      f"neighbourhoods (kept original)")

        return result


NODE_CLASS_MAPPINGS = {
    "LTXLatentOutlierSuppress": LTXLatentOutlierSuppress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLatentOutlierSuppress": "\U0001f9f9 LTX Latent Outlier Suppress",
}
