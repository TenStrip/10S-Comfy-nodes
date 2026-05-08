"""
LTX Latent Color Restore v1.0

Per-channel statistics matching node. Use to undo color shifts introduced
by the LTX upscale model (or any latent-domain transformation that drifts
the per-channel distribution).

================================================================================
WHAT IT DOES
================================================================================
Takes two LATENT inputs:
  samples    — the latent that has color drift (e.g., output of LTX upscaler)
  reference  — the latent with correct color distribution (e.g., the
               pre-upscale latent or any reference with proper colors)

For each of the 128 VAE channels independently, the node:
  1. Computes the reference's per-channel mean and std (across all spatial
     positions and frames).
  2. Computes the samples' per-channel mean and std.
  3. Re-scales samples so their per-channel mean and std match the
     reference's.

Spatial detail in samples is fully preserved (we only shift and rescale,
not reshape). Frame-to-frame variation is preserved (statistics are
computed across frames, not per-frame, to avoid washing out legitimate
motion-driven color changes).

================================================================================
WHY THIS HELPS
================================================================================
The LTX upscale model produces output whose per-channel distribution is
slightly shifted from the input distribution. The shift is roughly uniform
across the spatial extent — i.e., the model makes the whole image
slightly warmer (or cooler, etc.) regardless of content.

This is exactly what per-channel mean/std matching corrects. The spatial
detail the upscaler added stays; the global tonal drift is undone.

================================================================================
WHEN TO USE
================================================================================
Best use case: post-LTX-upscaler color correction.

  Original latent (pre-upscale)
        |
        +-> reference input
        |
  LTX upsampler (tiled or otherwise)
        |
        v
  samples input
        |
        v
  LTX Latent Color Restore
        |
        v
  KSampler / further processing

================================================================================
PARAMETERS
================================================================================
  samples    : LATENT to color-correct (the upscaled latent)
  reference  : LATENT with correct colors (the pre-upscale latent)

  strength   : 0.0 = no correction (passthrough)
               1.0 = full per-channel matching (default)
               In between: weighted blend of original and corrected.

  scope      : 'global'    : statistics across all frames + positions
                            (default; preserves frame-to-frame variation)
               'per_frame' : statistics computed per frame
                            (matches each frame's distribution to the
                            reference's overall distribution; more
                            aggressive)

  debug      : Verbose logging of computed statistics.
"""

import torch


class LTXLatentColorRestore:
    """
    Per-channel statistics matching to undo upscaler-induced color shifts.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples":   ("LATENT",),
                "reference": ("LATENT",),
            },
            "optional": {
                "strength": ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scope":    (["global", "per_frame"], {"default": "global"}),
                "debug":    ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "restore"
    CATEGORY = "10S Nodes/Latent"
    DESCRIPTION = (
        "Match per-channel statistics of a latent to a reference latent. Use after "
        "LTX upsampler to undo color drift while preserving spatial detail. "
        "Reference is typically the pre-upscale latent."
    )

    def restore(self, samples, reference, strength=1.0, scope="global", debug=False):
        in_lat = samples["samples"]
        ref_lat = reference["samples"]

        if debug:
            print(f"\u2192 [10S] LatentColorRestore: "
                  f"samples shape={tuple(in_lat.shape)} dtype={in_lat.dtype}")
            print(f"  \u00b7 reference shape={tuple(ref_lat.shape)} dtype={ref_lat.dtype}")
            print(f"  \u00b7 strength={strength} scope={scope}")

        # Both should be 5D (B, C, F, H, W) for video latents
        if in_lat.dim() != 5 or ref_lat.dim() != 5:
            print(f"\u2192 [10S] LatentColorRestore: expected 5D latents, got "
                  f"input.dim={in_lat.dim()} ref.dim={ref_lat.dim()}; passing through")
            return (samples,)

        if in_lat.shape[1] != ref_lat.shape[1]:
            print(f"\u2192 [10S] LatentColorRestore: channel mismatch "
                  f"({in_lat.shape[1]} vs {ref_lat.shape[1]}); passing through")
            return (samples,)

        if strength <= 0.0:
            return (samples,)

        # Cast both to float32 for accurate statistics
        orig_dtype = in_lat.dtype
        in_f = in_lat.float()
        ref_f = ref_lat.float()

        # Reduce dims for statistics computation
        if scope == "global":
            # Statistics across (B, F, H, W) for each channel C
            # Result shape: (1, C, 1, 1, 1) — broadcasts across all positions
            reduce_dims = (0, 2, 3, 4)
        else:  # per_frame
            # Statistics per frame: across (B, H, W) for each (C, F)
            # Result shape: (1, C, F, 1, 1) — same frame index uses same stats
            reduce_dims = (0, 3, 4)

        in_mean = in_f.mean(dim=reduce_dims, keepdim=True)
        in_std = in_f.std(dim=reduce_dims, keepdim=True) + 1e-6

        ref_mean = ref_f.mean(dim=reduce_dims, keepdim=True)
        ref_std = ref_f.std(dim=reduce_dims, keepdim=True) + 1e-6

        # Normalise input then rescale to reference distribution
        normalized = (in_f - in_mean) / in_std
        restored = normalized * ref_std + ref_mean

        # Strength blend
        if strength < 1.0:
            output = strength * restored + (1.0 - strength) * in_f
        else:
            output = restored

        if debug:
            # Compare per-channel mean shifts (showing first 4 channels)
            shifts = (ref_mean - in_mean).flatten()[:4].tolist()
            ratios = (ref_std / in_std).flatten()[:4].tolist()
            print(f"  \u00b7 per-channel mean shift (first 4 ch): "
                  f"{[f'{v:+.4f}' for v in shifts]}")
            print(f"  \u00b7 per-channel std ratio (first 4 ch): "
                  f"{[f'{v:.4f}' for v in ratios]}")
            # Aggregate magnitude of correction
            mean_shift_magnitude = (ref_mean - in_mean).abs().mean().item()
            std_ratio_dev = (ref_std / in_std - 1.0).abs().mean().item()
            print(f"  \u00b7 correction magnitude: "
                  f"avg_mean_shift={mean_shift_magnitude:.4f} "
                  f"avg_std_dev_from_1={std_ratio_dev:.4f}")
            if mean_shift_magnitude > 0.5:
                print(f"  \u26a0  large mean shifts detected \u2014 confirms color drift "
                      f"in source. Correction is meaningful.")
            elif mean_shift_magnitude < 0.01:
                print(f"  \u26a0  very small mean shifts \u2014 source has minimal "
                      f"color drift, correction will be near-passthrough.")

        output = output.to(dtype=orig_dtype)

        return_dict = samples.copy()
        return_dict["samples"] = output
        if debug:
            print(f"\u2192 [10S] LatentColorRestore: output shape={tuple(output.shape)}")
        return (return_dict,)


NODE_CLASS_MAPPINGS = {
    "LTXLatentColorRestore": LTXLatentColorRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLatentColorRestore": "\U0001f3a8 LTX Latent Color Restore",
}
