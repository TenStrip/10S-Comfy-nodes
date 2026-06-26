"""
ltx_reference_conditioning.py — Encode an image to a reference latent
and attach it to an LTX2.3 MODEL for prefix-injection during sampling.

Companion to ltx_reference_enable.py — that node patches the model to
accept the reference latent; this node provides the latent.

Workflow:
    [Load Diffusion Model (any LTX2.3)] → MODEL ──→ LTX Reference Enable ──→ MODEL'
                                                                             │
    [Load VAE (LTX2 Video)]      → VAE ──┐                                   │
    [Load Image]                 → IMAGE ─┴→ LTX Reference Conditioning ─────┤
                                                                             ▼
                            [Your existing sampler chain — i2v latent
                             conditioning, KSampler, etc., all work
                             alongside this]

VAE format note:
    Comfy's VAE wrapper expects standard IMAGE format (B, H, W, C) in
    [0, 1] range. We pad spatial dims to multiples of 32 (the VAE
    downscale factor) and pass directly to vae.encode — the wrapper
    handles channels-first conversion, range normalization, and
    temporal dim arrangement for video VAEs internally.

Pairs especially well with:
    - Standard LTX2 i2v frame_0 latent conditioning (this gives the
      spatial anchor; reference tokens reinforce identity)
    - Existing latent_anchor / likeness_anchor nodes (orthogonal
      mechanisms — different intervention points in the model)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def _pad_image_to_multiple(image_bhwc: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    """Pad (B, H, W, C) image so H and W are multiples of divisor."""
    if image_bhwc.dim() != 4:
        raise ValueError(
            f"Expected IMAGE tensor (B, H, W, C), got {tuple(image_bhwc.shape)}"
        )
    B, H, W, C = image_bhwc.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return image_bhwc
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous()


class LTXReferenceConditioning:
    """Encode an image to a reference latent and attach to MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "target_latent": ("LATENT", {
                    "tooltip": "Optional. Wire the same LATENT that goes "
                               "into your sampler. The image will be "
                               "resized in pixel space to match this "
                               "latent's spatial dimensions before VAE "
                               "encoding, guaranteeing memory and target "
                               "have matching patches-per-frame. Without "
                               "this, you'll get a tensor mismatch error "
                               "if your image and target latent don't "
                               "match exactly.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the reference latent magnitude. "
                               "1.0 = native VAE output. <1.0 reduces "
                               "influence; >1.0 boosts (may distort). "
                               "Set to 0.0 to bypass completely and clear "
                               "any prior reference state.",
                }),
                "position_mode": (["reference", "prefix_continuous"], {
                    "default": "reference",
                    "tooltip": "How reference tokens are positioned in the "
                               "attention sequence. 'reference' (default): "
                               "tokens overlap target's first frame "
                               "temporally — uniform identity influence "
                               "across all generated frames. "
                               "'prefix_continuous': tokens placed before "
                               "target temporally — equivalent to standard "
                               "i2v prior-context conditioning.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print detailed per-call info to the console. "
                               "Useful for debugging. Disable for normal use.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attach"
    CATEGORY = "10S Nodes/LTX2"
    DESCRIPTION = (
        "Encodes an image through the LTX2 video VAE and attaches the "
        "resulting latent to the MODEL as a reference for prefix "
        "injection during sampling. Requires LTX Reference Enable "
        "upstream. Works on any LTX2.3 checkpoint (vanilla, Echo, "
        "merged variants). Complementary to standard i2v latent "
        "conditioning and existing latent_anchor/likeness_anchor "
        "approaches — these are different intervention points and "
        "can be combined."
    )

    def attach(self, model, vae, image, target_latent=None,
               strength=1.0, position_mode="reference", verbose=False):
        # Bypass + clear state
        if strength == 0.0:
            print("[LTX Reference Conditioning] strength=0, bypassing and clearing state.")
            model = model.clone()
            if hasattr(model, "model_options") and isinstance(model.model_options, dict):
                to = model.model_options.get("transformer_options")
                if isinstance(to, dict):
                    to.pop("reference_latent", None)
                    to.pop("reference_position_mode", None)
                    # Backwards compat: also clear old Echo-branded keys
                    to.pop("memory_video", None)
                    to.pop("memory_position_mode", None)
            try:
                dm = model.model.diffusion_model
                for attr in ("_ltx_reference_latent", "_echo_memory_video"):
                    if hasattr(dm, attr):
                        delattr(dm, attr)
            except Exception:
                pass
            return (model,)

        # Standard Comfy IMAGE format
        if image.shape[-1] > 3:
            image = image[..., :3]

        # If target_latent is wired, resize the image in pixel space to
        # match the target's spatial dimensions BEFORE VAE encoding.
        # This guarantees memory_patches_per_frame == target_patches_per_frame,
        # which is required for the modulation extension in _prepare_timestep
        # to produce a tensor matching the prefixed vx sequence length.
        if target_latent is not None:
            try:
                # Extract latent tensor from Comfy LATENT dict
                tl = target_latent
                if isinstance(tl, dict):
                    tl = tl.get("samples", tl)
                if isinstance(tl, torch.Tensor) and tl.dim() == 5:
                    # (B, C, F, H_lat, W_lat) → pixel dims: H_lat*32, W_lat*32
                    H_lat = int(tl.shape[3])
                    W_lat = int(tl.shape[4])
                    target_H_px = H_lat * 32
                    target_W_px = W_lat * 32

                    img_H = int(image.shape[1])
                    img_W = int(image.shape[2])

                    if (img_H, img_W) != (target_H_px, target_W_px):
                        # (B, H, W, C) → (B, C, H, W) for F.interpolate
                        img_chw = image.permute(0, 3, 1, 2).contiguous()
                        img_chw = F.interpolate(
                            img_chw, size=(target_H_px, target_W_px),
                            mode='bilinear', align_corners=False, antialias=True
                        )
                        image = img_chw.permute(0, 2, 3, 1).contiguous()
                        if verbose:
                            print(
                                f"[LTX Reference Conditioning] Resized image "
                                f"from {img_H}x{img_W} → {target_H_px}x{target_W_px} "
                                f"to match target latent spatial."
                            )
                else:
                    print(
                        f"[LTX Reference Conditioning] WARN: target_latent "
                        f"didn't have expected (B,C,F,H,W) shape "
                        f"(got {type(tl).__name__}). Using image as-is."
                    )
            except Exception as e:
                print(
                    f"[LTX Reference Conditioning] WARN: couldn't use "
                    f"target_latent for resize: {type(e).__name__}: {e}. "
                    f"Using image at its native size."
                )

        # Pad to multiples of 32 (VAE downscale)
        image = _pad_image_to_multiple(image, divisor=32)

        # Encode raw latent through VAE
        try:
            encoded = vae.encode(image)
        except Exception as e:
            raise RuntimeError(
                f"[LTX Reference Conditioning] VAE encode failed: "
                f"{type(e).__name__}: {e}. "
                f"Image shape (B,H,W,C) = {tuple(image.shape)}. "
                f"Ensure VAE is the LTX2 video VAE."
            )

        # Normalize to (B, C, F, H, W) — what the patchifier wants
        if isinstance(encoded, dict):
            for k in ("samples", "latent", "x"):
                if k in encoded:
                    encoded = encoded[k]
                    break

        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError(
                f"[LTX Reference Conditioning] VAE returned non-tensor: "
                f"type={type(encoded).__name__}"
            )

        if encoded.dim() == 5:
            reference_latent = encoded.contiguous()
        elif encoded.dim() == 4:
            reference_latent = encoded.unsqueeze(2).contiguous()
        else:
            raise RuntimeError(
                f"[LTX Reference Conditioning] Unexpected VAE output "
                f"dimensionality: {encoded.dim()}D, shape {tuple(encoded.shape)}"
            )

        # Normalize the latent to the model's expected distribution.
        # The DiT was trained on latents that go through process_latent_in
        # (scale + shift to a standardized distribution). vae.encode
        # produces RAW VAE output before this normalization. During
        # normal sampling, Comfy calls process_latent_in on the noise
        # latent before passing to the diffusion model — but our
        # reference latent bypasses that path, so we apply it here.
        # Without this step, reference tokens have a different magnitude
        # distribution than the target tokens they're prepended to,
        # causing the red-tint / off-color artifacts the user observes.
        try:
            base_model = model.model  # BaseModel wrapping diffusion_model
            if hasattr(base_model, "process_latent_in"):
                ref_pre_norm_stats = (
                    float(reference_latent.mean().item()),
                    float(reference_latent.std().item()),
                )
                reference_latent = base_model.process_latent_in(reference_latent)
                ref_post_norm_stats = (
                    float(reference_latent.mean().item()),
                    float(reference_latent.std().item()),
                )
                if verbose:
                    print(
                        f"[LTX Reference Conditioning] Normalized latent via "
                        f"process_latent_in: "
                        f"mean {ref_pre_norm_stats[0]:.3f}→{ref_post_norm_stats[0]:.3f}, "
                        f"std {ref_pre_norm_stats[1]:.3f}→{ref_post_norm_stats[1]:.3f}"
                    )
            else:
                print(
                    f"[LTX Reference Conditioning] WARN: model has no "
                    f"process_latent_in method. Reference latent stays raw. "
                    f"This may cause distribution mismatch (red tint, "
                    f"off-color artifacts) — the target latent gets "
                    f"normalized during sampling but ours doesn't."
                )
        except Exception as e:
            print(
                f"[LTX Reference Conditioning] WARN: latent normalization "
                f"failed: {type(e).__name__}: {e}. Using raw latent — "
                f"may cause color artifacts."
            )

        if strength != 1.0:
            reference_latent = reference_latent * strength

        # Sanity check
        C_latent = reference_latent.shape[1]
        if C_latent != 128:
            print(
                f"[LTX Reference Conditioning] Warning: VAE produced "
                f"{C_latent} latent channels, expected 128 for LTX2. "
                f"This may indicate a mismatched VAE."
            )

        # Clone model and attach via BOTH channels for reliability
        model = model.clone()
        if not hasattr(model, "model_options") or model.model_options is None:
            model.model_options = {}
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["reference_latent"] = reference_latent
        model.model_options["transformer_options"]["reference_position_mode"] = position_mode
        # Backwards compat: also set old Echo keys so any older patched
        # model still finds the latent
        model.model_options["transformer_options"]["memory_video"] = reference_latent
        model.model_options["transformer_options"]["memory_position_mode"] = position_mode

        try:
            dm = model.model.diffusion_model
            dm._ltx_reference_latent = reference_latent
            dm._echo_memory_video = reference_latent  # backwards compat
        except Exception as e:
            print(f"[LTX Reference Conditioning] Warning: couldn't set "
                  f"attribute side-channel: {e}")

        if verbose:
            print(
                f"[LTX Reference Conditioning] Attached reference_latent: "
                f"shape={tuple(reference_latent.shape)} (B,C,F,H,W), "
                f"mode={position_mode}, strength={strength}"
            )

        return (model,)


class LTXReferenceProbe:
    """Diagnostic — verify the reference-token wiring before sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "probe"
    CATEGORY = "10S Nodes/Debug"
    DESCRIPTION = (
        "Inspects a MODEL after LTX Reference Enable + Conditioning to "
        "verify wiring. Reports patch status, reference latent presence "
        "and shape, position mode. Use before the sampler to catch "
        "config issues without running a generation."
    )

    def probe(self, model):
        lines = []
        lines.append("=" * 70)
        lines.append("[10S] LTX Reference Probe")
        lines.append("=" * 70)

        # Patch status
        try:
            import comfy.ldm.lightricks.av_model as av_module
            LTXAVModel = av_module.LTXAVModel
            pi_name = LTXAVModel._process_input.__name__
            pt_name = LTXAVModel._prepare_timestep.__name__
            patched = pi_name == "_patched_process_input"
            lines.append(f"\nClass-level patches:")
            lines.append(f"  LTXAVModel._process_input    = {pi_name}  "
                         f"{'✓' if patched else '✗ NOT patched'}")
            lines.append(f"  LTXAVModel._prepare_timestep = {pt_name}  "
                         f"{'✓' if pt_name == '_patched_prepare_timestep' else '✗ NOT patched'}")
        except Exception as e:
            lines.append(f"  ✗ Couldn't import LTXAVModel: {e}")

        # Instance state
        try:
            diffusion_model = model.model.diffusion_model
            lines.append(f"\nDiffusion model:")
            lines.append(f"  class: {type(diffusion_model).__name__}")
            patchifier = diffusion_model.patchifier
            ltx_wrapped = getattr(patchifier, "_ltx_ref_wrapped", False)
            echo_wrapped = getattr(patchifier, "_echo_memory_wrapped", False)
            wrapped = ltx_wrapped or echo_wrapped
            wrap_src = "LTX" if ltx_wrapped else ("Echo (legacy)" if echo_wrapped else None)
            lines.append(f"  patchifier.unpatchify wrapped: "
                         f"{'✓ yes (' + wrap_src + ')' if wrapped else '✗ NO — Reference Enable not applied'}")

            zero_flag = (getattr(diffusion_model, "_ltx_zero_ref_timesteps", None)
                         or getattr(diffusion_model, "_echo_zero_memory_timesteps", None))
            lines.append(f"  zero_ref_timesteps: {bool(zero_flag)}")
        except Exception as e:
            lines.append(f"  ✗ Couldn't inspect diffusion_model: {e}")

        # model_options
        lines.append(f"\nmodel.model_options:")
        if not hasattr(model, "model_options") or model.model_options is None:
            lines.append(f"  ✗ not present")
        else:
            mo = model.model_options
            if isinstance(mo, dict):
                to = mo.get("transformer_options")
                if to is None:
                    lines.append(f"  ✗ 'transformer_options' key missing")
                else:
                    lines.append(f"  transformer_options keys: {list(to.keys()) if isinstance(to, dict) else type(to)}")
                    ref = to.get("reference_latent") if isinstance(to, dict) else None
                    if ref is None:
                        ref = to.get("memory_video") if isinstance(to, dict) else None
                    if ref is None:
                        lines.append(f"  ✗ reference_latent (or memory_video) NOT set")
                    else:
                        mode = to.get("reference_position_mode", to.get("memory_position_mode", "unknown"))
                        lines.append(f"  ✓ reference_latent shape: {tuple(ref.shape)} "
                                     f"dtype={ref.dtype} mode={mode}")

        # Attribute side-channel
        try:
            attr = getattr(model.model.diffusion_model, "_ltx_reference_latent", None) \
                or getattr(model.model.diffusion_model, "_echo_memory_video", None)
            lines.append(f"\nAttribute side-channel:")
            if attr is None:
                lines.append(f"  diffusion_model._ltx_reference_latent: not set")
            else:
                lines.append(f"  ✓ shape: {tuple(attr.shape)}")
        except Exception as e:
            lines.append(f"  (couldn't check: {e})")

        lines.append("=" * 70)
        report = "\n".join(lines)
        print(report)
        return (model, report)


class LTXReferenceBypass:
    """Clear reference state from a MODEL — bypass memory injection.

    Use between sampling passes when memory injection isn't wanted for
    a particular pass. Common pattern: apply Reference Conditioning for
    your initial generation pass (where identity injection matters),
    then route the MODEL through Bypass before an upscale/refinement
    sampler where you want the model running clean.

    This clears the reference latent from BOTH model_options and the
    diffusion_model attribute side-channel, so downstream Enable patches
    see no reference and act as passthrough.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "bypass"
    CATEGORY = "10S Nodes/LTX2"
    DESCRIPTION = (
        "Clears any attached reference_latent from MODEL so downstream "
        "samplers run without memory injection. Useful between passes — "
        "e.g., apply Reference Conditioning for the primary generation "
        "pass, then route through Bypass before an upscale tiled sampler "
        "where memory isn't needed (and where the first-frame attention "
        "to memory tokens can introduce minor distortion)."
    )

    def bypass(self, model):
        model = model.clone()
        # Clear from model_options.transformer_options
        if hasattr(model, "model_options") and isinstance(model.model_options, dict):
            to = model.model_options.get("transformer_options")
            if isinstance(to, dict):
                for k in ("reference_latent", "reference_position_mode",
                          "memory_video", "memory_position_mode"):
                    to.pop(k, None)
        # Clear attribute side-channel from diffusion_model
        try:
            dm = model.model.diffusion_model
            for attr in ("_ltx_reference_latent", "_echo_memory_video",
                          "_pending_ref_seq_len", "_pending_ref_frames",
                          "_pending_memory_seq_len", "_pending_memory_frames"):
                if hasattr(dm, attr):
                    try:
                        delattr(dm, attr)
                    except Exception:
                        setattr(dm, attr, None)
        except Exception:
            pass
        print("[LTX Reference Bypass] Cleared reference state from MODEL.")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "LTXReferenceConditioning": LTXReferenceConditioning,
    "LTXReferenceProbe": LTXReferenceProbe,
    "LTXReferenceBypass": LTXReferenceBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXReferenceConditioning": "\U0001f3b4 LTX Reference Conditioning",
    "LTXReferenceProbe": "\U0001f50e LTX Reference Probe",
    "LTXReferenceBypass": "\U0001f6ab LTX Reference Bypass",
}
