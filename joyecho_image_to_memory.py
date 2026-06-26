"""
joyecho_image_to_memory.py

Build a 'fake shot 0' memory bank entry from a single image, suitable for
i2v identity conditioning via JoyAI-Echo's PairedAudioVideoMemoryBank.

How it works
────────────
Echo's PairedAudioVideoMemoryBank.save_memory_slot() requires a non-None
audio_latent on every entry (hardcoded raise in _prepare_audio_latent).
We provide a zero-tensor placeholder for the audio side. When the
downstream JoyEcho_SingleShotGenerate runs with enable_audio_memory=False,
build_paired_audio_memory_kwargs returns an empty dict and the audio
side is bypassed at the model call — the placeholder never reaches the
transformer.

What does reach the transformer is the video memory, derived from the
PIL frame list we stored. The frames are a single image replicated N
times. The transformer attends to this memory as an identity reference
for the next generated shot.

Output shape
────────────
Returns {"bank": PairedAudioVideoMemoryBank} as a JOYECHO_MEMORY type —
plugs directly into JoyEcho_SingleShotGenerate's 'memory' input.

Critical setup
──────────────
- Set enable_audio_memory=False on JoyEcho_SingleShotGenerate. Its default
  is True, which would try to consume our placeholder audio and fail.
- Consider trimming the first 2–3 frames of the generated video. They
  may be transitional as the model reconciles 'static memory' with
  'starting motion'. Beyond that the output stabilizes.

Dependencies
────────────
Requires ltx_distillation (installs with the ComfyUI_JoyAI_Echo custom
node package). If those deps aren't present, this node loads but errors
clearly on use.
"""

from __future__ import annotations

import torch
import numpy as np


# Conditional imports — let the node module load even if JoyAI deps missing
try:
    from ltx_distillation.inference.memory_multishot import (
        PairedAudioVideoMemoryBank,
    )
    from PIL import Image
    _DEPS_OK = True
    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    PairedAudioVideoMemoryBank = None  # type: ignore
    Image = None  # type: ignore
    _DEPS_OK = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


def _comfy_image_to_pil(image: torch.Tensor):
    """Convert a Comfy IMAGE tensor (B, H, W, C) in [0,1] → a PIL.Image.

    If batch > 1, uses the first item.
    """
    if image.dim() != 4:
        raise ValueError(
            f"Expected IMAGE tensor of shape (B, H, W, C), got {tuple(image.shape)}"
        )
    arr = image[0].detach().cpu().float().numpy()  # (H, W, C) in [0, 1]
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


class JoyEcho_ImageToMemoryShot:
    """Build a 'fake shot 0' memory bank entry from a single image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "video_clip_num_frames": ("INT", {
                    "default": 9, "min": 3, "max": 33, "step": 1,
                    "tooltip": "Frames stored in the fake shot 0. Echo's "
                               "official inference uses 9. First 2-3 frames "
                               "of generated output may be transitional and "
                               "worth trimming downstream.",
                }),
                "audio_window_size": ("INT", {
                    "default": 32, "min": 1, "max": 1024, "step": 1,
                    "tooltip": "Audio window size used by Echo internally for "
                               "the placeholder audio_latent. Echo's memory "
                               "bank rejects None for audio_latent, hence the "
                               "placeholder. Contents are never consumed when "
                               "downstream enable_audio_memory=False.",
                }),
                "memory_max_size": ("INT", {
                    "default": 7, "min": 1, "max": 31, "step": 1,
                    "tooltip": "Memory bank capacity. Default 7 matches "
                               "Echo's official config. Irrelevant for "
                               "single-shot use (we only store 1 entry).",
                }),
                "num_fix_frames": ("INT", {
                    "default": 3, "min": 0, "max": 31, "step": 1,
                    "tooltip": "Non-evictable slots at the bank start. "
                               "Default 3 matches Echo's official config. "
                               "Irrelevant for single-shot use; kept for "
                               "compatibility with multi-shot chaining.",
                }),
            },
        }

    RETURN_TYPES = ("JOYECHO_MEMORY",)
    RETURN_NAMES = ("memory",)
    FUNCTION = "build_memory"
    CATEGORY = "10S Nodes/JoyAI Echo"
    DESCRIPTION = (
        "Build a 'fake shot 0' memory bank entry from a single image. "
        "Wire output to JoyEcho_SingleShotGenerate's 'memory' input. "
        "CRITICAL: set enable_audio_memory=False on that node — its "
        "default of True will try to consume the zero-tensor placeholder "
        "audio_latent we provide here. The image becomes the identity "
        "reference for the generated shot. Consider trimming first 2-3 "
        "frames of output (may be transitional). Requires the "
        "ltx_distillation package (from ComfyUI_JoyAI_Echo)."
    )

    def build_memory(
        self,
        image,
        video_clip_num_frames=9,
        audio_window_size=32,
        memory_max_size=7,
        num_fix_frames=3,
    ):
        if not _DEPS_OK:
            raise RuntimeError(
                f"[JoyEcho ImageToMemory] JoyAI-Echo deps not installed. "
                f"This node requires the 'ltx_distillation' Python package, "
                f"which comes with the ComfyUI_JoyAI_Echo custom node pack. "
                f"Install that pack first. Original ImportError: "
                f"{_IMPORT_ERROR}"
            )

        # Convert IMAGE tensor → PIL Image, then replicate to N frames
        pil_image = _comfy_image_to_pil(image)
        pil_frames = [pil_image.copy() for _ in range(int(video_clip_num_frames))]

        # Build placeholder audio_latent — zero tensor of shape [1, T, 1].
        # Echo's _prepare_audio_latent requires a 3D tensor (B, T, C) but
        # the contents never reach the model when downstream sets
        # enable_audio_memory=False. Any 3D shape works; we use minimal.
        placeholder_audio = torch.zeros(
            1, max(int(audio_window_size), 1), 1,
            dtype=torch.float32,
        )

        # Construct the memory bank with Echo's standard config
        memory_bank = PairedAudioVideoMemoryBank(
            max_size=int(memory_max_size),
            save_mode="random_every_shot_frame",
            num_fix_frames=int(num_fix_frames),
        )

        # Save the fake shot 0 entry. audio_waveform=None forces the
        # simpler center-window selection path that doesn't need real audio.
        metadata = memory_bank.save_memory_slot(
            pil_frames,
            placeholder_audio,
            audio_window_size=int(audio_window_size),
            video_clip_num_frames=int(video_clip_num_frames),
            audio_waveform=None,
            audio_sample_rate=16000,
            video_fps=24.0,
            audio_window_selection_mode="center",
            video_frame_selection_mode="center",
        )

        print(
            f"[JoyEcho ImageToMemory] Created shot-0 entry: "
            f"frames={len(pil_frames)}, image_size={pil_image.size}, "
            f"bank_size={len(memory_bank)}/{memory_max_size}"
        )
        print(
            "  \u26a0 IMPORTANT: set enable_audio_memory=False on "
            "JoyEcho_SingleShotGenerate. This node uses a placeholder "
            "audio_latent; enabling audio memory will try to consume it "
            "and produce garbage."
        )

        return ({"bank": memory_bank},)


NODE_CLASS_MAPPINGS = {
    "JoyEcho_ImageToMemoryShot": JoyEcho_ImageToMemoryShot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyEcho_ImageToMemoryShot": "\U0001f3b4 JoyEcho Image \u2192 Memory Shot",
}
