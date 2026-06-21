"""LTX LoRA Stack (AV split).

Consolidates a chain of LoRA Loader Advanced nodes into a single stack
loader with per-LoRA audio/video block-group scaling. Each LoRA in the
stack has three knobs:

  - strength: overall LoRA multiplier
  - audio_weight: scale applied to audio-related layers
  - video_weight: scale applied to video / main DiT layers

Final per-layer scale = strength × (audio_weight if audio_layer else video_weight)

CLASSIFICATION RULE (case-insensitive substring "audio" in model key):

AUDIO bucket — keys containing "audio":
  - audio_attn1, audio_attn2          (audio self/cross attention)
  - audio_ff                          (audio feed-forward)
  - audio_adaln_single                (audio adaLN modulation)
  - audio_patchify_proj               (audio input projection)
  - audio_proj_out                    (audio output projection)
  - audio_prompt_adaln_single         (audio prompt modulation)
  - audio_embeddings_connector.*      (audio routing connector)
  - audio_to_video_attn               (A→V cross-attention)
  - video_to_audio_attn               (V→A cross-attention; contains "audio")

VIDEO bucket — everything else:
  - transformer_blocks.{i}.attn1      (video self-attention)
  - transformer_blocks.{i}.attn2      (text→video cross-attention)
  - transformer_blocks.{i}.ff         (video feed-forward)
  - adaln_single                      (video adaLN, not the audio variant)
  - prompt_adaln_single               (text→video prompt modulation)
  - patchify_proj                     (video input projection)
  - proj_out                          (video output projection)
  - video_embeddings_connector.*      (video routing connector)

The substring rule is exact — V→A cross-attention contains "audio" in
its name, so it correctly lands in the AUDIO bucket. The video connector
contains "video" but not "audio", so it stays in VIDEO. If you want to
group video_embeddings_connector into AUDIO instead, edit the classifier
in this file.
"""

import folder_paths
import comfy.utils
import comfy.lora


class LTXLoraStackAV:
    """LoRA stack loader with separate audio/video block-group scaling."""

    MAX_LORAS = 12

    @classmethod
    def INPUT_TYPES(cls):
        try:
            lora_list = folder_paths.get_filename_list("loras")
        except Exception:
            lora_list = []

        required = {
            "model": ("MODEL",),
            "num_loras": ("INT", {
                "default": 1, "min": 1, "max": cls.MAX_LORAS, "step": 1,
                "tooltip": "How many LoRA slots to actually use. Only the "
                           f"first N slots (1-{cls.MAX_LORAS}) are applied; "
                           "the rest are ignored entirely regardless of "
                           "their values. Bump this up as you add more "
                           "LoRAs to your stack.",
            }),
        }
        optional = {}
        for i in range(1, cls.MAX_LORAS + 1):
            optional[f"lora_{i}"] = (
                ["None"] + lora_list,
                {
                    "default": "None",
                    "tooltip": f"LoRA file for slot {i}. Set to 'None' to "
                               f"disable this slot.",
                },
            )
            optional[f"strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Overall LoRA strength multiplier. Combined "
                               "with audio_weight and video_weight per layer.",
                },
            )
            optional[f"audio_weight_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Audio-block scale. Multiplied by strength to "
                               "produce final scale for audio layers: "
                               "audio_attn, audio_ff, audio_adaln, "
                               "audio_embeddings_connector, and both A↔V "
                               "cross-attention pairs. Set to 0 to disable "
                               "audio-side adjustments from this LoRA.",
                },
            )
            optional[f"video_weight_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Video-block scale. Multiplied by strength to "
                               "produce final scale for video layers: main "
                               "DiT transformer_blocks (attn1/attn2/ff), "
                               "video_embeddings_connector, and video "
                               "adaLN modulation. Set to 0 to disable "
                               "video-side adjustments from this LoRA.",
                },
            )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "10S Nodes/LoRA"
    DESCRIPTION = (
        "Stack up to 12 LoRAs onto an LTX2 model with separate audio and "
        "video block-group scaling per LoRA. Replaces a chain of LoRA "
        "Loader Advanced nodes. Per-layer effective strength is: "
        "strength × (audio_weight if layer is audio else video_weight). "
        "Audio classification: keys containing 'audio' substring "
        "(catches audio_attn, audio_ff, audio_adaln, both cross-attention "
        "directions, and the audio embeddings connector). Set num_loras "
        "to control how many slots are actually used — slots beyond that "
        "count are ignored regardless of their values."
    )

    # ─── Classification ─────────────────────────────────────────────────────

    @staticmethod
    def _is_audio_key(key: str) -> bool:
        """True if the model key represents an audio-pathway layer."""
        return "audio" in key.lower()

    # ─── LoRA application ───────────────────────────────────────────────────

    def _apply_lora_with_split(self, model, lora_name, strength,
                                 audio_weight, video_weight, debug=False):
        """Apply one LoRA, splitting its patches into audio/video buckets
        and applying each bucket with its own scaled strength."""
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            print(f"\u2192 [LTX LoRA Stack] LoRA file not found in loras "
                  f"folder: '{lora_name}'")
            return model

        try:
            lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except Exception as e:
            print(f"\u2192 [LTX LoRA Stack] Failed to load '{lora_name}': "
                  f"{type(e).__name__}: {e}")
            return model

        # Build model→LoRA key mapping. This translates LoRA-style key
        # prefixes (e.g., 'diffusion_model.transformer_blocks.0.attn1.to_q')
        # into model state_dict keys.
        try:
            key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        except Exception as e:
            print(f"\u2192 [LTX LoRA Stack] Couldn't build key map for "
                  f"'{lora_name}': {type(e).__name__}: {e}")
            return model

        # Convert LoRA tensors into Comfy's patch format.
        # load_lora returns {model_state_dict_key: (lora_tuple, ...), ...}
        try:
            loaded_lora = comfy.lora.load_lora(lora_dict, key_map)
        except Exception as e:
            print(f"\u2192 [LTX LoRA Stack] Couldn't parse LoRA structure "
                  f"for '{lora_name}': {type(e).__name__}: {e}")
            return model

        if not loaded_lora:
            print(f"\u2192 [LTX LoRA Stack] '{lora_name}' produced 0 matching "
                  f"patches — LoRA keys don't align with this model. Skipping.")
            return model

        # Split patches by classification rule
        audio_patches = {}
        video_patches = {}
        for key, patch in loaded_lora.items():
            if self._is_audio_key(key):
                audio_patches[key] = patch
            else:
                video_patches[key] = patch

        n_audio = len(audio_patches)
        n_video = len(video_patches)
        audio_scale = strength * audio_weight
        video_scale = strength * video_weight

        print(f"\u2192 [LTX LoRA Stack] {lora_name}: "
              f"audio={n_audio} layers \u00d7 {audio_scale:+.3f} | "
              f"video={n_video} layers \u00d7 {video_scale:+.3f}")

        # Apply each bucket with its own strength. Skip empty / zero-scale
        # buckets to avoid no-op patch entries.
        if audio_patches and audio_scale != 0.0:
            model.add_patches(audio_patches, audio_scale)
        elif audio_patches and debug:
            print(f"  \u00b7 (audio bucket has {n_audio} layers but "
                  f"audio_scale=0; skipping)")

        if video_patches and video_scale != 0.0:
            model.add_patches(video_patches, video_scale)
        elif video_patches and debug:
            print(f"  \u00b7 (video bucket has {n_video} layers but "
                  f"video_scale=0; skipping)")

        return model

    # ─── Entry point ────────────────────────────────────────────────────────

    def load_loras(self, model, num_loras=1, **kwargs):
        # Clone the model so we don't mutate the upstream ModelPatcher
        model = model.clone()

        # Clamp num_loras into valid range (defensive — the UI already
        # enforces this, but workflow files could carry out-of-range values)
        num_loras = max(1, min(int(num_loras), self.MAX_LORAS))

        applied_count = 0
        skipped_inactive = 0
        for i in range(1, self.MAX_LORAS + 1):
            if i > num_loras:
                # Beyond the user-specified count — silently skip regardless
                # of slot values. This makes the count input the single
                # source of truth for "how many LoRAs are active."
                skipped_inactive += 1
                continue

            lora_name = kwargs.get(f"lora_{i}", "None")
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            audio_weight = float(kwargs.get(f"audio_weight_{i}", 1.0))
            video_weight = float(kwargs.get(f"video_weight_{i}", 1.0))

            if lora_name == "None":
                continue
            if strength == 0.0:
                continue
            if audio_weight == 0.0 and video_weight == 0.0:
                continue

            model = self._apply_lora_with_split(
                model, lora_name, strength, audio_weight, video_weight,
            )
            applied_count += 1

        if applied_count == 0:
            print(f"\u2192 [LTX LoRA Stack] No LoRAs applied "
                  f"(num_loras={num_loras}, all active slots are 'None' "
                  f"or zero-strength; {skipped_inactive} slots beyond "
                  f"num_loras ignored)")

        return (model,)


NODE_CLASS_MAPPINGS = {
    "LTXLoraStackAV": LTXLoraStackAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLoraStackAV": "\U0001f39a\ufe0f LTX LoRA Stack (AV)",
}
