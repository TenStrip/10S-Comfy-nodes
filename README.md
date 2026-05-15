# 10S Nodes for ComfyUI

A collection of custom nodes for ComfyUI focused on identity preservation, latent-space stabilization, and **upscale-pass quality** for the **LTX2 video diffusion model** by Lightricks. Nodes operate via PyTorch forward hooks on the DiT backbone — no model retraining required.

> **Compatibility note:** these nodes are specifically tuned for LTX2/LTX-AV (the dual-stream video+audio DiT). They will not work as-is on other diffusion models — the hooks rely on LTX2's specific block structure (`BasicAVTransformerBlock`).

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TenStrip/10S-Comfy-nodes.git 10S_Nodes
```

Restart ComfyUI. The new nodes appear under `10S Nodes/` in the node-add menu.

To update later:
```bash
cd ComfyUI/custom_nodes/10S_Nodes
git pull
```

Optional dependency: **MediaPipe** is used by the Latent Face Detector for face bbox detection. Without it, the detector falls back to OpenCV's Haar cascades (less accurate). Install with:

```bash
pip install mediapipe
```

All other dependencies come from ComfyUI's existing environment (PyTorch, comfy package).

---

## Headline Nodes

These two solve the most common LTX2 quality issues directly. If you're only going to use one or two nodes from this package, these are them.

### 🎲 LTX Tiled Sampler — *the upscale-pass fix*

**Category:** `10S Nodes/Sampling`

Spatially-tiled drop-in replacement for `SamplerCustomAdvanced`. Solves the **broad hue shift / conditioning drift problem** that occurs when running a second sampling pass on an upscaled latent.

**The problem this solves:** when the upscale pass runs on a 2× upscaled latent (4× more spatial tokens), each video token receives ~1/4 the per-token text-conditioning influence the model was trained with. The model also operates outside its trained spatial-token-count range. This manifests as broad hue shifts, color drift, and prompt adherence loss in the upscale-pass output. After exhaustive investigation it became clear: this is induced by the *sampler*, not the upsampler or VAE.

**The mechanism:** split the latent spatially along its longer axis with overlap, sample each tile through the full pipeline at training-distribution token count, blend tiles with cosine-windowed Hann overlap. Each tile stays within the model's "comfortable" extent. Optionally, one chosen tile can carry the audio for the pass, preserving video-audio cross-attention for lipsync.

**Key parameters:**
- `tile_axis` — `auto` (default; splits the longer dimension), `H`, or `W`
- `n_tiles` — number of tiles along the chosen axis (default 2; sufficient for most aspect ratios)
- `tile_overlap` — overlap in latent tokens (default 8; hides seams reliably)
- `max_size_for_no_tile` — auto-skip tiling if both dims are at or below this (default 24)
- `audio_pass` — `passthrough` (audio unchanged) or `tile_carrying` (audio sampled through the carrier tile)
- `audio_carrier_tile` — `first` (top tile, ideal for vertical talking-face content), `middle` (centered for very large outputs), `last`

**Recommended config for vertical talking-face upscale:**
```
tile_axis           = auto
n_tiles             = 2
tile_overlap        = 8
audio_pass          = tile_carrying
audio_carrier_tile  = first
```

**For large landscape outputs (4K/8K):**
```
tile_axis           = auto
n_tiles             = 2 to 4
tile_overlap        = 8
audio_pass          = tile_carrying
audio_carrier_tile  = middle
```

**Compatibility note:** the upscale-pass workflow this targets uses light denoising (typical sigmas like `[0.85, 0.7, 0.4, 0]`) with `euler_ancestral_cfg_pp` and distilled CFG=1. The node is not suited for heavy-denoise-from-pure-noise generation, where per-tile divergence would produce visible seams.

---

### 🎯 LTX Latent Anchor Aware — *content-aware identity stabilization*

**Category:** `10S Nodes/Identity`

Inference-time regularizer that improves prompt + image conditioning adherence, scene composition consistency, and physical sensibility across long sampling chains. Adds optional spatial weighting from an external reference image.

**How it works:** snapshots the model's representation of the anchor frame at a chosen sampling step, then pulls all subsequent computation toward that cached state. The reference image (when connected) provides per-position energy weighting — high-energy regions of the reference get more anchor pull, low-energy regions get less.

**Key parameters:**
- `sigmas` (SIGMAS input) — connect for predictable cache timing
- `strength` — pull magnitude (typical 0.05-0.15)
- `cache_at_step` — sampling step at which to lock the anchor (typical 3-9)
- `similarity_threshold` — cosine sim cutoff (default 0.50)
- `decay_with_distance` — per-frame strength decay (default 0.0)
- `reference_image` (IMAGE) — optional external reference for spatial energy
- `vae` (VAE) — required if `reference_image` is connected
- `energy_threshold` — gating cutoff (default 0.30; 0 = uniform, 0.50 = above-median energy only)

**Recommended starting config:**
```
sigmas               = [connected]
strength             = 0.10
cache_at_step        = 6
similarity_threshold = 0.50
energy_threshold     = 0.30
reference_image      = [face crop or composition reference]
vae                  = [LTX VAE]
```

**Simple/advanced mode:** the simple-mode default exposes essential knobs. Toggle `advanced_mode=True` for `cache_mode`, `forwards_per_step`, `depth_curve`, `block_index_filter`, and other research parameters.

---

## Likeness Suite

These nodes work together to provide face-region identity preservation for prompts the model already partially knows. **For unique faces unknown to the base model, a subject-specific LoRA trained via LTX-Video-Trainer (~30 images, ~15 minutes) provides significantly better quality than any inference-time method in this package.** The Likeness suite is best used as a complement to LoRA workflows for stabilization during difficult prompts (rapid expression changes, hard motion), or for faces the model partially recognizes.

### 🎯 LTX Likeness Guide

**Category:** `10S Nodes/Identity`

Encodes a reference face into the conditioning pipeline. Auto-detects the face bbox via MediaPipe (with OpenCV Haar fallback), produces `reference_info` metadata that downstream nodes (LikenessAnchor, LikenessSemanticClamp) read from.

**Key parameters:**
- `image` (IMAGE) — reference face image
- `vae` (VAE) — LTX VAE for encoding
- `positive`, `negative` (CONDITIONING) — passed through with attention metadata
- `emit_latent` — `passthrough` (default, recommended) or `extend_latent` (legacy)
- `face_detect` — `auto` (MediaPipe→Haar fallback), `manual`, or `disabled`
- `manual_face_bbox` — `"x1,y1,x2,y2"` normalized 0-1 when `face_detect=manual`
- `reference_mask_mode` — `bbox_softfade` (default), `bbox_hard`, or `uniform`

The `emit_latent=passthrough` default is critical: extending the latent triggers learned end-keyframe behavior in the model regardless of conditioning placement. Pass the guide's effects through `reference_info` instead.

---

### 🪪 LTX Likeness Anchor

**Category:** `10S Nodes/Identity`

Per-block attn1 hook that pulls face-bbox video tokens toward reference identity features. Reads bbox from `reference_info` (LikenessGuide) or directly from `latent_frame_0`.

**Key parameters:**
- `model` (MODEL) — chain after LikenessGuide
- `strength` — pull magnitude (typical 0.10-0.30)
- `pull_mode` — `directional` (default, magnitude-preserving) or `additive` (legacy)
- `reference_source` — `auto`, `guide`, or `latent_frame_0`
- `sim_threshold` — cosine threshold for token matching (default 0.50)
- `late_block_falloff` — strength reduction on last 12 blocks (default 0.0; raise to 0.3-0.4 if face appears rigid)
- `depth_curve` — `flat` (default), `middle`, `late_focus`, `ramp_up`, `ramp_down`
- `bypass` (BOOLEAN) — when True, properly removes prior hooks (no leak across runs)

**Recommended config:**
```
strength            = 0.10-0.18
pull_mode           = directional
reference_source    = auto
late_block_falloff  = 0.4
depth_curve         = flat
```

**Important:** if chaining with Latent Anchor Aware (which also hooks attn1), the strengths *compound additively*. Each node's pull adds to the same attn1 residual. If using both, reduce individual strengths (e.g., AwareAnchor 0.08, LikenessAnchor 0.10 = combined effective ~0.18). At very high combined values, token-distribution variance can narrow visibly (color desaturation) — back off strength.

---

### 🪪 LTX Likeness Crop

**Category:** `10S Nodes/Identity`

Standalone face bbox cropper. Outputs cropped IMAGE of the detected face region plus the normalized bbox as STRING. Useful for previewing detection results or feeding face-only crops to other workflows.

---

### 🔎 LTX Latent Face Detector

**Category:** `10S Nodes/Diagnostic`

Standalone face detection node. Returns the normalized bbox as STRING (`"x1,y1,x2,y2"` format) for manual wiring into LikenessGuide's `manual_face_bbox` or LikenessAnchor's `override_face_bbox`. MediaPipe with OpenCV Haar fallback.

---

### 🧠 LTX Likeness Semantic Clamp ⚠ *experimental*

**Category:** `10S Nodes/Identity`

Semantic-aware text-token suppression for face-modifier vocabulary. Identifies which positive-prompt tokens are face-modifier-like (smiling, frowning, expressions) via embedding-space correspondence to a vocabulary, then selectively suppresses those tokens' attention contribution to face-bbox video tokens.

**Status:** Experimental. The mechanism is sound but the LTX2 text encoder's contextual blending produces flatter token similarities than typical CLIP-style encoders, making correspondence search noisier than ideal. Works best with `auto_threshold=p95` adaptive thresholding. Effects are subtle on most prompts; may be more pronounced on prompts with explicit expression directives.

**Key parameters:**
- `clip` (CLIP) — same encoder that produced your positive conditioning
- `positive` (CONDITIONING) — analyzed for face-modifier tokens
- `reference_info` (REFERENCE_INFO) — wire from LikenessGuide for bbox
- `suppression_strength` — 0.3-0.8 (default 0.5)
- `face_modifier_text` — comma-separated modifier vocabulary (default works for common prompts)
- `auto_threshold` — `p95` (default, recommended), `p98`, `p99`, or `disabled`
- `suppression_floor` — hard cutoff below which weights become 0 (default 0.3)
- `top_k` — confirming matches required (default 3)

Diagnostic output shows the per-token suppression distribution; a healthy run shows ~3-8% of tokens getting strong suppression. If diagnostic shows >40% suppressed, the threshold is too low for your encoder; if <1%, too high.

---

### 💨 LTX Action Amplifier ⚠ *experimental*

**Category:** `10S Nodes/Conditioning`

Selectively amplifies action / motion verb tokens in the positive prompt to make i2v output more responsive to verb-driven motion. Symmetric inverse of LikenessSemanticClamp — same correspondence-search backbone, but boosts matched tokens rather than suppressing them.

**Status:** Experimental. Replaces the deprecated blanket Text Amplifier approach with token-selective scaling. Capped boost ceiling (default `scale_ceiling=0.30`, so max +30% K/V scaling per matched token) keeps modifications controlled. Effect is subtle by design.

**Key parameters:**
- `clip` (CLIP), `positive` (CONDITIONING) — same as Semantic Clamp
- `amplification_strength` — 0.0-1.0 (default 0.3)
- `scale_ceiling` — maximum K/V scale factor delta (default 0.30 = max +30%)
- `auto_threshold` — `p95` default
- `action_vocabulary_text` — comma-separated verb vocabulary

Unlike Semantic Clamp, this applies uniformly across all video tokens (no bbox) — actions affect the whole frame.

---

## Supporting Nodes

### 🎯 LTX Latent Anchor

**Category:** `10S Nodes/Identity`

Content-blind variant of Latent Anchor Aware — same caching mechanism without the reference-image energy weighting. Use when you want the anchoring effect without supplying an external reference. The Aware variant supersedes this for most use cases.

---

### 🔍 LTX Latent Upsampler (Tiled)

**Category:** `10S Nodes/Latent`

Drop-in replacement for ComfyUI's stock `LTXVLatentUpsampler`. Adds spatial tiling with cosine-windowed overlap blending to address upscale-model failure modes at extreme aspect ratios. Auto-detects upscale ratio (works with x1.5, x2, etc.).

**Key parameters:**
- `tile_size` — spatial tile size in latent tokens (default 24, ~768 pixels)
- `overlap` — overlap between adjacent tiles (default 8)
- `max_size_for_no_tile` — skip tiling if both spatial dims ≤ this (default 32)
- `rotate_for_landscape` — transpose H/W before upscaling, then rotate back (experimental)

For most cases the defaults work. Lower `max_size_for_no_tile` to force tiling on smaller inputs for testing.

---

### 🔊 LTX Text Attention Amplifier ⚠ *deprecated*

**Category:** `10S Nodes/Identity`

Original blanket text cross-attention amplifier. Multiplies attn2 output uniformly. **Deprecated** in favor of the token-selective Action Amplifier — blanket amplification was found to produce noise at meaningful strength values. Kept for backward compatibility with existing workflows.

---

### 🔍 LTX Model Inspector

**Category:** `10S Nodes/Diagnostic`

Diagnostic node for inspecting LTX2 model structure — useful when developing new hook-based nodes. Lists transformer modules, prints parameter counts, traces tensor shapes through the forward chain.

---

## Recommended Workflow Patterns

### Two-pass I2V with full upscale-pass quality recovery

```
First-pass model
        ↓
   KSampler (first pass at native resolution)
        ↓
  Upsample (LTX Latent Upsampler Tiled)
        ↓
  Conditioning re-application
        ↓
  LTX Tiled Sampler (light refinement, audio_pass=tile_carrying)
        ↓
  VAE Decode
```

This is the workflow that motivated most of this package's development. The Tiled Sampler at the second-pass position is what converts a previously broken upscale pass into clean, lipsync-preserved output.

### Single-pass I2V with identity preservation

```
LoadModel → LTX Latent Anchor Aware → KSampler → VAE Decode
                  ↓
            sigmas, reference_image, vae
```

For face-targeted preservation, chain the Likeness suite:

```
LikenessGuide(image) ──→ pos', neg', reference_info
                              ↓
LikenessAnchor(model, reference_info, strength=0.10-0.18, pull_mode=directional)
                              ↓
KSampler → VAE Decode
```

### Maximum identity preservation: LoRA + inference-time stabilization

For unique faces (subjects unknown to the base model), the right approach is to train a subject-specific LoRA via Lightricks' [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video) (typically 30 images, ~15 minutes). Then use this package's nodes as a stabilization layer on top:

```
Base model + Subject LoRA
        ↓
LikenessGuide → LikenessAnchor → KSampler → VAE Decode
```

LoRA provides the identity; LikenessAnchor stabilizes against drift during difficult prompts. This combination produces better results than either approach alone.

### Combined identity + scene stabilization (chained)

```
Model → LTX Latent Anchor Aware → LTX Likeness Anchor → KSampler
```

Both nodes hook `attn1` with different sentinel attributes — they coexist. Important: their pull strengths *compound additively*, so reduce individual strengths when chained (try AwareAnchor 0.08 + LikenessAnchor 0.10).

---

## Architecture Notes

All identity nodes operate via PyTorch `register_forward_hook` on `transformer_blocks[i].attn1` (or `attn2` for the experimental token-selective nodes). Hooks return modified output tensors that flow forward to the next block's input.

**Why hook intervention works:** LTX2's DiT is content-blind to its own intermediate states. Modifying attention output adds a residual that the rest of the block's computation (cross-attention, FFN) integrates naturally. Strength values that look small numerically (0.10-0.20) compound across 48 blocks per step into substantive effects.

**The inference-time ceiling:** activation-level interventions can stabilize identity that the model already knows, but cannot impart new identity knowledge. For unique faces, a short LoRA training run (~15 minutes) imparts the identity at the weight level and produces dramatically better preservation. This package complements LoRA workflows rather than replacing them.

**Per-frame centered cosine similarity** is used throughout for matching. Raw cosine similarity is dominated by common-mode features (positional encoding, scaffold features) that all tokens share. Subtracting the per-frame mean leaves identity-specific deviations that discriminate cleanly.

**Cache-and-broadcast** (used by Latent Anchor Aware) snapshots the model's representation at peak conditioning alignment (mid-sampling) and uses it as a stable pull target. Conceptually adjacent to self-distillation and self-conditioning in diffusion, but operationalized as inference-time intervention rather than training-time signal.

**Tiled Sampler architecture** does not use MultiDiffusion-style per-step coordination. Instead, each tile runs the full sampling pipeline as a standalone clip at training-distribution token count, then results blend with cosine windows. This works for light refinement passes (low denoise, few steps) where per-tile divergence is bounded. The carrier tile uses a single wrapper sampling pass for video and audio together — the model's video-audio cross-attention runs naturally during that pass, and we extract both modalities from the flattened combined output.

**Bypass-safe hook management:** all hook-based nodes store PyTorch handle references and actively remove prior hooks when bypassed or re-applied. This prevents hooks from prior runs leaking into subsequent runs when `model.clone()` shares the underlying transformer blocks by reference.

---

## Compatibility & Limitations

- **LTX2-specific.** Hooks rely on `LTXAVModel.transformer_blocks` and `BasicAVTransformerBlock` structure. Will not work on other DiT-based video models without adaptation.
- **Distilled CFG=1 setup tested most extensively.** Standard CFG works but compounds hook calls per step (cond + uncond passes), changing effective strength. Set `forwards_per_step=2` in advanced mode.
- **Tiled Sampler is for light refinement only.** Heavy-denoise-from-noise generation produces tile divergence that the cosine blend can't reconcile. Use the full sampler for first-pass; tiled for upscale refinement only.
- **Inference-time ceiling for unique faces.** Activation-level interventions stabilize known identity; they don't teach new identity. For unique faces unknown to the base model, train a LoRA. The Likeness suite is best as a stabilization complement to LoRA workflows.

---

## Version History

**v1.6.0** — Likeness suite + experimental conditioning nodes
- Added LTX Likeness Guide / Anchor / Crop (face-region identity preservation, complement to LoRA)
- Added LTX Likeness Semantic Clamp (experimental, token-selective text-side suppression)
- Added LTX Action Amplifier (experimental, token-selective verb amplification)
- Added LTX Latent Face Detector (MediaPipe + OpenCV bbox detection)
- Removed LTX Face Attention Anchor (replaced by the Likeness Anchor architecture)
- Marked LTX Text Attention Amplifier as deprecated (blanket-scaling approach superseded by Action Amplifier's token-selective approach)
- Bypass-safe hook management across all identity nodes (stored handles, active cleanup)

**v1.2.0** — Audio-aware tiled sampling release
- Added LTX Tiled Sampler v2.0 (whole-pipeline spatial tiling with carrier-tile audio capture for lipsync preservation)
- Removed `LTXLatentColorRestore` and `LTXLatentOutlierSuppress` — these were investigative attempts at the upscale-pass color drift problem; the Tiled Sampler addresses the root cause directly so they're no longer needed
- Added LTX Text Attention Amplifier v1.1 (alternative approach to upscale-pass dilution)
- Latent Upsampler Tiled v1.1 (auto-detect upscale ratio, fix window-fade math for partial overlaps)

**v1.0.0** — Initial release
- Face Attention Anchor (v4.0)
- Latent Anchor (v1.4)
- Latent Anchor Aware (v2.3)
- Model Inspector

---

## License

MIT

## Author

TenStrip · [github.com/TenStrip](https://github.com/TenStrip)
