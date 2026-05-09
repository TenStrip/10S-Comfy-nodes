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

No external dependencies beyond ComfyUI's existing environment (PyTorch, comfy package). All nodes are pure Python.

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

## Supporting Nodes

### 🪪 LTX Face Attention Anchor

**Category:** `10S Nodes/Identity`

Face-region identity preservation via per-block attention residual modification. Tracks a face bbox in the conditioning frame across all video frames using cosine-similarity correspondence in centered feature space. Pulls drifted face tokens back toward the anchor frame's identity features.

**Key parameters:**
- `face_bbox_norm` — normalized bbox `"x1,y1,x2,y2"` in [0,1] (default `"0.35,0.10,0.65,0.50"`)
- `strength` — pull magnitude (typical 0.10-0.20)
- `inject_mode` — `tracked` (general use) or `tracked_correction` (drift recovery for hard cuts)
- `anchor_upsample` — bilinear upsample of anchor pool (default 2; raise to 3-4 for small-bbox composition)
- `spatial_prior` — Gaussian falloff confining intervention to face vicinity (default 0.5)
- `depth_curve` — per-block strength scaling: `flat` (default), `late_focus`, `ramp_up`, etc.

**General identity preservation:**
```
strength            = 0.10
inject_mode         = tracked
depth_curve         = flat
spatial_prior       = 0.5
anchor_upsample     = 2
```

**For hard scene cuts:**
```
strength            = 0.15
inject_mode         = tracked_correction
depth_curve         = late_focus
```

---

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

### 🔊 LTX Text Attention Amplifier

**Category:** `10S Nodes/Identity`

Modulates text cross-attention (`attn2`) influence per block. An alternative approach to the upscale-pass dilution problem (see Tiled Sampler), useful when tiling isn't appropriate. Bidirectional: < 1.0 suppresses text influence, > 1.0 amplifies.

**Key parameters:**
- `text_amplification` — multiplier (1.0 = no change; 1.2-1.5 typical for upscale dilution recovery; < 1.0 for suppression)
- `spatial_focus` — 0.0 = uniform, > 0.0 = Gaussian-weighted with full amplification at center
- `block_index_filter` — limit which blocks get the amplification

**For upscale-pass dilution recovery (if not using Tiled Sampler):**
```
text_amplification = 1.30
spatial_focus      = 0.0
```

**For reducing prompt over-fitting:**
```
text_amplification = 0.70
```

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

For face-targeted preservation, replace or chain with Face Attention Anchor: `Latent Anchor Aware → Face Anchor → KSampler`.

### Combined identity + scene stabilization (chained)

```
Model → LTX Latent Anchor Aware → LTX Face Attention Anchor → KSampler
```

All hook-based identity nodes coexist on the same model — different sentinel attributes prevent interference. Each can be bypassed independently.

---

## Architecture Notes

All identity nodes operate via PyTorch `register_forward_hook` on `transformer_blocks[i].attn1` (or `attn2` for Text Amplifier). Hooks return modified output tensors that flow forward to the next block's input.

**Why hook intervention works:** LTX2's DiT is content-blind to its own intermediate states. Modifying attention output adds a residual that the rest of the block's computation (cross-attention, FFN) integrates naturally. Strength values that look small numerically (0.10-0.20) compound across 48 blocks per step into substantive effects.

**Per-frame centered cosine similarity** is used throughout for matching. Raw cosine similarity is dominated by common-mode features (positional encoding, scaffold features) that all tokens share. Subtracting the per-frame mean leaves identity-specific deviations that discriminate cleanly.

**Cache-and-broadcast** (used by Latent Anchor Aware) snapshots the model's representation at peak conditioning alignment (mid-sampling) and uses it as a stable pull target. Conceptually adjacent to self-distillation and self-conditioning in diffusion, but operationalized as inference-time intervention rather than training-time signal.

**Tiled Sampler architecture** does not use MultiDiffusion-style per-step coordination. Instead, each tile runs the full sampling pipeline as a standalone clip at training-distribution token count, then results blend with cosine windows. This works for light refinement passes (low denoise, few steps) where per-tile divergence is bounded. The carrier tile uses a single wrapper sampling pass for video and audio together — the model's video-audio cross-attention runs naturally during that pass, and we extract both modalities from the flattened combined output.

---

## Compatibility & Limitations

- **LTX2-specific.** Hooks rely on `LTXAVModel.transformer_blocks` and `BasicAVTransformerBlock` structure. Will not work on other DiT-based video models without adaptation.
- **Distilled CFG=1 setup tested most extensively.** Standard CFG works but compounds hook calls per step (cond + uncond passes), changing effective strength. Set `forwards_per_step=2` in advanced mode.
- **Tiled Sampler is for light refinement only.** Heavy-denoise-from-noise generation produces tile divergence that the cosine blend can't reconcile. Use the full sampler for first-pass; tiled for upscale refinement only.

---

## Version History

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
