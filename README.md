# 10S Nodes for ComfyUI

Custom nodes for the LTX2 video diffusion model by Lightricks. Each node targets a specific quality or behavior issue in the base model. Identity preservation, latent stabilization, sequence-level reference injection, upscale-pass quality, and diagnostic tooling.

Nodes operate via PyTorch forward hooks and instance-level method patches on the DiT backbone. No model retraining required.

> **Compatibility note:** these nodes target LTX2 / LTX-AV (the dual-stream video+audio DiT). They rely on LTX2-specific class structure (`LTXAVModel`, `BasicAVTransformerBlock`, `SymmetricPatchifier`) and will not work on other diffusion models without adaptation.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TenStrip/10S-Comfy-nodes.git 10S_Nodes
```

Restart ComfyUI. Nodes appear under `10S Nodes/` in the node-add menu.

To update:
```bash
cd ComfyUI/custom_nodes/10S_Nodes
git pull
```

No external dependencies beyond ComfyUI's existing environment.

---

## Nodes

### LTX Face Identity Reinforcer

**Category:** `10S Nodes/Identity`

Single-node identity reinforcer that makes the [LTX-Best-Face-ID LoRA](https://huggingface.co/Alissonerdx/LTX-Best-Face-ID) by Alissonerdx work correctly with i2v conditioning. Originally the LoRA was T2V-only because its reference-latent injection at frame 0's RoPE grid competed with i2v's own frame 0 latent conditioning. This node implements the full mechanism from the creator's spec: overlap-layout reference token injection with a per-rotary-dimension phase rotation composed on top of RoPE, providing positional disambiguation between reference and target tokens without spatial collision.

Composes six mechanisms internally so the user sees one node:
- VAE encoding of the reference image
- YuNet face detection with auto-crop and padding
- Reference-to-reference alignment (secondary reference's face is scaled/positioned to match the primary's face bbox, eliminating aspect/proportion desync)
- Reference token injection at overlap coordinates (same T/H/W as target)
- Per-dimension RoPE phase rotation for reference positions (`source_id=2, phase_scale=1.0` matching Best-Face-ID's trained convention)
- Spatial mask gating with cosine falloff around the detected face region

**Wiring:**

```
Load Model → Load Best-Face-ID LoRA → LTX Face Identity Reinforcer → Sampler
                                              ↑
                                    VAE, reference_image, target_latent
                                    (reference_image_2 optional)
```

**Use case:** identity-consistent i2v generation. The reference face influences generation across all frames while i2v conditioning controls composition. Works with a single face reference (identity anchor) or with a second reference for additional identity signal (different angle, expression, or lighting of the same subject).

**Optional full-image mode:** with `auto_face_crop=False`, the entire reference image gets encoded rather than just a face-centered crop. Repurposes the tool as a general i2v stabilizer — reference tokens carry the full scene, providing lighting/composition anchoring throughout the temporal sequence instead of only at frame 0.

Credit for the underlying Best-Face-ID technique and LoRA: [Alissonerdx](https://huggingface.co/Alissonerdx/LTX-Best-Face-ID).

### LTX Reference Enable / Conditioning / Bypass / Probe

**Category:** `10S Nodes/LTX2` (Probe in `10S Nodes/Debug`)

Reference token prefix injection at the sequence level. Encodes an input image to a latent and prepends its tokens to the video token sequence inside the transformer. The model attends to these prepended tokens via standard self-attention, producing subtle identity influence across all generated frames.

**Use case:** identity preservation that composes with standard i2v latent conditioning. The reference token mechanism is a different intervention point than the latent anchor family — anchors modulate attention output mid-forward; reference tokens add new tokens at sequence input. Both can be applied together.

**Wiring:**

```
Load Model → LTX Reference Enable → MODEL'
                                       │
Load VAE   → VAE   ──┐                 │
Load Image → IMAGE ──┤                 │
EmptyLatent → LATENT ┴→ LTX Reference Conditioning (target_latent input) → MODEL''
                                                                              │
                                                                              ▼
                                                                          KSampler
```

- **LTX Reference Enable** — applies class-level patches to `LTXAVModel`. Safe passthrough when no reference is attached.
- **LTX Reference Conditioning** — encodes IMAGE through VAE, normalizes via `process_latent_in`, attaches to MODEL. Wire the same LATENT going to your sampler into the optional `target_latent` input — the image will be pixel-space resized to match before encoding. `position_mode` defaults to `reference` (memory positions overlap target frame 0, uniform influence across frames). `strength` defaults to 1.0; reduce to 0.6-0.8 if first-frame distortion is visible.
- **LTX Reference Bypass** — clears reference state from a MODEL. Use between sampler passes when memory injection isn't wanted for a particular pass (typical: upscale/refinement pass after the primary generation pass).
- **LTX Reference Probe** — diagnostic node, reports patch status and reference state. Use before a sampler to verify wiring without running a generation.

Works on any LTX2.3 checkpoint (vanilla LTX2, Echo, merged variants). Handles tiled samplers and CFG batching transparently — the patches auto-resize and broadcast as needed.

### LTX Tiled Sampler

**Category:** `10S Nodes/Sampling`

Drop-in replacement for `SamplerCustomAdvanced`. Splits the latent spatially along its longer axis with overlap, samples each tile through the full pipeline at training-distribution token count, blends with cosine-windowed Hann overlap.

**Use case:** upscale-pass quality. When a second sampling pass runs on a 2× upscaled latent (4× more spatial tokens), each video token receives ~1/4 the per-token text-conditioning influence the model was trained with, and the model operates outside its trained spatial-token-count range. This causes broad hue shifts, color drift, and prompt adherence loss. Tiling restores each tile to the trained distribution.

**Key parameters:**
- `tile_axis` — `auto` (splits longer dimension), `H`, or `W`
- `n_tiles` — number of tiles along the chosen axis (default 2)
- `tile_overlap` — overlap in latent tokens (default 8)
- `max_size_for_no_tile` — auto-skip tiling if both dims are at or below this (default 24)
- `audio_pass` — `passthrough` (audio unchanged) or `tile_carrying` (audio sampled through a chosen carrier tile)
- `audio_carrier_tile` — `first`, `middle`, or `last`
- `bypass_tiling` — `True` for transparent passthrough to single-pass sampling

Use for refinement passes (low denoise, few steps). Not intended for heavy-denoise first-pass generation.

### LTX Latent Anchor Aware

**Category:** `10S Nodes/Identity`

Inference-time regularizer. Snapshots the model's representation at a mid-sampling step (when conditioning alignment peaks) and pulls subsequent computation toward the cached state. Optionally weighted by spatial energy from an external reference image.

**Use case:** prompt adherence, scene composition stability, physical sensibility. Reduces late-step drift away from prompt-aligned content.

**Hook target:** `transformer_blocks[i].attn1` output (post-self-attention residual modification).

### LTX Latent Anchor

**Category:** `10S Nodes/Identity`

Basic variant of Latent Anchor Aware without external reference image weighting. Lighter, fewer parameters.

### LTX Face Attention Anchor

**Category:** `10S Nodes/Identity`

Face-targeted identity preservation. Uses cosine similarity matching with per-frame centered features to detect and stabilize face-region tokens across generation steps.

**Hook target:** `transformer_blocks[i].attn1` output, with face-region masking.

### LTX Text Attention Amplifier

**Category:** `10S Nodes/Identity`

Boosts text cross-attention strength. Alternative approach to upscale-pass conditioning dilution (the Tiled Sampler addresses the root cause; this is an additive boost).

**Hook target:** `transformer_blocks[i].attn2` output (text cross-attention).

### LTX Latent Upsampler (Tiled)

**Category:** `10S Nodes/Upscale`

Drop-in for `LTXVLatentUpsampler` with spatial tiling for memory-safe upscaling of large or extreme-aspect inputs. Auto-detects upscale ratio (×1.5, ×2).

**Note:** investigation showed that the upscaler itself wasn't the source of the upscale-pass color shift — the sampler operating on upscaled token counts was. This node remains useful as a memory-safer upscaler for very large inputs.

### LTX Model Inspector

**Category:** `10S Nodes/Debug`

Inspects LTX2 module structure (block count, submodules, attention layer shapes). Used during node development and debugging.

### JoyEcho Image-to-Memory

**Category:** `10S Nodes/JoyAI Echo`

Builds a memory bank entry for the `ComfyUI_JoyAI_Echo` package's native pipeline. Only useful if that package is installed and you're using its `BidirectionalMemoryAVInferencePipeline` rather than the standard Comfy sampling path. Independent of the LTX Reference nodes above.

---

## Workflow Patterns

### Best-Face-ID i2v identity (recommended for face reference workflows)

```
Load Model → Load Best-Face-ID LoRA → LTX Face Identity Reinforcer → KSampler
                                              ↑
                                    VAE, reference_image, target_latent
```

Replaces the LikenessAnchor pattern for face-identity workflows. The reinforcer node handles VAE encoding, face detection, reference alignment, phase rotation, and spatial gating internally. Single-node drop-in.

For full-scene i2v stabilization (non-face use case), set `auto_face_crop=False` and `spatial_gating=off` — reference then acts as a general scene anchor across the temporal sequence.

### Identity-preserved generation with reference injection

```
Load Model → LTX Reference Enable ──→ LTX Reference Conditioning ──→ KSampler
                                              ↑
                                    image, vae, target_latent
```

Optionally combine with standard i2v frame_0 latent conditioning (your existing workflow). At i2v strength 0.4 + reference strength 1.0, identity preservation is strong without compositional divergence.

### Two-pass with reference + tiled upscale

```
Pass 1:  Model → Reference Enable → Reference Conditioning → KSampler (primary pass)
                                                                  │
Pass 2:  Model → Reference Bypass ───→ LTX Tiled Sampler (upscale, no reference)
```

Branch the model after Enable. The primary pass gets the reference; the upscale pass goes through Bypass to clear reference state and runs clean. Avoids first-frame distortion that can appear when reference tokens are applied during refinement passes.

### Identity preservation via latent anchors

```
Model → LTX Latent Anchor Aware → LTX Face Attention Anchor → KSampler
```

All hook-based identity nodes coexist on the same model. Different sentinel attributes prevent interference. Each can be bypassed independently.

### Reference + anchors combined

```
Model → Reference Enable → Reference Conditioning → Latent Anchor Aware → KSampler
```

Reference tokens (sequence-input intervention) and latent anchors (mid-forward attention modification) are at different intervention points and compose without conflict.

---

## Architecture Notes

**Hook-based identity nodes** operate via PyTorch `register_forward_hook` on `transformer_blocks[i].attn1` (or `attn2` for Text Amplifier). Hooks return modified output tensors that flow forward to the next block. LTX2's DiT is content-blind to its own intermediate states, so modifying attention output adds a residual that subsequent block computation integrates naturally. Strength values that look small numerically (0.10-0.20) compound across 48 blocks per step.

**Reference token injection** operates via instance-level patches on `LTXAVModel._process_input`, `_prepare_timestep`, and `patchifier.unpatchify`. Reference tokens are prepended to the video token sequence using the same patchifier and 3D RoPE positions as target tokens — no special handling, just additional sequence positions the self-attention naturally attends to. Per-frame compressed modulation tensors (`CompressedTimestep`) are extended at the `num_frames` dimension to match the prepended token count.

**Tiled Sampler** does not use MultiDiffusion-style per-step coordination. Each tile runs the full sampling pipeline as a standalone clip at training-distribution token count, then results blend with cosine windows. Works for light refinement passes (low denoise, few steps). The carrier tile uses a single wrapper sampling pass for video and audio together — the model's video-audio cross-attention runs naturally during that pass.

**Per-frame centered cosine similarity** is used by anchor nodes for token matching. Raw cosine similarity is dominated by common-mode features (positional encoding, scaffold features) shared by all tokens. Subtracting the per-frame mean leaves identity-specific deviations that discriminate cleanly.

---

## Compatibility & Limitations

- **LTX2-specific.** Patches and hooks rely on LTX2 class structure. Will not work on other DiT-based video models without adaptation.
- **Distilled CFG=1 setup tested most extensively.** Standard CFG works but compounds hook calls per step. Set `forwards_per_step=2` on anchor nodes in advanced mode.
- **Tiled Sampler is for light refinement only.** Heavy-denoise-from-noise generation produces tile divergence that cosine blending can't reconcile. Use full sampler for first pass; tiled for upscale refinement.
- **Reference Conditioning requires matching spatial dims.** Wire `target_latent` to the Conditioning node, or ensure your image and target latent have matching `(H/32, W/32)` latent dimensions.

---

## Version History

**v1.9.6** — ComfyUI 0.28.0 compatibility fix
- Fixed `AttributeError: 'ModelPatcherDynamic' object has no attribute 'parameters'` in LTX Latent Upsampler (Tiled)
- ComfyUI 0.28.0 changed `LatentUpscaleModelLoader` to return a ModelPatcher rather than a bare `nn.Module`
- Latent Upsampler Tiled v1.2 — model handle resolved through `_resolve_upscale_model()`; residency delegated to `model_management.load_models_gpu`
- Backward compatible with pre-0.28 ComfyUI (legacy bare-module path retained)

**v1.9.4** — LTX Face Identity Reinforcer (Best-Face-ID i2v integration)
- Added LTXFaceIdentityReinforcer — single-node identity reinforcer for the LTX-Best-Face-ID LoRA
- Full mechanism implementation per creator's spec: overlap coordinate placement + per-dimension RoPE phase rotation
- Makes Best-Face-ID work with i2v (previously T2V-only)
- Composes VAE encoding, YuNet face detection, reference alignment, phase rotation, and spatial gating into one node
- Extended `ltx_reference_enable.py` with `_prepare_positional_embeddings` wrapper implementing source phase composition

**v1.9.0** — LTX Reference Token Injection
- Added LTX Reference Enable, Conditioning, Bypass, Probe
- Sequence-level reference token prefix injection for identity preservation
- Works on any LTX2.3 checkpoint, composes with i2v and anchor nodes

**v1.2.0** — Audio-aware tiled sampling release
- Added LTX Tiled Sampler v2.0 (whole-pipeline spatial tiling with carrier-tile audio capture for lipsync preservation)
- Added LTX Text Attention Amplifier v1.1
- Latent Upsampler Tiled v1.1 (auto-detect upscale ratio)

**v1.0.0** — Initial release
- Face Attention Anchor, Latent Anchor, Latent Anchor Aware, Model Inspector

---

## License

MIT

## Author

TenStrip · [github.com/TenStrip](https://github.com/TenStrip)
