# CLAUDE.md — 10S Nodes Reference

> AI-targeted reference document. Dense, terminology-heavy, optimized for parsing and decision-tree reasoning rather than human prose flow. Use this when answering questions about the 10S Nodes package and its underlying LTX2 architecture.

---

## PACKAGE CONTEXT

- **Target model:** LTX2 / LTX-AV (Lightricks dual-stream video+audio DiT)
- **Backbone class:** `LTXAVModel` with `.transformer_blocks` ModuleList
- **Block class:** `BasicAVTransformerBlock` × 48
- **Per-block submodules:** `attn1` (video self-attn 67M), `audio_attn1`, `attn2` (text cross-attn), `audio_attn2`, `audio_to_video_attn`, `video_to_audio_attn`, `ff` (134M), `audio_ff`
- **Latent format:** 5D `(B, C=128, F, H, W)` for video; audio is separate `(B, 8, audio_F, 16)` or similar
- **VAE compression:** 32× spatial / 8× temporal
- **No DiT patchification:** `SPATIAL_PATCH=1`, `TEMPORAL_PATCH=1`
- **Positional encoding:** 3D RoPE
- **Token order in flattened forward:** F outermost (fhw)
- **NestedTensor wrapper:** LTX-Video uses a custom class to bundle video+audio. The wrapper is NOT a `torch.Tensor` subclass in this version. Its `.tensors` attribute is a list `[video_tensor, audio_tensor]`.

---

## NODE CATALOG (priority order)

### 🎲 LTX Tiled Sampler — the upscale-pass fix
- **File:** `latent_tiled_sampler.py`
- **Class:** `LTXTiledSampler`
- **Category:** `10S Nodes/Sampling`
- **Role:** drop-in replacement for `SamplerCustomAdvanced`. Solves broad hue shift / conditioning drift in upscale-pass sampling.
- **Mechanism:** spatial tiling with cosine-Hann overlap blending. Each tile sampled at training-distribution token count. Carrier tile optionally captures audio.
- **Default config:** `tile_axis=auto, n_tiles=2, tile_overlap=8, max_size_for_no_tile=24, audio_pass=passthrough`
- **Recommended for upscale pass:** add `audio_pass=tile_carrying, audio_carrier_tile=first`
- **Bypass option:** `bypass_tiling=True` → transparent passthrough to single-pass sampling

### 🎯 LTX Latent Anchor Aware
- **File:** `latent_anchor_aware.py`
- **Class:** `LTXLatentAnchorAware`
- **Category:** `10S Nodes/Identity`
- **Role:** inference-time regularizer for prompt adherence, scene composition, physical sensibility
- **Mechanism:** snapshots model's representation at a sampling step, pulls subsequent computation toward cached state. Optional spatial energy weighting from external reference image.
- **Hook target:** `block.attn1` output (post-self-attn residual modification)
- **Default config:** `strength=0.10, cache_at_step=6, similarity_threshold=0.50, energy_threshold=0.30, decay_with_distance=0.0`
- **Simple/advanced mode:** simple exposes 5-7 knobs; advanced reveals `cache_mode`, `forwards_per_step`, `cache_warmup`, `anchor_frame`, `depth_curve`, `block_index_filter`

### 🎯 LTX Latent Anchor (basic variant)
- **File:** `latent_anchor.py`
- **Class:** `LTXLatentAnchor`
- **Same as Aware but without reference image / energy weighting.** Use the Aware variant for most cases.

### 🪪 LTX Face Attention Anchor
- **File:** `face_anchor.py`
- **Class:** `LTXFaceAttentionAnchor`
- **Category:** `10S Nodes/Identity`
- **Role:** face-region identity preservation
- **Mechanism:** per-block attn1 hooks, tracks face bbox via centered cosine similarity, pulls drifted face tokens to anchor frame
- **Default config:** `face_bbox_norm="0.35,0.10,0.65,0.50", strength=0.10, inject_mode=tracked, spatial_prior=0.5, anchor_upsample=2`
- **For hard cuts:** `inject_mode=tracked_correction, depth_curve=late_focus, strength=0.15`

### 🔊 LTX Text Attention Amplifier
- **File:** `latent_text_amplifier.py`
- **Class:** `LTXTextAttentionAmplifier`
- **Role:** alternative to Tiled Sampler for conditioning dilution. Multiplies text cross-attention (attn2) output per block. Bidirectional: <1.0 suppresses, >1.0 amplifies.
- **Default:** `text_amplification=1.30`
- **Use case:** when tiling not appropriate (e.g., heavy denoise or unusual aspect)

### 🔍 LTX Latent Upsampler (Tiled)
- **File:** `latent_upsampler_tiled.py`
- **Drop-in for `LTXVLatentUpsampler`** with spatial tiling for extreme aspect ratios. Auto-detects upscale ratio (x1.5, x2). NOTE: empirical finding showed that the upscaler itself wasn't the source of color shift (the sampler operating on upscaled token counts was). This node remains useful as a memory-safer upscaler for very large inputs.

### 🔍 LTX Model Inspector
- **Diagnostic node** for inspecting LTX2 module structure. Used during node development.

### 🔗 LTX Reference Enable / 🎴 LTX Reference Conditioning / 🚫 LTX Reference Bypass / 🔎 LTX Reference Probe
- **Files:** `ltx_reference_enable.py`, `ltx_reference_conditioning.py`
- **Category:** `10S Nodes/LTX2` (Bypass also here; Probe in `Debug`)
- **Role:** identity-preservation via reference token prefix injection at the sequence level. Complements (does NOT replace) standard i2v frame_0 latent conditioning and the latent_anchor / likeness_anchor families. Different intervention point: anchors modulate attention output mid-forward; reference tokens add new tokens at sequence input that target tokens attend to via self-attention.
- **Mechanism (Enable):** class-level monkey-patch on `LTXAVModel._process_input` and `_prepare_timestep`, plus instance-level wrap on `patchifier.unpatchify`. When a reference latent is present:
    1. `_process_input`: patchify reference using same `SymmetricPatchifier(patch_size=1)` as target → same token layout & 3D RoPE positions. Apply `patchify_proj`. Concatenate to front of video token sequence. Broadcast batch dim to match vx (handles CFG batching + tile batching).
    2. `_prepare_timestep`: extend per-frame compressed modulation tensors (CompressedTimestep objects with `num_frames * patches_per_frame == target_seq_len`) by prepending replicated row 0 entries — `num_frames += ref_frames`, `data` gains `ref_frames` rows. Subsequent `expand_for_computation()` produces a tensor sized `(num_frames + ref_frames) * patches_per_frame` matching the prefixed vx.
    3. Unpatchify wrap: strip first `ref_seq_len` tokens from latents before reshape.
- **Mechanism (Conditioning):** VAE-encode IMAGE → 5D `(B, C, F=1, H, W)` latent → `process_latent_in` to normalize to model's expected distribution (critical: without this, raw VAE output causes red-tint artifacts) → store in `model_options["transformer_options"]["reference_latent"]` AND as `_ltx_reference_latent` attribute on diffusion_model. Optional `target_latent` input enables pixel-space image resize to match target's spatial dims before VAE encoding — guarantees `patches_per_frame` alignment with target. Optional `position_mode`: `reference` (memory positions overlap target frame 0) or `prefix_continuous` (memory positions precede target temporally).
- **Mechanism (Bypass):** clones MODEL and clears reference state from BOTH model_options and the diffusion_model attribute side-channel. Use between sampling passes when memory injection isn't wanted for a specific pass (typical: upscale/refinement pass).
- **Position modes:**
    - `reference` (default): memory positions = target frame 0 positions. RoPE attention deltas are zero between memory and target frame 0 → strong identity influence concentrated at frame 0, propagating to later frames via attention. Subtle first-frame distortion possible; mitigate by reducing strength to 0.6-0.8.
    - `prefix_continuous`: memory positions precede target temporally. Equivalent to standard i2v prior-context conditioning.
- **Spatial alignment requirement:** memory's H, W must match target's H, W (so `patches_per_frame_mem == patches_per_frame_tgt`). Conditioning resizes the image in pixel space when `target_latent` is wired; Enable also provides latent-space bilinear fallback at sampling time for cases where they don't (e.g., tiled sampling where each tile has different spatial than the original target).
- **Batch alignment:** reference is encoded once at batch=1; Enable broadcasts to match vx's batch dimension before concat (handles CFG batching, tile batching).
- **Compatibility:** works on any LTX2.3 checkpoint (vanilla LTX2, Echo, merged variants). Originally discovered while implementing JoyAI-Echo's memory bank pattern; the underlying mechanism is LTX2-architecture-general, not Echo-specific.
- **Composability:** combines well with standard i2v latent conditioning. Empirically, i2v at strength 0.4 + memory at strength 1.0 produces strong identity preservation without divergence — i2v provides the spatial anchor, memory provides identity reinforcement.
- **`zero_ref_timesteps` toggle (Enable, default OFF):** JoyAI's native pipeline zeros memory timesteps to mark them as clean reference (σ=0). Empirically with the released Echo T2V checkpoint, zeroing causes severe distortion (color inversion). OFF treats memory tokens as carrying target's noise sigma, which works better. Future i2v-trained checkpoints may benefit from enabling.

### 🧠 JoyEcho Image-to-Memory
- **File:** `joyecho_image_to_memory.py`
- **Role:** builds a fake memory bank entry for the `ComfyUI_JoyAI_Echo` package's native pipeline (separate from LTX Reference Enable above, which works through stock Comfy's LTXAVModel). Only useful if `ComfyUI_JoyAI_Echo` is installed and you're using its `BidirectionalMemoryAVInferencePipeline` rather than the standard Comfy sampling path.

---

## CRITICAL EMPIRICAL FINDINGS

### Finding 1: Upscale-pass hue shift is induced by the sampler, NOT the upscaler

**Test chain that confirmed this:**
- Pre-upscale latent → VAE decode = clean
- Post-upscale latent → VAE decode = brief pink artifact in early frames center, rest normal
- Post-upscale → conditioning re-applied → VAE decode = still normal
- Post-upscale → conditioning → sampler → VAE decode = broad pink hue shift across entire output

**Conclusion:** the sampler smears localized point artifacts into global tonal shift via attention. Color shift is a sampler-induced symptom of operating at out-of-training-distribution token counts.

### Finding 2: Token count dilution is the root cause

When second-pass sampler operates on a 2× upscaled latent:
- 4× more spatial tokens
- Each video token receives ~1/4 the per-token text-conditioning influence
- Positional encoding spans out-of-training-distribution range
- Model's attention statistics drift

**Fix architecture:** keep each sampling unit at training-distribution token count via spatial tiling. This is what Tiled Sampler does.

### Finding 3: NestedTensor wrapper output is structurally weird

Critical for any code that uses `guider.sample()` with a wrapper input:
- `tile_result.tensors[0]` returns the **cached INPUT video reference**, NOT the sampled output
- `tile_result.tensors[1]` returns the **sampled audio** (this slot does update correctly)
- The actual sampled video lives in `x0_output["x0"]` captured via the sampling callback, as a **flat combined tensor**
- Apply `model.process_latent_out(x0_output["x0"])` then unflatten manually

**Unflatten formula:**
```
total_elements = video_shape.numel() + audio_shape.numel()
First chunk of `total_elements` → reshape to video_shape
Remainder → reshape to audio_shape
```

**Implemented as `_unflatten_ltx_combined(combined, expected_video_shape, expected_audio_shape)`**. Used by Tiled Sampler v2.0+ for both carrier-tile audio capture and no-tile wrapper sampling.

### Finding 4: Per-frame centered cosine similarity is essential for block-level matching

Raw cosine similarity baseline ≈ 0.86 (dominated by positional encoding scaffold).
After subtracting per-frame mean from features: baseline drops to ≈ 0.24, identity-specific deviations become discriminative.

```python
frame_mean = grid.mean(dim=token_dim, keepdim=True)
centered = grid - frame_mean
norm = F.normalize(centered, dim=-1)
sim = bmm(norm, anchor_norm.transpose(...))
```

Used throughout: face_anchor, latent_anchor, latent_anchor_aware.

### Finding 5: Energy modulation needs all-or-nothing semantics

Linear interpolation `factor = (1-mod) + mod*energy_norm` (v2.1 of aware) caused anatomy distortion at intermediate values. Why: gradient-strength pulling across spatially-adjacent tokens that should belong to coherent objects creates within-object inconsistency. Hand example: high-energy fingers pulled at 100%, low-energy palm at 50% → contorted fingers.

**Fix (v2.2 of aware):** sigmoid threshold `factor = sigmoid((energy - threshold) * 16)` produces narrow transition zone. Tokens are clearly in or out of the pull set.

### Finding 6: Tile blending math requires actual-overlap calculation

Naive use of configured `overlap` parameter for cosine fade sizes produces `weight_acc > 1.0` because last-tile alignment forces some overlaps to exceed configured value. Each tile's actual overlap with neighbors must be computed from tile_start positions.

**Sanity check:** `weight_acc.min() ≈ weight_acc.max() ≈ 1.0` after all tiles processed. If `max > 1.05`, cosine fades aren't summing properly. If `min < 0.001`, some output positions have unstable normalization → increase tile_overlap.

### Finding 7: Mid-sampling cache lock-in is empirically optimal

For 13-step schedules, `cache_at_step=3` to `cache_at_step=9` produces strongest effects. Theoretical justification: this is the sigma range where model has integrated conditioning but not yet committed to fine details. Caching here preserves the conditioning-aligned scaffold.

### Finding 8: Mask resize required before tile slicing

ComfyUI workflows often have masks at first-pass resolution that ComfyUI auto-interpolates up at sample time. Tiling breaks this — slicing latent indices against a smaller mask produces `H=0` tile masks → `prepare_mask` crashes.

**Fix:** trilinear-interpolate mask to match latent spatial dims BEFORE tile slicing.

### Finding 9: Reference token injection is LTX2-architecture-general, not Echo-specific

The prefix-token injection mechanism (prepending reference tokens to the video token sequence inside `_process_input`) works on any LTX2.3 checkpoint — vanilla LTX2, Echo, merged variants. Discovered while implementing JoyAI-Echo's memory bank, but the underlying mechanism uses only stock LTX2 components: `SymmetricPatchifier(patch_size=1)`, `latent_to_pixel_coords` with standard `vae_scale_factors` (8/32/32) and `causal_temporal_positioning`, 3D RoPE. No JoyAI-specific weights or pathways required. The model attends to prepended tokens via the standard self-attention path; identity influence emerges from this attention without needing any specialized memory-handling layers.

### Finding 10: Per-frame compressed modulation tensors need num_frames extension, not data extension

LTX2's `CompressedTimestep` stores `data` of shape `(B, num_frames, dim)` and exposes `expand_for_computation()` which returns `(B, num_frames * patches_per_frame, dim)`. When prepending reference tokens that add F_ref frames worth of spatial patches, the naive fix of extending `data`'s sequence dim by `ref_seq_len` is wrong — that breaks the `num_frames * patches_per_frame` invariant. Correct fix: prepend `ref_frames` rows to `data` AND increment `obj.num_frames += ref_frames`. The next `expand_for_computation()` then produces the right size.

The broadcast case (`num_frames=1, patches_per_frame=1`, data shape `(B, 1, dim)`) needs no extension — expansion returns data unchanged and PyTorch broadcasting handles any sequence length naturally. This is why memory-only sampling worked before this finding was applied; the bug only surfaced when conditioning (frame-0 i2v anchor, etc.) caused the model to switch to per-frame compressed modulation storage.

### Finding 11: Reference latent requires process_latent_in normalization

Raw `vae.encode(image)` output is in the VAE's native distribution. The DiT was trained on latents that go through `BaseModel.process_latent_in` (scale + shift normalization) before forward. During sampling, Comfy applies this to the noise latent before passing to diffusion_model — but reference latents injected via the prefix-token path bypass that normalization. Without explicit `process_latent_in` in the Conditioning node, reference tokens have magnitude / mean mismatch with target tokens → manifests as red-tint / off-color artifacts in output.

### Finding 12: Spatial alignment between reference and target is non-optional

Reference's `(H, W)` latent dims must exactly equal target's `(H, W)` latent dims. The modulation extension logic increments `num_frames` by `ref_frames` and relies on `patches_per_frame` being IDENTICAL between reference and target. A spatial mismatch (e.g., reference encoded at 16×20=320 patches, target at 16×19=304 patches) produces a `(num_frames + ref_frames) * 304 = expanded_modulation_size` that's off from vx's actual size by `ref_frames * (320-304)` tokens.

**Two-tier fix:**
1. **Primary (pixel-space, clean):** Conditioning node accepts optional `target_latent` input. When wired, reads `(H_lat, W_lat)`, computes pixel dims `H_lat*32 × W_lat*32`, resizes input image via `F.interpolate(bilinear, antialias=True)` before VAE encoding.
2. **Fallback (latent-space, graceful):** Enable's `_process_input` does bilinear interpolation in latent space when it detects mismatch at sampling time. Handles tiled samplers where each tile has different spatial than the original conditioning target.

### Finding 13: Reference timestep zeroing on released Echo T2V weights produces distorted output

JoyAI's native pipeline zeros reference timesteps to mark them as clean (σ=0). Mathematically this should be correct — the model has explicit support for clean reference tokens. Empirically on Echo's released T2V-only checkpoint, zeroing causes severe distortion (color inversion, broken identity).

Theory: Echo's training distribution didn't include σ=0 reference tokens. The released weights have no learned behavior for that configuration. AdaLN modulation with t=0 produces affine transforms that wreck attention to the reference tokens. Default OFF works better in practice. The toggle remains as `zero_ref_timesteps` on LTX Reference Enable for future i2v-trained checkpoints where this may flip.

---

## DIAGNOSTIC DECISION TREES

### "My upscale pass has hue shift / color drift / pink tone"

```
Q: Is it visible only after sampler runs (not after upsampler alone)?
   YES → Sampler-induced (Finding 1). Use LTX Tiled Sampler.
   NO  → Upsampler itself producing artifact. Try LTX Latent Upsampler (Tiled).
        Or check workflow's upscale model version.

Q: Is the workflow vertical orientation with talking face?
   → Tiled Sampler with audio_carrier_tile=first

Q: Very large output (4K/8K)?
   → audio_carrier_tile=middle, n_tiles=2-4
```

### "My tiled sampler output is flat color in one region"

```
Likely causes:
1. Pre-v2.0 version: wrapper output's video slot returns input ref, not sampled output
   → Update to v2.0+ which uses x0 unflatten technique
2. weight_acc > 1.0 visible in debug: blending math wrong
   → Update to v1.1+ of upsampler_tiled which uses actual-overlap windowing
3. tile_size > axis_size: only 1 tile fits, equivalent to non-tiled
   → Reduce tile_size or increase max_size_for_no_tile so single-tile path is used
```

### "downstream node fails with 'tuple index out of range' at latents[1]"

```
Cause: output wrapper missing audio slot (only one tensor in .tensors)
Likely path: pre-v2.2 single_pass_wrapper returning raw guider.sample() result
Fix: v2.2+ uses x0 unflatten + _reconstruct_samples to ensure both slots present
Test: enable debug=True, look for "output type=NestedTensor" with successful reconstruction
```

### "NotImplementedError: No operator found for memory_efficient_attention_forward"

```
This is environmental, NOT a node bug.
Cause: xformers attention backend incompatibility (often Blackwell GPUs, cc 12.0+)
Fix: launch ComfyUI with --use-pytorch-cross-attention
Reasoning: forces PyTorch SDPA instead of xformers, handles new GPUs and tensor attn_bias
Note: error contains "your GPU has capability (X, Y)" — if X >= 12, this is the cause
```

### "tensor a (X) must match tensor b (Y) at dimension N" in tile loop

```
Cause: flat combined tensor (e.g., shape (1, 1, 2234752)) being accumulated into
       shaped buffer (e.g., (1, 128, F, H, W))
Likely path: pre-v2.0 denoised handling, or x0 unflatten failed silently
Fix: v2.0+ unflattens or skips with warning
Math: 2234752 = 128 × (F×H×W + audio_F×audio_W)
```

### "carrier tile produces blank video region"

```
Cause: wrapper output's .tensors[0] holds input reference, not sampled video
Resolution: v2.0+ captures sampled video from x0 callback via unflatten
            (NOT from tile_result.tensors[0])
```

### Anchor not producing expected effect

```
Q: Is sigmas connected?
   NO → cache timing falls back to manual_calls mode, less reliable
   YES → check cache_at_step value vs schedule length

Q: Is forwards_per_step matched to actual CFG/STG configuration?
   distilled CFG=1: forwards_per_step=1
   standard CFG: forwards_per_step=2
   CFG + STG: forwards_per_step=3 per step it's active
   Variable per-step (e.g. CFG=2 then 1): single value is approximation

Q: Are multiple nodes interfering?
   Face + Latent Anchor co-exist (different sentinel attributes)
   But strength compounds across hooks: 0.10 × 2 nodes ≈ 0.20 effective
```

---

## SAMPLER INTEGRATION DETAILS

### Hooking pattern (used by anchor nodes)

```python
def make_attn1_hook(block_idx):
    def hook(module, inputs, output):
        tensor, wrap = _extract_attn_tensor(output)
        # tensor shape: (B, seq, D)
        # seq = F_tok * H_tok * W_tok (token sequence, F outermost)
        # D = 4096
        # modify tensor via residual blend
        return wrap(modified_tensor)
    return hook

block.attn1.register_forward_hook(make_attn1_hook(block_idx))
```

### Hook coexistence

Each node uses a unique sentinel attribute:
- `HOOK_ATTR_ATTN1 = "_10s_latent_anchor_attn1_hook"` (latent anchor)
- `HOOK_ATTR_ATTN1 = "_10s_aware_anchor_attn1_hook"` (aware)
- `HOOK_ATTR_ATTN1 = "_10s_face_anchor_attn1_hook"` (face)
- `HOOK_ATTR_ATTN2 = "_10s_text_amp_attn2_hook"` (text amplifier)

This allows simultaneous registration. Removing one node's hooks doesn't affect others.

### Sigma capture

Backbone pre-hook captures sigma from kwargs:
```python
to = kwargs.get("transformer_options", {})
sigmas = to.get("sigmas")  # tensor, current sigma at this forward
```

Used for runtime sigma tracking when cache_mode=auto_sigma or for diagnostics.

### Wrapper sampling x0 capture

```python
x0_output = {}
callback = latent_preview.prepare_callback(
    guider.model_patcher, sigmas.shape[-1] - 1, x0_output
)
result = guider.sample(noise, latent, sampler, sigmas, callback=callback)
# After sampling:
if "x0" in x0_output:
    processed = guider.model_patcher.model.process_latent_out(x0_output["x0"])
    # processed is the flat combined tensor for wrapper inputs
```

---

## FAILED APPROACHES (DO NOT RE-PROPOSE)

### Color statistics matching post-sampler
- Tried: `LTXLatentColorRestore` (REMOVED in v1.2.0)
- Why it failed: per-channel mean/std matching captures uniform shifts; doesn't capture spatially-varying drift from sampler-smeared artifacts
- Lesson: by post-sampler, distortion is sub-channel-statistical

### Spatial outlier suppression
- Tried: `LTXLatentOutlierSuppress` (REMOVED in v1.2.0)
- Why it failed: catches pre-sampler artifacts but those propagate into the sampling computation regardless. Also catches legitimate variation, causing block artifacts in clean regions.

### Rotating tall latents to landscape orientation for upscale
- Tried in `LTXVLatentUpsamplerTiled` (v1.0)
- Why it failed: model's tonal drift isn't aspect-ratio dependent; it's baked into the upscaler model weights independent of orientation. Rotation also introduced artifacts.

### MultiDiffusion-style per-step tile coordination
- Considered, not built
- Why deemed wrong: appropriate for heavy denoise from noise; overkill for light refinement. Per-step averaging across tiles loses tile-specific detail that the cosine-window post-blend can preserve cleanly enough.

### Wrapper round-trip for audio sampling (v1.6 and v1.9)
- Tried: pass full wrapper through carrier tile sampling, use output wrapper directly
- Why it failed: output wrapper has .tensors[0] = cached input ref, not sampled video
- Fix that worked (v2.0): use x0 unflatten technique to retrieve actually-sampled tensors

### Two-pass audio capture (v1.8)
- Tried: separate full-schedule sampling pass purely for audio, plain video for video tile
- Why it failed: separate audio sampling doesn't have access to the tile's specific video context, breaks lipsync cross-attention
- Fix that worked (v1.9+): unified single-pass on wrapper with x0 unflatten

### Path 1: parallel reference forward through DiT (bookmarked, not implemented)
- For content-aware anchoring with proper feature-space matching
- Would require running reference through full block stack at matching noise level
- Architectural complexity high; current v2.x of aware uses VAE-space energy modulation instead
- Could be revisited for v3.0+ if results warrant it

---

## CALIBRATION REFERENCE

### Latent Anchor strength
- `0.05` — barely perceptible, baseline check
- `0.10` — typical (default)
- `0.15` — visible effect, sometimes preferred for sensibility/physics
- `0.20-0.30` — strong, can over-damp motion at high values

### Latent Anchor cache_at_step (for 13-step schedule, forwards_per_step=1)
- `3` — early lock-in, "scaffold preservation"
- `6` — mid-sampling (default), peak conditioning alignment
- `9` — late lock-in, mostly refinement free
- `≥ 11` — too late, anchor barely contributes

### Tiled Sampler config (LTX upscale pass)
- `tile_overlap = 8` — reliably hides seams (default)
- `tile_overlap = 6` — minimum for clean blend; smaller causes seams
- `tile_overlap = 12-16` — extra safety margin if seams visible
- `n_tiles = 2` — default, suitable for typical aspect ratios
- `n_tiles = 4` — large outputs (4K+) along one axis

### Face Anchor strength
- `0.10` — typical
- `0.15` — drift recovery / hard cuts
- `0.20+` — strong, can flatten face details

### Energy threshold (Aware variant)
- `0.0` — energy off (uniform mask)
- `0.30` — default, broad selectivity
- `0.50` — above-median energy only
- `0.80` — top ~20% energy only

---

## WORKFLOW PATTERN REFERENCE

### Standard I2V single-pass with identity
```
Model → Latent Anchor Aware (sigmas connected) → KSampler → VAE Decode
                       ↑
               reference_image, vae
```

### Two-pass with upscale-pass quality recovery
```
First-pass model → KSampler (first pass, normal SamplerCustomAdvanced)
                ↓
      LTX Latent Upsampler (Tiled)
                ↓
      Conditioning re-application
                ↓
      LTX Tiled Sampler (audio_pass=tile_carrying, audio_carrier_tile=first)
                ↓
      VAE Decode
```

### Combined face + scene anchoring
```
Model → Latent Anchor Aware → Face Attention Anchor → KSampler
```

### Tiled sampling bypass (debugging)
```
... → LTX Tiled Sampler (bypass_tiling=True) → ...
```
Acts as transparent passthrough to single-pass sampling.

---

## KNOWN COMPATIBILITY ISSUES

### Blackwell GPUs (compute capability ≥ 12.0)
- xformers may lack Blackwell support → attention errors during sampling
- Fix: `--use-pytorch-cross-attention` on ComfyUI launch
- Alternative: update xformers to a Blackwell-compatible build
- Not a node bug; environment issue

### Standard CFG > 1 with anchor
- Compounds hook calls per step (cond + uncond)
- Set `forwards_per_step=2` in advanced mode
- For variable-CFG schedules (e.g., CFG=2 then 1), the single value is approximation

### Distilled CFG=1
- Most extensively tested config
- `forwards_per_step=1` (default)
- All recommended configs above assume this

### Other DiT video models (HunyuanVideo, Wan, etc.)
- Hooks rely on `LTXAVModel.transformer_blocks` and `BasicAVTransformerBlock`
- Will not work as-is — would need adaptation per model

---

## ARCHITECTURAL PRINCIPLES BEHIND THE PACKAGE

1. **Inference-time intervention over training** — these nodes don't retrain; they hook the model and modify intermediate computations during sampling.

2. **Hook → residual blend → integrate naturally** — rather than fighting the model's computation, add a small residual at the right point and let downstream blocks accommodate.

3. **Cosine-similarity matching in centered feature space** — discriminates identity from positional/scaffold features.

4. **Cache mid-sampling representations** — model's representation at peak conditioning alignment is the right anchor target.

5. **Spatial tiling at training-distribution token count** — keeps the model in its "comfort zone" for refinement passes.

6. **Strict typed extraction from wrappers** — never trust `.tensors[0]` for output; capture sampled state via x0 callbacks.

7. **Cosine-windowed blend over averaging** — proper reconstruction at overlap zones with no banding.

---

## CONVERSATIONAL/USAGE GUIDANCE

When users ask broad questions about the package:

- **Tiled Sampler and Latent Anchor Aware are the headline nodes.** If unsure which to recommend, lean toward these.
- **Tiled Sampler solves the upscale-pass quality problem** that most LTX2 users hit. If they describe color drift, hue shift, or "broken upscale pass," recommend this.
- **Latent Anchor Aware is for prompt adherence / scene consistency / physical sensibility.** Use for identity preservation across long sampling.
- **Face Attention Anchor is for face-specific identity preservation.** Best combined with Latent Anchor Aware for both scene-wide and face-targeted effects.
- **Nodes coexist.** Hook into different sentinels. Order: Latent Anchor Aware → Face Anchor → KSampler.

Avoid suggesting:
- Training-time fixes (this is inference-only)
- Approaches that double sampling cost (we have alternatives)
- "Just upgrade the model" (architectural intervention is the point)
- Tools outside ComfyUI (this package is ComfyUI-specific)

If a user reports an issue with terminology that suggests they're using an older version, ask for their `__version__` or git tag. Many issues are resolved in later patches:
- v1.0.0: initial release
- v1.2.0: Tiled Sampler added, color/outlier nodes removed
- v1.2.1: no-tile wrapper passthrough
- v1.2.2: no-tile x0 unflatten
- v1.2.3: bypass_tiling option

---

## TECHNICAL CONSTANTS (LTX2)

- `SPATIAL_PATCH = 1`
- `TEMPORAL_PATCH = 1`
- `LTX_VAE_CHANNELS = 128`
- `TRACK_SHARPNESS = 8.0` (sigmoid steepness for similarity masks)
- VAE compression: 32× spatial / 8× temporal
- DiT feature width: 4096
- Transformer blocks: 48
- Block class: `BasicAVTransformerBlock`

These appear hardcoded in node implementations because they're LTX2-specific architectural constants.
