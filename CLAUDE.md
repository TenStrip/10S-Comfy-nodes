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
- **Text encoder:** Gemma-3-12B variant, 6144-dim embeddings, produces flatter contextual similarities than typical CLIP/T5 encoders

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

### 🎯 LTX Likeness Guide
- **File:** `latent_likeness_guide.py`
- **Class:** `LTXLikenessGuide`
- **Category:** `10S Nodes/Identity`
- **Role:** encodes reference face into conditioning pipeline, produces `reference_info` metadata
- **Mechanism:** face bbox auto-detection (MediaPipe → OpenCV Haar fallback), conditioning attention metadata, latent shape preservation
- **Default config:** `emit_latent=passthrough, face_detect=auto, reference_mask_mode=bbox_softfade`
- **Critical:** `emit_latent=passthrough` is the recommended default. `extend_latent` triggers learned end-keyframe behavior in the model.

### 🪪 LTX Likeness Anchor
- **File:** `latent_likeness_anchor.py`
- **Class:** `LTXLikenessAnchor`
- **Category:** `10S Nodes/Identity`
- **Role:** per-block attn1 hook pulling face-bbox video tokens toward reference identity features
- **Mechanism:** directional pull (magnitude-preserving rotation toward target) or additive (legacy). Reads bbox from `reference_info` or `latent_frame_0`.
- **Hook target:** `block.attn1` output, residual blend
- **Default config:** `strength=0.10, pull_mode=directional, reference_source=auto, sim_threshold=0.50, late_block_falloff=0.0, depth_curve=flat`
- **Recommended:** `strength=0.10-0.18, late_block_falloff=0.4` for most cases
- **Bypass-safe:** stores hook handles, actively removes prior hooks on bypass/re-apply
- **Chains with AwareAnchor:** strengths COMPOUND additively on shared attn1 layer. Reduce individual values when chained (e.g., 0.08 + 0.10).

### 🪪 LTX Likeness Crop
- **File:** `latent_likeness_guide.py` (same file)
- **Standalone face bbox cropper.** Outputs IMAGE + STRING bbox. Useful for previewing detection or feeding cropped face to other workflows.

### 🔎 LTX Latent Face Detector
- **File:** `latent_face_detector.py`
- **Class:** `LTXLatentFaceDetector`
- **Standalone face bbox detector.** Returns normalized bbox STRING for manual wiring.

### 🧠 LTX Likeness Semantic Clamp ⚠ EXPERIMENTAL
- **File:** `latent_likeness_semantic_clamp.py`
- **Class:** `LTXLikenessSemanticClamp`
- **Category:** `10S Nodes/Identity`
- **Status:** Experimental. LTX2 encoder's contextual blending limits correspondence search precision.
- **Mechanism:** monkey-patches `attn2.forward`. Identifies face-modifier-like positive prompt tokens via embedding-space correspondence to a vocabulary, fingerprint-matches positive (skip uncond pass), applies dual-attention-blend (one normal pass + one K/V-suppressed pass) blended by bbox mask.
- **Default config:** `suppression_strength=0.5, face_modifier_text=<built-in>, auto_threshold=p95, similarity_sharpness=16, suppression_floor=0.3, top_k=3, soft_edge_frac=0.15`
- **Critical:** `auto_threshold=p95` is mandatory for LTX2 encoder; raw similarity caps ~0.55. Suppression floor 0.3 eliminates sigmoid soft-tail leak (otherwise 66%+ of tokens get partial suppression which approximates the failed blanket Clamp approach).
- **Healthy diagnostic:** 3-8% of tokens >0.5 suppression; >0.1 and >0.3 buckets should be similar (floor working).

### 💨 LTX Action Amplifier ⚠ EXPERIMENTAL
- **File:** `latent_action_amplifier.py`
- **Class:** `LTXActionAmplifier`
- **Category:** `10S Nodes/Conditioning`
- **Status:** Experimental. Replaces the deprecated blanket Text Amplifier with token-selective scaling.
- **Mechanism:** monkey-patches `attn2.forward`. Single attention pass (vs dual-blend in SemanticClamp). Scales K/V by `(1 + amplification_strength × weight × scale_ceiling)` for matched action verb tokens. Uniform across all video tokens (no bbox).
- **Default config:** `amplification_strength=0.3, scale_ceiling=0.30, action_vocabulary_text=<built-in>, auto_threshold=p95, similarity_sharpness=16, amplification_floor=0.3, top_k=3`
- **Critical:** `scale_ceiling=0.30` caps per-token K/V scaling at +30% max. Higher values risk same noise pattern as deprecated TextAmplifier.
- **Symmetric inverse of SemanticClamp:** same correspondence-search backbone, boost instead of suppress.

### 🔊 LTX Text Attention Amplifier ⚠ DEPRECATED
- **File:** `latent_text_amplifier.py`
- **Class:** `LTXTextAttentionAmplifier`
- **Status:** Deprecated. Blanket attn2 output multiplier produced noise at meaningful strengths. Superseded by Action Amplifier's token-selective approach. Kept for backward compatibility.
- **Default:** `text_amplification=1.30, spatial_focus=0.0`

### 🎯 STG Guider
- **File:** `stg_guider.py`
- **Class:** `LTX2STGGuider` (internal name preserved to avoid collision with Comfy's `STGGuider`); display name "STG Guider"
- **Category:** `10S Nodes/Sampling`
- **Role:** GUIDER node implementing Spatio-Temporal Guidance for LTX2
- **Mechanism:** wraps `comfy.samplers.CFGGuider`. At each step:
  - Pass 1: positive prediction
  - Pass 2: negative prediction (when cfg ≠ 1)
  - Pass 3: perturbed prediction — runs the model with self-attention V-shortcutted (returns V instead of softmax(QK)·V) on transformer blocks in `block_indices`
  - Combines: `pred = pos + (cfg-1)·(pos-neg) + stg_scale·(pos-perturbed)`
  - Optional per-step STD rescale: `pred · (rescale·pos.std/pred.std + (1-rescale))`
- **Default config:**
  - `cfg_per_step="2.0,1.5,1.0,1.0,..."` length must match sigmas
  - `stg_scale_per_step="2.0,1.5,1.0,...,0.7,0.4,0.0,0.0"` taper to 0 in last steps
  - `stg_rescale_per_step="1.0,..."`
  - `block_indices="14, 19"` matches upstream Lightricks default
  - Use `block_indices="9999"` to functionally disable STG
- **Mode flags (per-signal):**
  - `cfg_mode`, `stg_mode`, `stg_rescale_mode` ∈ {`per_step_list`, `sigma_curve`}
  - `sigma_curve` mode: linear interpolation in sigma between `<signal>_max` (at sigma_max) and `<signal>_min` (at sigma=0)
  - Each signal independently mode-switchable
- **Critical implementation details:**
  - Patches every transformer block via `set_model_patch_replace(_, "dit", "double_block", i)`, but only perturbs blocks listed in `target_block_indices` at runtime
  - V-shortcut applies ONLY to the FIRST `optimized_attention` call per block (self-attention attn1). Subsequent calls (cross-attention attn2 with Q from text, K/V from video) fall through unmodified. This is enforced via per-block instance flag in `_PatchAttention`, plus shape guard `q.shape[1] == k.shape[1] == v.shape[1]`.
  - Perturbing cross-attention causes shape mismatches with `_attention_with_guide_mask` and `adaLN` paths — guard is essential.
  - Sigmas input MUST match the sampler's actual schedule (including any easing). Mismatch silently misaligns per-step parameters.
- **Anchor compatibility:** anchors fire inside the model forward; STG guider orchestrates three forwards from outside the model. Anchors automatically fire on all three passes (positive, negative, perturbed) since their hooks are unconditional.
- **Diagnostic mode:** when `debug=True`, the first perturbed pass logs which attention calls were V-shortcutted, with shape info — useful for verifying no cross-attention got perturbed accidentally.

### 🔍 LTX Latent Upsampler (Tiled)
- **File:** `latent_upsampler_tiled.py`
- **Drop-in for `LTXVLatentUpsampler`** with spatial tiling for extreme aspect ratios. Auto-detects upscale ratio (x1.5, x2). NOTE: empirical finding showed that the upscaler itself wasn't the source of color shift (the sampler operating on upscaled token counts was). This node remains useful as a memory-safer upscaler for very large inputs.

### 🔍 LTX Model Inspector
- **Diagnostic node** for inspecting LTX2 module structure. Used during node development.

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

Used throughout: latent_anchor, latent_anchor_aware, latent_likeness_anchor.

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

### Finding 9: Inference-time interventions have a hard ceiling for unique-face preservation

**Empirical confirmation:** a 30-image, 15-minute LoRA training run via LTX-Video-Trainer produces dramatically better unique-face preservation than ANY inference-time chain in this package can achieve. The model's split attention between text-video and image-video modes makes it more comfortable drawing a familiar (trained) face than keeping a unique unfamiliar one intact.

**Conclusion:** activation-level interventions stabilize identity the model already knows; they cannot impart new identity knowledge. For unique faces:
- Right answer: train a subject LoRA (~15 min, ~30 images)
- This package's role: stabilization complement to LoRA, OR for faces partially known to the base model

The Likeness suite is best positioned as augmenting LoRA workflows, not replacing them.

### Finding 10: LTX2 text encoder produces flatter similarities than CLIP/T5

Gemma-3-12B's contextual attention heavily blends adjacent tokens. Result:
- Raw cosine similarity values cap ~0.55 even for literal word matches in face-modifier vocabulary
- Per-token correspondence search is noisier than in image-feature space
- Mandatory: `auto_threshold=p95` percentile-based gating instead of absolute thresholds
- Top-K mean (K=3) instead of max similarity to require multiple confirming matches
- Hard `suppression_floor=0.3` to eliminate sigmoid soft-tail leak

Without these, SemanticClamp and ActionAmplifier degrade to blanket-scaling failure mode of the deprecated TextAmplifier.

### Finding 11: Hook bypass leak when `model.clone()` shares modules

PyTorch hooks registered on `transformer_blocks[i].attn1` persist across `model.clone()` calls because clones share the underlying ModuleList. Early-return-on-bypass without cleanup leaves prior runs' hooks installed and firing silently.

**Fix:** store PyTorch handle returned by `register_forward_hook`, call `.remove()` on bypass. Same pattern for monkey-patched `forward` methods (store original, restore on bypass). Applied to LikenessAnchor v1.2+ and SemanticClamp/ActionAmplifier from v1 onward.

**Symptom:** visual changes persist after setting bypass=True; only resolve after deleting the node and clearing models.

### Finding 12: Latent extension triggers learned keyframe behavior regardless of conditioning

Even when conditioning explicitly marks the appended frame as a silent reference, the *spatial pattern* of preserved content at the temporal end of the latent triggers learned end-keyframe behavior. LikenessGuide v1.4+ defaults to `emit_latent=passthrough` for this reason — the guide's effects flow through `reference_info` metadata, not through latent extension.

### Finding 13: Anchors + Tiled Sampler produce tile-boundary artifacts

**Empirical observation:** in two-pass workflows (anchors during first pass → tiled refinement), the second-pass output shows visible oversaturation / "burned features" in the tile region containing the anchor target (face). The artifact is deterministic with seed and worsens with aspect ratios where the anchor-affected content is concentrated in one tile.

**Root cause (architectural, not a bug):** anchors operate per-forward-pass on whatever spatial extent the model receives. When Tiled Sampler slices the latent into tiles, each tile presented to the model is a sub-extent (e.g., (F=46, H=27, W=26) instead of full (F=46, H=46, W=26)). The anchor's hook fires per-tile with bbox normalization, similarity matching, and cached features computed in the tile's LOCAL coordinates. The anchor has no awareness it's processing a tile rather than a full latent.

Even when anchors are placed ONLY in the first-pass branch (separate model from the Tiled Sampler), the first-pass *latent* carries the anchor's effect baked into its statistical distribution — variance contraction in the bbox region, deeper outlier tails from directional pull. This non-uniform distribution survives upscaling and creates tile-boundary statistical mismatch during tiled refinement.

**Symptom signature:**
- Tile containing face content: lower per-channel std, deeper negative outliers (saturated when VAE-decoded)
- Other tiles: natural per-channel stats
- Per-channel mean difference between tiles can be 2-3+ for affected channels

**Failed mitigation attempt:** global per-channel mean/std matching across tiles (the rolled-back `match_tile_stats` option in Tiled Sampler v2.5). The artifact is spatially-localized within each tile (face bbox region only), so global per-channel matching is too coarse — it can't correct localized statistical pockets without spatial awareness.

**Working mitigation (architectural):** use a two-model-branch workflow. One branch goes through anchors into first-pass sampler; a separate branch (clean model from the same MODEL source) goes into Tiled Sampler. The first-pass latent has the anchor's effect; the tiled refinement runs on a clean model. This is now documented as the standard pattern in README.

### Finding 14: STG attention scope — self-attention only, never cross-attention

**Empirical confirmation:** the V-shortcut perturbation (replacing `optimized_attention` with `return v`) MUST be limited to self-attention (attn1 — first attention call per block). Applying it to cross-attention (attn2 — second call, where Q is from text and K/V from video) causes:
- Shape mismatch with LTX2's `_attention_with_guide_mask` paths (output sliced as `out[:, :guide_start, :]` expects specific shape that V doesn't match)
- Shape mismatch with `adaLN` gating (output multiplied by Q-side gate that has text-token shape ~1024, while V has video-token shape ~10752)

**Implementation:** STG Guider's `_PatchAttention` uses a per-block instance flag (`_first_call_done`) that flips True on the first attention call, plus a guard `q.shape[1] == k.shape[1] == v.shape[1]` that confirms self-attention by sequence-length equality. Cross-attention falls through to the original attention function.

**Also confirmed:** the original Lightricks STG used `attn_idx=[0]` filtering which is implicitly "skip first attention call per block." Our implementation makes this explicit. STG fundamentally requires partial perturbation — perturbing ALL blocks (no `block_indices` filtering) produces catastrophically degraded output (dark grey, destroyed audio) because the perturbed prediction becomes mostly noise rather than a meaningfully different prediction.

### Finding 15: STG late-step perturbation corrupts refined features

**Empirical observation:** STG with constant `stg_scale > 0` throughout sampling produces oversaturation / burned features signature. Tapering `stg_scale_per_step` to 0.0 in the last ~30% of steps eliminates this.

**Mechanism:** at high sigma (early sampling), the latent is mostly noise. The perturbed prediction (with self-attention skipped) is also mostly noise-like — the (pos - perturbed) difference encodes useful structural information. STG works as intended.

At low sigma (late sampling), the latent has committed structure. The perturbed prediction is now a *sharp but broken* version of the refined image. The (pos - perturbed) difference encodes "how the refined image looks broken" — adding that signal to the noise prediction injects exactly the burned/dark artifacts. STG's mechanism (push away from broken reference) becomes counterproductive when the reference is broken in detail-relevant ways.

**Recommended schedule:** STG active in first ~60-70% of steps (structure formation), taper to 0 in last 30% (detail refinement).

### Finding 16: Sigma alignment between sampler and guider is critical

**Empirical bug:** if the sampler runs on eased/normalized sigmas but the STG Guider's `sigmas` input is the original un-eased schedule, per-step parameter lookup silently misaligns. The guider's `_index_for_sigma()` finds the closest sigma in its list at-or-above the current step's actual sigma, but the un-eased and eased schedules differ — so the parameters applied don't match what the sampler thinks it's at.

**Fix:** pass the EXACT same `SIGMAS` tensor to both the STG Guider and the sampler. Any normalization or easing nodes must be upstream of both. Documented as the primary "must do" in the STG Guider tooltip and docstring.

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
   AwareAnchor + LikenessAnchor co-exist (different sentinel attributes)
   But strengths COMPOUND additively on shared attn1 layer
   AwareAnchor 0.10 + LikenessAnchor 0.10 → effective ~0.20
   Reduce individual values when chained.

Q: Visual changes persist after bypass=True?
   Pre-v1.2: hook bypass leak (Finding 11)
   Fix: update to v1.2+ which stores handles and removes on bypass
```

### SemanticClamp / ActionAmplifier produces too much / too little suppression

```
Q: Diagnostic shows >40% tokens with moderate suppression?
   → Threshold too low. With auto_threshold enabled, this means the score
     distribution is very flat. Increase top_k to 5+ or use auto_threshold=p98.
     Without auto_threshold, the literal sim_threshold is too low.

Q: Diagnostic shows <1% tokens with any suppression?
   → Threshold too high OR prompt has no matching vocabulary.
     Try auto_threshold=p95 if disabled. Verify prompt actually contains
     face-modifier or action words.

Q: Healthy distribution = >0.5 and >0.3 buckets ~3-10% of tokens,
   >0.1 and >0.3 buckets similar (floor working).
```

### Likeness Anchor produces color desaturation / variance collapse

```
Cause: many tokens pulled toward same target direction narrows token-distribution variance
       in feature space. Per-token magnitude preserved (directional mode does this) but
       cross-token spread shrinks → visible color/contrast reduction.

Mitigations:
- Lower strength (0.05-0.15 typical safe range)
- Enable late_block_falloff=0.3-0.4
- Tighten sim_threshold to pull fewer tokens
- Confirm bypass actually clears prior hooks (Finding 11 — pre-v1.2 leak)
```

### Tiled Sampler output: first half oversaturated / "burned features" in tile 1

```
Cause: upstream anchors (LikenessAnchor / AwareAnchor / LatentAnchor) shaped the first-pass
       latent into a non-uniform statistical distribution. The anchored region (typically
       face bbox) has narrower per-channel variance and deeper outlier tails. This survives
       upscaling and shows as visible saturation difference at tile boundaries (Finding 13).

Diagnosis:
- Enable debug=True on Tiled Sampler; compare tile 1's per-channel mean/std vs tile 2's.
  If per-channel mean range differs by 2-3+ between tiles, anchor influence is confirmed.

Fix (architectural — the right one):
- Restructure workflow so the MODEL connects to TWO separate branches:
  - Branch A: through anchors → first-pass sampler
  - Branch B: clean (no anchors) → Tiled Sampler
- The first-pass latent carries the anchor's effect; the upscale-pass refines clean

NOT a fix:
- Stat matching across tiles (rolled back as v2.5 match_tile_stats option) —
  the artifact is spatially-localized within tiles, global per-channel matching can't fix it
- Lower anchor strength alone — reduces but doesn't eliminate at strengths visible enough to help
- More tile overlap — doesn't help, the difference is content-level
```

### STG output: oversaturation / burned features

```
Q: Are stg_scale values non-zero in the last 30% of steps?
   YES → STG perturbing already-refined features (Finding 15)
   Fix: taper stg_scale_per_step to 0.0 in last 4-5 steps of a 13-step schedule
        Example: "2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.4, 0.0, 0.0, 0.0, 0.0"

Q: Is block_indices set to many blocks?
   YES → Over-perturbation. The default "14, 19" is calibrated with stg_scale ~2.0.
         More blocks ≈ more perturbation; reduce stg_scale proportionally or revert to default.

Q: Do the sampler and guider receive the SAME sigmas?
   Check: if you have a normalize/ease node between schedule and sampler, the guider
   MUST receive the SAME post-easing sigmas (Finding 16).
   Symptom of mismatch: STG values fire at wrong steps; produces unpredictable artifacts.
```

### STG appears to do nothing

```
Q: Is block_indices set to "9999" or similar out-of-range value?
   YES → That's the explicit disable. STG perturbation is fully off.
        Re-enable with "14, 19" (default) or other valid indices 0-47.

Q: Are stg_scale_per_step values all 0.0?
   YES → STG contribution is zero. Set non-zero values.

Q: Is sigmas tensor empty or wrong length?
   Check debug=True output. The guider's printed sigmas should match the sampler's.
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

handle = block.attn1.register_forward_hook(make_attn1_hook(block_idx))
# Store handle for later .remove() on bypass
setattr(attn1, "_10s_likeness_anchor_handle", handle)
```

### Monkey-patch pattern (used by SemanticClamp / ActionAmplifier)

```python
original_forward = attn2.forward
setattr(attn2, ORIGINAL_FORWARD_ATTR, original_forward)

def patched_forward(*args, **kwargs):
    # Fingerprint check against captured positive tensor
    if not _matches_positive(args, state):
        return original_forward(*args, **kwargs)
    # Modify K/V tensor, call original with modified args
    ...

attn2.forward = patched_forward
```

On bypass: restore `attn2.forward = getattr(attn2, ORIGINAL_FORWARD_ATTR)`.

### Hook coexistence

Each node uses a unique sentinel attribute:
- `HOOK_ATTR_ATTN1 = "_10s_latent_anchor_attn1_hook"` (latent anchor)
- `HOOK_ATTR_ATTN1 = "_10s_aware_anchor_attn1_hook"` (aware)
- `HOOK_ATTR_ATTN1 = "_10s_likeness_anchor_attn1_hook"` (likeness)
- `HOOK_ATTR_ATTN2 = "_10s_text_amp_attn2_hook"` (deprecated text amp)
- `HOOK_ATTR_ATTN2 = "_10s_likeness_semantic_clamp_attn2_hook"` (semantic clamp)
- `HOOK_ATTR_ATTN2 = "_10s_action_amp_attn2_hook"` (action amplifier)

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

### Conditioning fingerprint (SemanticClamp / ActionAmplifier)

To distinguish positive from negative pass at runtime (when CFG > 1):
```python
def _fingerprint_tensor(t):
    shape = tuple(t.shape)
    flat = t.flatten().detach().cpu()
    idxs = torch.linspace(0, flat.numel() - 1, 16, dtype=torch.long)
    vals_rounded = tuple(round(flat[idxs[i]].item(), 4) for i in range(16))
    return (shape, vals_rounded)
```

Compute at apply time on positive conditioning, compare at runtime against incoming K/V tensor. Mismatch = uncond pass, pass through unchanged.

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

### Latent extension for face reference (LikenessGuide pre-v1.4)
- Tried: append face reference as additional latent frame, mark via conditioning as silent
- Why it failed: spatial pattern of preserved content at temporal end triggers learned end-keyframe behavior regardless of conditioning marking
- Fix that worked (v1.4): `emit_latent=passthrough` default, effects flow through reference_info metadata only

### Hard noise_mask=0.0 preservation
- Tried: explicit mask preventing certain regions from receiving noise
- Why it failed: decoder periodic artifacts in preserved regions

### Zeroing non-bbox reference latent regions
- Tried: keep face region, zero rest
- Why it failed: content cliff at zero boundary

### Additive pull mode (LikenessAnchor pre-v1.2)
- Tried: add residual directly to attn1 output
- Why it failed: produced color fade because rotation magnitude wasn't preserved
- Fix that worked: directional mode (magnitude-preserving feature rotation)

### Blanket attn2 magnitude scaling (deprecated LikenessClamp removed v1.5.0)
- Tried: scale down all attn2 output by `(1 - strength × bbox_mask)` in face region
- Why it failed: attn2 carries ALL text-conditioning influence including scene/style/composition. Scaling wholesale cuts off the model's primary path to know what to compose in that region → catastrophic noise artifacts.
- Fix that worked: SemanticClamp v1.0 — token-selective suppression on K/V via correspondence search, only matched modifier tokens affected, leaves non-modifier tokens (positional, scene, style) intact

### Blanket text amplification (deprecated TextAmplifier)
- Tried: multiply all attn2 output by constant >1.0
- Why it failed: shifts attention mass non-selectively, boosts pad tokens and noise positions, produces noise
- Fix that worked: ActionAmplifier — token-selective via correspondence search, capped at +30% K/V scaling per matched token, only verb/motion vocabulary affected

### Per-token magnitude scaling (max single-token similarity)
- Tried: derive suppression weight from `max(sim(pos_token, mod_token) for mod_token in vocab)`
- Why it failed: any single outlier match in 100+ modifier tokens triggers suppression, false positives dominate
- Fix that worked: top-K mean (K=3) requires multiple confirming matches

### Path 1: parallel reference forward through DiT (bookmarked, not implemented)
- For content-aware anchoring with proper feature-space matching
- Would require running reference through full block stack at matching noise level
- Architectural complexity high; current v2.x of aware uses VAE-space energy modulation instead
- Could be revisited for v3.0+ if results warrant it

### Token-positional reordering ("token organizer" concept)
- Conceptual proposal: reorder tokens in conditioning to put high-priority tokens first
- Why deemed wrong: positional encoding is baked into encoded token features. Reordering after encoding breaks positional information; reordering before encoding just rewrites the prompt. Transformer attention is set-based with positional encoding, not sequence-based.

### Multi-image reference attention hook
- Considered as extension of LikenessGuide
- Status: bookmarked, not built. Inference-time, still subject to the ceiling vs LoRA (Finding 9).

### Inline LoRA training in custom node
- Considered for unique-face workflows
- Why deemed inappropriate: competes with mature LTX-Video-Trainer. Memory budget tight on 32GB cards. Training takes 15 min minimum; not meaningfully faster than just running the trainer separately and loading the LoRA. Inline transient weights don't save user effort vs saved LoRA file.

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

### Likeness Anchor strength
- `0.05-0.10` — gentle, baseline
- `0.10-0.18` — typical operating range
- `0.20-0.30` — strong, watch for variance collapse (Finding 9 mitigation)
- `> 0.40` — risk of color desaturation from cross-token variance contraction

### SemanticClamp suppression_strength
- `0.3` — gentle, recommended start
- `0.5-0.7` — typical for hard expression prompts
- `0.8-1.0` — strong, can produce rigid face

### ActionAmplifier amplification_strength × scale_ceiling
- `0.3 × 0.30` (default) = max +9% K/V per token
- `0.5 × 0.30` = max +15% K/V per token
- `1.0 × 0.30` = max +30% K/V per token (recommended ceiling)
- Higher scale_ceiling risks blanket TextAmplifier noise pattern

### Tiled Sampler config (LTX upscale pass)
- `tile_overlap = 8` — reliably hides seams (default)
- `tile_overlap = 6` — minimum for clean blend; smaller causes seams
- `tile_overlap = 12-16` — extra safety margin if seams visible
- `n_tiles = 2` — default, suitable for typical aspect ratios
- `n_tiles = 4` — large outputs (4K+) along one axis

### Energy threshold (Aware variant)
- `0.0` — energy off (uniform mask)
- `0.30` — default, broad selectivity
- `0.50` — above-median energy only
- `0.80` — top ~20% energy only

### auto_threshold percentile mode (SemanticClamp / ActionAmplifier)
- `p90` — top 10% (broader effect)
- `p95` — default, balanced
- `p98` — top 2% (narrower, only clear matches)
- `p99` — top 1% (very tight)
- `disabled` — use literal similarity_threshold value

---

## WORKFLOW PATTERN REFERENCE

### Standard I2V single-pass with identity
```
Model → Latent Anchor Aware (sigmas connected) → KSampler → VAE Decode
                       ↑
               reference_image, vae
```

### I2V with face-region preservation
```
LikenessGuide(image, vae, pos, neg)
    ↓ pos', neg', reference_info
LikenessAnchor(model, reference_info, strength=0.10-0.18)
    ↓ model'
KSampler → VAE Decode
```

### Maximum identity (LoRA + inference stabilization)
```
Base model + Subject LoRA (trained via LTX-Video-Trainer, 30 images, 15 min)
    ↓
LikenessGuide → LikenessAnchor → KSampler → VAE Decode
```

### Two-pass with upscale-pass quality recovery (CORRECTED — branch split)

```
MODEL ─┬─→ LikenessAnchor / AwareAnchor → KSampler (first pass)
       │                                       ↓
       │                            [first-pass latent]
       │                                       ↓
       │                          LTX Latent Upsampler (Tiled)
       │                                       ↓
       │                          Conditioning re-application
       │                                       ↓
       └─→ (clean, NO anchors) → LTX Tiled Sampler (audio_pass=tile_carrying)
                                                ↓
                                           VAE Decode
```

**Critical:** the MODEL source connects to TWO branches. Anchors go ONLY in the first-pass branch. The Tiled Sampler model is clean. See Finding 13 for why — anchors during tiled refinement produce visible tile-boundary artifacts. The first-pass latent carries the anchor's effect; the upscale-pass refines that latent with a clean model.

### STG-augmented sampling (motion-heavy content)
```
Model → [optional: anchors] → STG Guider (sigmas, per-step lists) → SamplerCustomAdvanced (same sigmas)
                                                                                ↓
                                                                           VAE Decode
```
- Use in place of standard CFGGuider when motion quality matters
- block_indices="14, 19" default — perturbs 2 mid-depth blocks
- Taper stg_scale to 0 in last 30% of steps (Finding 15)
- block_indices="9999" functionally disables STG while keeping CFG schedule

### Combined scene + face anchoring (advanced)
```
Model → Latent Anchor Aware → Likeness Anchor → KSampler
        strength=0.08          strength=0.10
```
Both hook attn1, strengths compound. Reduce individual values.

### Experimental full chain (use sparingly)
```
Model → AwareAnchor → LikenessAnchor → SemanticClamp → ActionAmplifier → KSampler
            ↑              ↑                ↑                ↑
        attn1 hook    attn1 hook       attn2 patch      attn2 patch
```
Each at gentle strengths. attn2 patches stack sequentially (one wraps the other at runtime). Cost ~3x at attn2 layer (small fraction of total step time).

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

### MediaPipe optional dependency
- Latent Face Detector and Likeness Guide use MediaPipe when available
- Falls back to OpenCV Haar cascades when MediaPipe missing
- Install with `pip install mediapipe` for better detection accuracy

---

## ARCHITECTURAL PRINCIPLES BEHIND THE PACKAGE

1. **Inference-time intervention over training (with acknowledged ceiling)** — these nodes don't retrain; they hook the model and modify intermediate computations during sampling. For unique-face preservation specifically, this approach has a real ceiling vs subject-LoRA training. The package is positioned as complement to LoRA workflows.

2. **Hook → residual blend → integrate naturally** — rather than fighting the model's computation, add a small residual at the right point and let downstream blocks accommodate.

3. **Cosine-similarity matching in centered feature space** — discriminates identity from positional/scaffold features.

4. **Cache mid-sampling representations** — model's representation at peak conditioning alignment is the right anchor target.

5. **Spatial tiling at training-distribution token count** — keeps the model in its "comfort zone" for refinement passes.

6. **Strict typed extraction from wrappers** — never trust `.tensors[0]` for output; capture sampled state via x0 callbacks.

7. **Cosine-windowed blend over averaging** — proper reconstruction at overlap zones with no banding.

8. **Token-selective conditioning intervention over blanket** — correspondence search to identify specific tokens, then per-token modification. Blanket scaling failed (deprecated TextAmplifier, deprecated LikenessClamp); selective scaling works (ActionAmplifier, SemanticClamp).

9. **Adaptive thresholding for encoder-agnostic operation** — `auto_threshold=p95` and similar percentile-based gates adapt to LTX2's flatter contextual embeddings vs typical CLIP/T5 absolute-value thresholds.

10. **Bypass-safe hook management** — store PyTorch handles, actively remove on bypass. Prevents prior-run hook leak from `model.clone()` shared modules.

---

## CONVERSATIONAL/USAGE GUIDANCE

When users ask broad questions about the package:

- **Tiled Sampler and Latent Anchor Aware are the headline nodes.** If unsure which to recommend, lean toward these.
- **Tiled Sampler solves the upscale-pass quality problem** that most LTX2 users hit. If they describe color drift, hue shift, or "broken upscale pass," recommend this.
- **Latent Anchor Aware is for prompt adherence / scene consistency / physical sensibility.** Use for identity preservation across long sampling.
- **For face-region identity:** the Likeness suite (Guide + Anchor) is the current path. For unique faces unknown to the base model, recommend training a LoRA via LTX-Video-Trainer; the Likeness suite then complements rather than replaces that.
- **Semantic Clamp and Action Amplifier are experimental.** Don't recommend as primary tools; mention them as available experiments when users specifically ask about token-selective conditioning.
- **Nodes coexist.** Hook into different sentinels. Common order: Latent Anchor Aware → Likeness Anchor → KSampler.
- **Acknowledge the inference-time ceiling.** When users push for stronger unique-face preservation, point them to LoRA training rather than turning up strength values past safe ranges.

Avoid suggesting:
- Training-time fixes via this package (it's inference-only; for training, point to LTX-Video-Trainer)
- Approaches that double sampling cost (we have alternatives)
- "Just upgrade the model" (architectural intervention is the point)
- Tools outside ComfyUI (this package is ComfyUI-specific)
- The deprecated Text Amplifier as primary recommendation (kept for backward compat only)
- LikenessClamp (removed in v1.5.0, replaced by SemanticClamp)
- FaceAttentionAnchor (removed in v1.6.0, replaced by LikenessAnchor)

If a user reports an issue with terminology that suggests they're using an older version, ask for their `__version__` or git tag. Many issues are resolved in later patches:
- v1.0.0: initial release
- v1.2.0: Tiled Sampler added, color/outlier nodes removed
- v1.2.1: no-tile wrapper passthrough
- v1.2.2: no-tile x0 unflatten
- v1.2.3: bypass_tiling option
- v1.5.0: LikenessClamp removed, SemanticClamp added
- v1.6.0: Likeness suite, ActionAmplifier added, FaceAttentionAnchor removed, bypass-safe hook management
- v1.7.0: STG Guider added (block-selective, per-step lists + sigma_curve modes), anchor+tile interaction documented (Finding 13), STG late-step taper documented (Finding 15), sigma-alignment requirement documented (Finding 16), Tiled Sampler v2.5 with diagnostic instrumentation

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
- Text encoder dim: 6144 (Gemma-3-12B)
- Text encoder max similarity for face-modifier matching: ~0.55 (flatter than CLIP)
- ID-LoRA upstream merge: ComfyUI PR #13111 (`LTXVReferenceAudio` for voice identity)

These appear hardcoded in node implementations because they're LTX2-specific architectural constants.
