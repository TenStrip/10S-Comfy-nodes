# 10S Nodes for ComfyUI

A collection of custom nodes for ComfyUI focused on identity preservation, latent-space stabilization, and upscale-pass quality for the **LTX2 video diffusion model** by Lightricks. Nodes operate via PyTorch forward hooks on the DiT backbone — no model retraining required.

> **Compatibility note:** these nodes are specifically tuned for LTX2/LTX-AV (the dual-stream video+audio DiT). They will not work as-is on other diffusion models. The hooks rely on LTX2's specific block structure (`BasicAVTransformerBlock`).

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TenStrip/10S-Comfy-nodes.git 10S_Nodes
```

Restart ComfyUI. The new nodes appear under `10S Nodes/` in the node-add menu.

No external dependencies beyond ComfyUI's existing environment (PyTorch, comfy package). All nodes are pure Python.

---

## Node Reference

All nodes appear in the menu under one of these categories:
- `10S Nodes/Identity` — DiT hook-based interventions
- `10S Nodes/Latent` — direct latent-space operations
- `10S Nodes/` — utility/legacy nodes

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

**Recommended starting config for general identity preservation:**
```
strength            = 0.10
inject_mode         = tracked
depth_curve         = flat
spatial_prior       = 0.5
anchor_upsample     = 2
```

For hard scene cuts:
```
strength            = 0.15
inject_mode         = tracked_correction
depth_curve         = late_focus
```

---

### 🎯 LTX Latent Anchor

**Category:** `10S Nodes/Identity`

Whole-scene self-anchoring regularizer. Caches the model's representation of the anchor frame at a chosen sampling step, then pulls all subsequent computation toward that cached state. Empirically improves prompt adherence, scene composition consistency, and physical sensibility across long sampling chains.

**How it works:** at sampling step N, snapshot the anchor frame's features at every block depth. Use those as a pull target for subsequent computation, restoring "the conditioning manifold the model converged on at peak alignment."

**Key parameters:**
- `sigmas` (SIGMAS input) — connect from your sigmas node for predictable cache timing
- `strength` — pull magnitude (typical 0.05-0.15)
- `cache_at_step` — sampling step at which to lock the anchor (typical 3-9 on 13-step run)
- `similarity_threshold` — cosine sim cutoff for the pull mask (default 0.50)
- `decay_with_distance` — per-frame strength decay (0 = uniform, higher = anchor frame more pulled, distant frames freer to move)

**Simple/advanced mode:** the simple-mode default exposes 5 essential knobs. Toggle `advanced_mode=True` for cache_mode (`schedule`/`live_extraction`/`manual_calls`), `forwards_per_step` (CFG-aware), `depth_curve`, `block_index_filter`, and others.

**Recommended starting config:**
```
sigmas               = [connected]
strength             = 0.08
cache_at_step        = 6
similarity_threshold = 0.50
decay_with_distance  = 0.0
```

For sensibility/physics emphasis with motion preservation:
```
strength             = 0.15
cache_at_step        = 3
decay_with_distance  = 0.5
```

---

### 🎯 LTX Latent Anchor Aware

**Category:** `10S Nodes/Identity`

Content-aware variant of the latent anchor. Adds optional spatial weighting from an external reference image: high-energy regions of the reference get more anchor pull, low-energy regions get less. The matching itself uses the running latent (correct feature space); the reference contributes only spatial focus.

**Key additions over basic Latent Anchor:**
- `reference_image` (IMAGE input) — external reference for spatial energy
- `vae` (VAE input) — required if reference_image is connected
- `energy_latent` (LATENT input) — alternative direct VAE-space energy source (e.g. for piping in pre-existing conditioned latents)
- `energy_threshold` — cutoff above which tokens get pulled (default 0.30)
  - 0.0 = uniform mask (energy off, equivalent to plain Latent Anchor)
  - 0.30 = pull above-lower-third energy regions
  - 0.50 = pull above-median energy
  - 0.80 = pull only top ~20% energy regions

**Recommended starting config (face/identity reference):**
```
reference_image      = [face crop or full reference]
vae                  = [LTX VAE]
sigmas               = [connected]
strength             = 0.10
cache_at_step        = 6
similarity_threshold = 0.50
energy_threshold     = 0.30
```

---

### 🔊 LTX Text Attention Amplifier

**Category:** `10S Nodes/Identity`

Modulates text cross-attention (`attn2`) influence per block. Compensates for **conditioning dilution at upscaled token counts** — when a second sampling pass operates on a 2× upscaled latent (4× more tokens), each video token receives proportionally less text conditioning influence per attention pass. This manifests as hue shifts, color drift, and prompt adherence loss in the upscale-pass output.

The amplifier multiplies `attn2` output per block. Bidirectional: < 1.0 suppresses text influence, > 1.0 amplifies.

**Key parameters:**
- `text_amplification` — multiplier (1.0 = no change; 1.2-1.5 typical for upscale dilution recovery; < 1.0 for suppression)
- `spatial_focus` — 0.0 = uniform, > 0.0 = Gaussian-weighted with full amplification at center, no amp at edges (model "comfort zone" experiment)
- `block_index_filter` — limit which blocks get the amplification

**Recommended for upscale-pass dilution recovery:**
```
text_amplification = 1.30
spatial_focus      = 0.0
```

For reducing prompt over-fitting:
```
text_amplification = 0.70
spatial_focus      = 0.0
```

---

### 🔍 LTX Latent Upsampler (Tiled)

**Category:** `10S Nodes/Latent`

Drop-in replacement for ComfyUI's stock `LTXVLatentUpsampler`. Adds spatial tiling with cosine-windowed overlap blending to address upscale-model failure modes at extreme aspect ratios. Auto-detects upscale ratio (works with x1.5, x2, etc.). Auto-skips tiling for small inputs.

**Key parameters:**
- `tile_size` — spatial tile size in latent tokens (default 24, ~768 pixels)
- `overlap` — overlap between adjacent tiles (default 8)
- `max_size_for_no_tile` — skip tiling if both spatial dims ≤ this (default 32)
- `rotate_for_landscape` — transpose H/W before upscaling, then rotate back (experimental)

For most cases the defaults work. Lower `max_size_for_no_tile` to force tiling on smaller inputs for testing.

---

### 🎨 LTX Latent Color Restore

**Category:** `10S Nodes/Latent`

Per-channel statistics matching to undo color drift introduced by latent transformations. Takes a clean reference latent and adjusts the input latent so its per-channel mean and std match the reference's, while preserving spatial detail.

**Inputs:**
- `samples` — latent with potential color drift
- `reference` — clean reference latent

**Key parameters:**
- `strength` — 0.0 (passthrough) to 1.0 (full match)
- `scope` — `global` (preserves frame-to-frame variation) or `per_frame` (more aggressive)

Typical use: post-LTX-upsampler color correction, with the pre-upscale latent as reference.

---

### 🧹 LTX Latent Outlier Suppress

**Category:** `10S Nodes/Latent`

Detects and suppresses spatially localized outlier positions in a latent relative to a clean reference distribution. Z-scores every position against per-channel reference statistics; flagged positions are replaced via spatial in-painting from non-flagged neighbors.

Useful when upscaling introduces concentrated point artifacts in specific regions/frames that would otherwise be smeared by subsequent sampling into broader distortions.

**Inputs:**
- `samples` — post-upscale latent
- `reference` — pre-upscale latent

**Key parameters:**
- `z_threshold` — outlier detection threshold (default 4.0; higher = more selective)
- `channel_quorum` — number of channels needing to exceed threshold for a position to be flagged (default 1)
- `suppression_radius` — spatial radius for in-painting (default 3)

---

### 🔍 LTX Model Inspector

**Category:** `10S Nodes/Diagnostic`

Diagnostic node for inspecting LTX2 model structure — useful when developing new hook-based nodes. Lists all transformer modules, prints parameter counts, and traces tensor shapes through the forward chain.

---

### Other Nodes

**Latent post-processors** (`10S Nodes/`): existing utilities including `LatentTemporalUpsampler` and related operations on the latent stream. See source for full list.

---

## Recommended Workflow Patterns

### Single-pass I2V with identity preservation

```
LoadModel → LTX Latent Anchor → KSampler → VAE Decode
                  ↓
         (sigmas connected)
```

For face-targeted preservation, replace Latent Anchor with Face Attention Anchor. For both broad scene stabilization AND face-specific correction, chain them: `Latent Anchor → Face Anchor → KSampler`.

### Upscale-pass quality recovery

```
First-pass output → LTX Latent Upsampler (Tiled) → 
   conditioning re-application → 
   LTX Text Attention Amplifier → KSampler (light, low denoise)
```

The Text Amplifier on the upscale-pass model recovers conditioning fidelity that's diluted by the 4× larger token count.

### Combined identity + scene stabilization (chained)

```
Model → LTX Latent Anchor → LTX Face Attention Anchor → KSampler
```

All hook-based identity nodes coexist on the same model — different sentinel attributes prevent interference.

---

## Architecture Notes

All identity nodes operate via PyTorch `register_forward_hook` on `transformer_blocks[i].attn1` (or `attn2` for Text Amplifier). Hooks return modified output tensors that flow forward to the next block's input.

**Why hook intervention works:** LTX2's DiT is content-blind to its own intermediate states. Modifying attention output adds a residual that the rest of the block's computation (cross-attention, FFN) integrates naturally. Strength values that look small numerically (0.10-0.20) compound across 48 blocks per step into substantive effects.

**Per-frame centered cosine similarity** is used throughout for matching. Raw cosine similarity is dominated by common-mode features (positional encoding, scaffold features) that all tokens share. Subtracting the per-frame mean leaves identity-specific deviations that discriminate cleanly.

**Cache-and-broadcast** (used by Latent Anchor and the Aware variant) snapshots the model's representation at peak conditioning alignment (mid-sampling) and uses it as a stable pull target. Conceptually adjacent to self-distillation and self-conditioning in diffusion, but operationalized as inference-time intervention rather than training-time signal.

---

## Compatibility & Limitations

- **LTX2-specific.** Hooks rely on `LTXAVModel.transformer_blocks` and `BasicAVTransformerBlock` structure. Will not work on other DiT-based video models without adaptation.
- **Distilled CFG=1 setup tested most extensively.** Standard CFG works but compounds hook calls per step (cond + uncond passes), changing effective strength. Set `forwards_per_step=2` in advanced mode.
- **Hook lifecycle.** Each node clears its own previous hooks before registering new ones. Nodes from this package coexist via distinct sentinel attributes. Bypassing one node doesn't affect others.

---

## Version History

**v1.0.0** — Initial release
- Face Attention Anchor (v4.0)
- Latent Anchor (v1.4)
- Latent Anchor Aware (v2.3)
- Latent Upsampler Tiled (v1.1)
- Latent Color Restore (v1.0)
- Latent Outlier Suppress (v1.0)
- Text Attention Amplifier (v1.1)
- Model Inspector
- Existing latent processors (nodes.py)

---

## License

MIT

## Author

TenStrip · [github.com/TenStrip](https://github.com/TenStrip)
