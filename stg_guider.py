"""
STG Guider v1.0

Spatio-Temporal Guidance (STG) guider for LTX2, based on Lightricks'
STGGuiderAdvanced but customized for the 10S workflow pattern.

================================================================================
WHAT IS STG
================================================================================

Spatio-Temporal Guidance produces a third "perturbed" model prediction at
each sampling step by SKIPPING attention computation on transformer blocks
(returning v directly instead of softmax(QK)·v). The final noise prediction
combines three predictions:

    noise_pred = pos
               + (cfg - 1) * (pos - neg)             ← standard CFG
               + stg_scale * (pos - perturbed)        ← STG enhancement

The (pos - perturbed) term pushes the result AWAY from the degraded
perturbed prediction — sharpening temporal coherence and structure. STG
costs an extra forward pass per step but tends to improve motion
consistency and frame-to-frame coherence in video DiTs.

================================================================================
DIFFERENCES FROM UPSTREAM STGGuiderAdvanced
================================================================================

1. SIGMAS input wired directly. Upstream took a comma-separated string;
   we wire the SIGMAS tensor (same one driving the sampler) directly.
   This guarantees the per-step parameters align with the actual sampler
   schedule.

2. Block indices exposed as a simple comma-separated input with a sane
   default ('14, 19' matching upstream Lightricks). Perturbing all blocks
   produces a far-too-strong effect (dark output, audio destruction) at
   the same stg_scale calibrated for partial perturbation. Pass '9999'
   (or any out-of-range index) to functionally disable STG perturbation
   while keeping the per-step CFG schedule active. STG perturbation
   targets the FIRST attention call in each selected block — which is
   self-attention (attn1); cross-attention with text is left alone since
   its shape contracts don't permit the V-shortcut substitution.

3. cfg_star_rescale removed. The CFG* rescale heuristic adds a normalization
   step that hasn't shown clear benefit in LTX2 workflows; removing it
   simplifies the node and reduces failure modes.

4. APG / momentum / norm_threshold removed. APG is a separate guidance
   method from STG and conceptually belongs in its own node. The
   experimental APG path in the upstream guider added complexity without
   clear benefit when mixed with STG.

5. Presets removed. The 10S workflow uses explicit per-step parameter
   lists; presets that hide the values lead to confusion when debugging
   variable-CFG schedules.

================================================================================
COMPATIBILITY WITH 10S ANCHOR / CLAMP NODES
================================================================================

This guider is fully compatible with the 10S identity and conditioning
nodes (LikenessAnchor, LikenessSemanticClamp, ActionAmplifier,
AwareAnchor). Those nodes operate via attention hooks INSIDE the model's
forward pass; this guider orchestrates THREE forward passes per step
(positive, negative, perturbed) FROM OUTSIDE the model.

What happens automatically:
  - Positive pass: anchors and clamps fire, modifying attention as usual
  - Negative pass: same anchors/clamps fire on the uncond prediction
  - Perturbed pass: anchors STILL fire (hooks are unconditional), but
    they operate on degraded features since attention has been skipped
    on all blocks for this pass

The anchors' contribution to the perturbed prediction is largely benign —
it pulls the perturbed result slightly toward identity, which doesn't
interfere with the STG perturbation goal of "structurally degraded
reference." If you observe identity drift specifically on STG-enabled
samples, the issue is more likely in anchor configuration than in STG
itself.

================================================================================
USAGE
================================================================================

    Model → [Anchor/Clamp nodes] → STG Guider → SamplerCustomAdvanced
                                          ↑
                                    sigmas, cfg_per_step, stg_per_step

The number of values in cfg_per_step and stg_scale_per_step must equal
the number of sigma transitions (len(sigmas) - 1, since the last sigma
is typically 0.0 representing "end of sampling").

Per-step strings accept comma-separated floats. Examples for a 13-step
schedule:
    cfg_per_step       = "2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"
    stg_scale_per_step = "2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"
    stg_rescale_per_step = "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"

For distilled CFG=1 workflows, set cfg_per_step="1.0, 1.0, ..." and rely
on STG alone for guidance.
"""

import contextlib
import math
from dataclasses import dataclass
from typing import List, Optional, Union

import comfy.ldm.modules.attention
import comfy.samplers
import torch
from comfy.model_patcher import ModelPatcher


# ─────────────────────────────────────────────────────────────────────────────
# Core STG math (unchanged from upstream)
# ─────────────────────────────────────────────────────────────────────────────

def _stg_combine(noise_pred_pos, noise_pred_neg, noise_pred_perturbed,
                 cfg_scale, stg_scale, rescale_scale):
    """Combine the three predictions into the final guided noise prediction.

    formula:
        noise_pred = pos
                   + (cfg - 1) * (pos - neg)
                   + stg_scale * (pos - perturbed)

    rescale_scale (per-step STD-matching to keep magnitudes stable):
        if rescale != 0:
            factor = pos.std() / noise_pred.std()
            factor = rescale * factor + (1 - rescale)
            noise_pred *= factor
    """
    noise_pred = (
        noise_pred_pos
        + (cfg_scale - 1) * (noise_pred_pos - noise_pred_neg)
        + stg_scale * (noise_pred_pos - noise_pred_perturbed)
    )
    if rescale_scale != 0:
        factor = noise_pred_pos.std() / noise_pred.std().clamp(min=1e-8)
        factor = rescale_scale * factor + (1 - rescale_scale)
        noise_pred = noise_pred * factor
    return noise_pred


# ─────────────────────────────────────────────────────────────────────────────
# Attention skipping for the perturbed pass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _STGFlag:
    do_skip: bool = False
    target_block_indices: Optional[List[int]] = None


class _PatchAttention(contextlib.AbstractContextManager):
    """Context manager that replaces optimized_attention with a no-op
    (returns V directly, skipping QK) for the FIRST attention call inside
    the wrapped scope — which is self-attention (attn1) when used inside
    a block's forward pass.

    Why first-only: STG perturbation is defined for spatio-temporal SELF-
    attention. Cross-attention (text→video) has different shape contracts
    in LTX2's _attention_with_guide_mask and adaLN paths — skipping it
    causes shape mismatches when the model uses guide masks or gates that
    expect specific output shapes. Self-attention is well-behaved under
    this replacement because Q, K, V all share the same sequence length.

    Each context-manager instance tracks its own "first call done" flag,
    so it's safe to construct a fresh _PatchAttention per block.

    Class-level diagnostic counters (off by default) record:
      - skipped: V-shortcut applied (intended self-attention path)
      - shape_blocked: would-be first call had shape mismatch → fell through
      - subsequent: second+ attention call in block → fell through (cross-attn)
      - shape_skipped_seqs: list of (q.shape[1], k.shape[1], v.shape[1])
        for the shape-blocked cases — to spot if cross-attn ever masquerades
        as self-attn shape-wise
      - subsequent_seqs: same for the subsequent (post-first) calls
    """

    diag_enabled = False
    diag_skipped = 0
    diag_shape_blocked = 0
    diag_subsequent = 0
    diag_shape_skipped_seqs = []
    diag_subsequent_seqs = []
    DIAG_SEQ_SAMPLES_MAX = 50  # cap the per-shape lists so we don't blow memory

    @classmethod
    def reset_diag(cls):
        cls.diag_skipped = 0
        cls.diag_shape_blocked = 0
        cls.diag_subsequent = 0
        cls.diag_shape_skipped_seqs = []
        cls.diag_subsequent_seqs = []

    @classmethod
    def report_diag(cls, prefix: str = "  ·"):
        total = (cls.diag_skipped + cls.diag_shape_blocked
                 + cls.diag_subsequent)
        if total == 0:
            return
        print(f"{prefix} [STG diag] attention calls during perturbed pass:")
        print(f"{prefix}   skipped (V-shortcut applied, self-attn): "
              f"{cls.diag_skipped}")
        print(f"{prefix}   shape_blocked (first call but Q≠K≠V seq, "
              f"fell through): {cls.diag_shape_blocked}")
        print(f"{prefix}   subsequent (2nd+ call in block, fell through): "
              f"{cls.diag_subsequent}")
        # Unique shape signatures observed in shape_blocked and subsequent
        if cls.diag_shape_skipped_seqs:
            uniq = sorted(set(cls.diag_shape_skipped_seqs))
            print(f"{prefix}   shape_blocked unique (q,k,v) seqs: {uniq}")
        if cls.diag_subsequent_seqs:
            uniq = sorted(set(cls.diag_subsequent_seqs))
            print(f"{prefix}   subsequent unique (q,k,v) seqs: {uniq}")

    def __init__(self):
        self._first_call_done = False

    def __enter__(self):
        self.original_attention = comfy.ldm.modules.attention.optimized_attention
        self.original_attention_masked = (
            comfy.ldm.modules.attention.optimized_attention_masked
        )

        original_attn = self.original_attention
        original_attn_masked = self.original_attention_masked
        outer = self
        cls = type(self)

        def patched(q, k, v, heads, *args, **kwargs):
            if not outer._first_call_done:
                outer._first_call_done = True
                same_seq = (q.shape[1] == k.shape[1] == v.shape[1])
                if cls.diag_enabled:
                    if same_seq:
                        cls.diag_skipped += 1
                    else:
                        cls.diag_shape_blocked += 1
                        if len(cls.diag_shape_skipped_seqs) < \
                                cls.DIAG_SEQ_SAMPLES_MAX:
                            cls.diag_shape_skipped_seqs.append(
                                (int(q.shape[1]), int(k.shape[1]),
                                 int(v.shape[1]))
                            )
                if same_seq:
                    return v
            else:
                if cls.diag_enabled:
                    cls.diag_subsequent += 1
                    if len(cls.diag_subsequent_seqs) < \
                            cls.DIAG_SEQ_SAMPLES_MAX:
                        cls.diag_subsequent_seqs.append(
                            (int(q.shape[1]), int(k.shape[1]),
                             int(v.shape[1]))
                        )
            return original_attn(q, k, v, heads, *args, **kwargs)

        def patched_masked(q, k, v, heads, *args, **kwargs):
            if not outer._first_call_done:
                outer._first_call_done = True
                same_seq = (q.shape[1] == k.shape[1] == v.shape[1])
                if cls.diag_enabled:
                    if same_seq:
                        cls.diag_skipped += 1
                    else:
                        cls.diag_shape_blocked += 1
                        if len(cls.diag_shape_skipped_seqs) < \
                                cls.DIAG_SEQ_SAMPLES_MAX:
                            cls.diag_shape_skipped_seqs.append(
                                (int(q.shape[1]), int(k.shape[1]),
                                 int(v.shape[1]))
                            )
                if same_seq:
                    return v
            else:
                if cls.diag_enabled:
                    cls.diag_subsequent += 1
                    if len(cls.diag_subsequent_seqs) < \
                            cls.DIAG_SEQ_SAMPLES_MAX:
                        cls.diag_subsequent_seqs.append(
                            (int(q.shape[1]), int(k.shape[1]),
                             int(v.shape[1]))
                        )
            return original_attn_masked(q, k, v, heads, *args, **kwargs)

        comfy.ldm.modules.attention.optimized_attention = patched
        comfy.ldm.modules.attention.optimized_attention_masked = patched_masked

    def __exit__(self, exc_type, exc_value, traceback):
        comfy.ldm.modules.attention.optimized_attention = self.original_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.original_attention_masked
        )
        self.original_attention = None
        self.original_attention_masked = None


class _STGBlockWrapper:
    """Wraps a transformer block. During the perturbed pass, applies
    attention skipping ONLY if this block's index is in the target list.
    Blocks outside the target list run normally."""

    def __init__(self, block, stg_flag: _STGFlag, block_idx: int):
        self.flag = stg_flag
        self.block = block
        self.block_idx = block_idx

    def __call__(self, args, extra_args):
        context_manager = contextlib.nullcontext()
        if self.flag.do_skip:
            target = self.flag.target_block_indices
            # If no target list set, default to all blocks (legacy behavior;
            # rarely what you want). Otherwise only the listed blocks get
            # perturbed.
            if target is None or self.block_idx in target:
                context_manager = _PatchAttention()
        with context_manager:
            hidden_state = extra_args["original_block"](args)
        return hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# The guider
# ─────────────────────────────────────────────────────────────────────────────

class _LTX2STGGuider(comfy.samplers.CFGGuider):
    """
    STG guider customized for 10S workflows:
      - sigmas wired directly from SIGMAS input
      - attention always skipped on ALL blocks during perturbed pass
      - no cfg_star_rescale, no APG, no presets
    """

    def __init__(self, model: ModelPatcher,
                 sigmas: torch.Tensor,
                 cfg_list: List[float],
                 stg_scale_list: List[float],
                 stg_rescale_list: List[float],
                 block_indices: Optional[List[int]] = None,
                 cfg_mode: str = "per_step_list",
                 cfg_min: float = 1.0,
                 cfg_max: float = 1.0,
                 stg_mode: str = "per_step_list",
                 stg_min: float = 0.0,
                 stg_max: float = 0.0,
                 stg_rescale_mode: str = "per_step_list",
                 stg_rescale_min: float = 1.0,
                 stg_rescale_max: float = 1.0,
                 skip_steps_sigma_threshold: float = 1.0,
                 debug: bool = False):
        model = model.clone()
        super().__init__(model)

        if block_indices is None:
            block_indices = [14, 19]

        self.stg_flag = _STGFlag(
            do_skip=False,
            target_block_indices=list(block_indices),
        )
        self._patch_model(model, self.stg_flag)

        # Convert sigmas tensor to list of floats; align with per-step params
        if isinstance(sigmas, torch.Tensor):
            self.sigma_list = [float(s.item()) for s in sigmas]
        else:
            self.sigma_list = list(sigmas)

        # Compute sigma_max for curve interpolation (largest sigma in schedule).
        # Typically sigma_list[0] = 1.0 for LTX2.
        self.sigma_max = max(self.sigma_list) if self.sigma_list else 1.0

        self.cfg_list = list(cfg_list)
        self.stg_scale_list = list(stg_scale_list)
        self.stg_rescale_list = list(stg_rescale_list)

        # Curve mode configuration
        self.cfg_mode = cfg_mode
        self.cfg_min = float(cfg_min)
        self.cfg_max = float(cfg_max)
        self.stg_mode = stg_mode
        self.stg_min = float(stg_min)
        self.stg_max = float(stg_max)
        self.stg_rescale_mode = stg_rescale_mode
        self.stg_rescale_min = float(stg_rescale_min)
        self.stg_rescale_max = float(stg_rescale_max)

        self.skip_steps_sigma_threshold = float(skip_steps_sigma_threshold)
        self.debug = bool(debug)

        # Validate list lengths align with sigma transitions (only matters
        # for signals using per_step_list mode)
        n_transitions = len(self.sigma_list)
        for name, lst, mode in (
            ("cfg_per_step", self.cfg_list, self.cfg_mode),
            ("stg_scale_per_step", self.stg_scale_list, self.stg_mode),
            ("stg_rescale_per_step", self.stg_rescale_list, self.stg_rescale_mode),
        ):
            if mode != "per_step_list":
                continue
            if len(lst) < n_transitions:
                last_val = lst[-1] if lst else 1.0
                pad_count = n_transitions - len(lst)
                lst.extend([last_val] * pad_count)
                if debug:
                    print(f"→ [STG] {name}: padded {pad_count} entries "
                          f"with {last_val} to match sigma count")
            elif len(lst) > n_transitions:
                if debug:
                    print(f"→ [STG] {name}: trimmed to first "
                          f"{n_transitions} entries")
                del lst[n_transitions:]

        if debug:
            print(f"→ [STG] Guider initialized:")
            print(f"  · sigmas:          {self.sigma_list}")
            print(f"  · sigma_max:       {self.sigma_max}")
            if self.cfg_mode == "sigma_curve":
                print(f"  · cfg mode:        sigma_curve "
                      f"(min={self.cfg_min}, max={self.cfg_max})")
            else:
                print(f"  · cfg:             {self.cfg_list}")
            if self.stg_mode == "sigma_curve":
                print(f"  · stg_scale mode:  sigma_curve "
                      f"(min={self.stg_min}, max={self.stg_max})")
            else:
                print(f"  · stg_scale:       {self.stg_scale_list}")
            if self.stg_rescale_mode == "sigma_curve":
                print(f"  · stg_rescale mode: sigma_curve "
                      f"(min={self.stg_rescale_min}, max={self.stg_rescale_max})")
            else:
                print(f"  · stg_rescale:     {self.stg_rescale_list}")
            print(f"  · skip_threshold:  {self.skip_steps_sigma_threshold}")
            print(f"  · block_indices:   {self.stg_flag.target_block_indices} "
                  f"(blocks where attn1 gets V-shortcut during perturbed pass)")

    @classmethod
    def _patch_model(cls, model: ModelPatcher, stg_flag: _STGFlag):
        """Wrap every transformer block with the STG block wrapper. Each
        wrapper knows its own block index; the flag's target list decides
        which blocks actually get perturbed at runtime."""
        transformer_blocks = cls._get_transformer_blocks(model)
        for i, block in enumerate(transformer_blocks):
            model.set_model_patch_replace(
                _STGBlockWrapper(block, stg_flag, i),
                "dit", "double_block", i
            )

    @staticmethod
    def _get_transformer_blocks(model: ModelPatcher):
        diffusion_model = model.get_model_object("diffusion_model")
        key = "diffusion_model.transformer_blocks"
        # Handle wrapped LTXV models
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            key = "diffusion_model.transformer.transformer_blocks"
        return model.get_model_object(key)

    def _interp_sigma_curve(self, sigma: float, vmin: float, vmax: float) -> float:
        """Linear interpolation in sigma space:
            value(sigma) = vmin + (vmax - vmin) * (sigma / sigma_max)
        At sigma = sigma_max: value = vmax (start of sampling).
        At sigma = 0:         value = vmin (end of sampling).
        Linear in sigma — NOT linear in step index. Because LTX2 schedules
        concentrate sigmas non-uniformly, this differs from a per-step list
        with the same min/max bounds (e.g., at step 6 of 13, sigma is still
        ~0.81, so value is still ~81% of the way from vmin to vmax).
        """
        if self.sigma_max <= 0:
            return vmin
        t = max(0.0, min(1.0, sigma / self.sigma_max))
        return vmin + (vmax - vmin) * t

    def _index_for_sigma(self, sigma: float) -> int:
        """Find the per-step list index for the given sigma (used only when
        any signal is in per_step_list mode)."""
        higher = [s for s in self.sigma_list if s >= sigma]
        if not higher:
            idx = len(self.sigma_list) - 1
        else:
            closest_high = min(higher)
            idx = self.sigma_list.index(closest_high)
        return max(0, idx)

    def _params_for_sigma(self, sigma: float):
        """Given the current sigma, return (cfg, stg_scale, stg_rescale).
        Each signal uses its own mode — per-step list or sigma curve."""

        # Compute list-based values (only used if a signal is in list mode)
        idx = self._index_for_sigma(sigma)

        if self.cfg_mode == "sigma_curve":
            cfg_value = self._interp_sigma_curve(sigma, self.cfg_min, self.cfg_max)
        else:
            cfg_value = self.cfg_list[min(idx, len(self.cfg_list) - 1)]

        if self.stg_mode == "sigma_curve":
            stg_value = self._interp_sigma_curve(sigma, self.stg_min, self.stg_max)
        else:
            stg_value = self.stg_scale_list[min(idx, len(self.stg_scale_list) - 1)]

        if self.stg_rescale_mode == "sigma_curve":
            rescale_value = self._interp_sigma_curve(
                sigma, self.stg_rescale_min, self.stg_rescale_max
            )
        else:
            rescale_value = self.stg_rescale_list[
                min(idx, len(self.stg_rescale_list) - 1)
            ]

        return (cfg_value, stg_value, rescale_value)

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(self, x: torch.Tensor, timestep: torch.Tensor,
                      model_options: dict = {}, seed=None):
        # Early-skip: CFG zero init for very high-sigma steps
        cur_sigma = float(timestep.max().item() if timestep.numel() > 1
                          else timestep.item())
        if cur_sigma > self.skip_steps_sigma_threshold:
            return torch.zeros_like(x)

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        cfg_value, stg_scale, stg_rescale = self._params_for_sigma(cur_sigma)

        if self.debug:
            print(f"  · [STG] sigma={cur_sigma:.4f} → "
                  f"cfg={cfg_value:.2f} stg={stg_scale:.2f} "
                  f"rescale={stg_rescale:.2f}")

        # ─── Pass 1: positive ───────────────────────────────────────────────
        noise_pred_pos = comfy.samplers.calc_cond_batch(
            self.inner_model, [positive_cond], x, timestep, model_options,
        )[0]

        # ─── Pass 2: negative (only when cfg != 1.0) ────────────────────────
        noise_pred_neg = 0
        if not math.isclose(cfg_value, 1.0):
            noise_pred_neg = comfy.samplers.calc_cond_batch(
                self.inner_model, [negative_cond], x, timestep, model_options,
            )[0]

        # ─── Pass 3: perturbed (only when stg != 0) ─────────────────────────
        noise_pred_perturbed = 0
        if not math.isclose(stg_scale, 0.0):
            # Enable diagnostics for the first perturbed step only — this
            # gives a single representative readout without flooding the
            # console across all sampling steps.
            first_perturbed = (not getattr(self, "_diag_reported", False))
            if self.debug and first_perturbed:
                _PatchAttention.diag_enabled = True
                _PatchAttention.reset_diag()
            try:
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(
                    self.inner_model, [positive_cond], x, timestep, model_options,
                )[0]
            finally:
                self.stg_flag.do_skip = False
                if self.debug and first_perturbed:
                    _PatchAttention.diag_enabled = False
                    print(f"  · [STG diag] first perturbed pass diagnostics "
                          f"at sigma={cur_sigma:.4f}, "
                          f"target blocks="
                          f"{self.stg_flag.target_block_indices}:")
                    _PatchAttention.report_diag(prefix="    ")
                    self._diag_reported = True

        # ─── Combine ────────────────────────────────────────────────────────
        result = _stg_combine(
            noise_pred_pos, noise_pred_neg, noise_pred_perturbed,
            cfg_value, stg_scale, stg_rescale,
        )

        # Replicate sampler_post_cfg_function hooks (LoRA / detail fns / etc.)
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                "perturbed_cond": positive_cond,
                "perturbed_cond_denoised": noise_pred_perturbed,
            }
            result = fn(args)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class LTX2STGGuider:
    """ComfyUI node wrapping the STG guider."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX2 model. Chain after any anchor/clamp "
                               "nodes — those operate INSIDE the model's "
                               "forward; this guider orchestrates three "
                               "forwards from outside.",
                }),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS", {
                    "tooltip": "Sigma schedule. Same SIGMAS that drive your "
                               "sampler. Per-step parameter lists must "
                               "align with this length.",
                }),
                "cfg_per_step": ("STRING", {
                    "default": "2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                    "multiline": False,
                    "tooltip": "Comma-separated CFG values per sampling "
                               "step. Length must match number of sigmas "
                               "(typically steps + 1, since last sigma is "
                               "0.0). For distilled CFG=1, set all to 1.0.",
                }),
                "stg_scale_per_step": ("STRING", {
                    "default": "2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                    "multiline": False,
                    "tooltip": "Comma-separated STG scale values per step. "
                               "0.0 disables STG for that step (skips the "
                               "perturbed forward). Typical: 2.0 early, "
                               "ramping to 1.0 for the bulk of sampling.",
                }),
                "stg_rescale_per_step": ("STRING", {
                    "default": "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                    "multiline": False,
                    "tooltip": "Per-step STD-rescaling factor (0.0 = no "
                               "rescale; 1.0 = full rescale to match "
                               "positive prediction's standard deviation). "
                               "Typically 1.0 throughout.",
                }),
            },
            "optional": {
                "cfg_mode": (["per_step_list", "sigma_curve"], {
                    "default": "per_step_list",
                    "tooltip": "How CFG values are determined per step.\n"
                               "  per_step_list: parse cfg_per_step "
                               "(default, full manual control).\n"
                               "  sigma_curve: linear interpolation in "
                               "sigma-space between cfg_min and cfg_max — "
                               "cfg_per_step is IGNORED.",
                }),
                "cfg_min": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.05,
                    "tooltip": "CFG value when sigma reaches 0 (end of "
                               "sampling). Only used when cfg_mode="
                               "sigma_curve. 1.0 = no CFG at the end.",
                }),
                "cfg_max": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 30.0, "step": 0.05,
                    "tooltip": "CFG value when sigma is at schedule max "
                               "(start of sampling). Only used when "
                               "cfg_mode=sigma_curve. The curve interpolates "
                               "linearly in sigma between cfg_max (start) "
                               "and cfg_min (end). Note: linear in sigma, "
                               "NOT in step index — LTX2 sigma schedules "
                               "concentrate non-uniformly, so the curve is "
                               "intentionally steeper at the start.",
                }),
                "stg_mode": (["per_step_list", "sigma_curve"], {
                    "default": "per_step_list",
                    "tooltip": "How STG scale values are determined.\n"
                               "  per_step_list: parse stg_scale_per_step.\n"
                               "  sigma_curve: linear interpolation between "
                               "stg_min and stg_max — stg_scale_per_step "
                               "is IGNORED.",
                }),
                "stg_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.05,
                    "tooltip": "STG scale at sigma=0. Only used when "
                               "stg_mode=sigma_curve. 0.0 = no STG at end.",
                }),
                "stg_max": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 30.0, "step": 0.05,
                    "tooltip": "STG scale at schedule's max sigma. Only "
                               "used when stg_mode=sigma_curve.",
                }),
                "stg_rescale_mode": (["per_step_list", "sigma_curve"], {
                    "default": "per_step_list",
                    "tooltip": "How STG rescale values are determined.",
                }),
                "stg_rescale_min": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "STG rescale at sigma=0. Only used when "
                               "stg_rescale_mode=sigma_curve.",
                }),
                "stg_rescale_max": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "STG rescale at schedule's max sigma.",
                }),
                "block_indices": ("STRING", {
                    "default": "14, 19",
                    "multiline": False,
                    "tooltip": "Comma-separated indices of transformer "
                               "blocks where self-attention will be "
                               "V-shortcut (perturbed) during the STG pass. "
                               "Default '14, 19' matches upstream Lightricks "
                               "— two mid-depth blocks produce a modest, "
                               "well-calibrated perturbation. Adding more "
                               "blocks dramatically strengthens STG; you "
                               "would need to reduce stg_scale "
                               "proportionally. Set '9999' (any out-of-"
                               "range index) to functionally disable STG "
                               "perturbation while keeping the per-step "
                               "CFG schedule active.",
                }),
                "skip_steps_sigma_threshold": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Sigmas above this threshold get a zero "
                               "noise prediction (CFG zero-init). 1.0 "
                               "default = no skipping (sigmas typically "
                               "max at 1.0). Lower to skip very-noisy "
                               "early steps.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "10S Nodes/Sampling"
    DESCRIPTION = (
        "STG (Spatio-Temporal Guidance) guider for LTX2. Perturbs self-"
        "attention on selected transformer blocks during a third forward "
        "pass per step, combining the result via "
        "pred = pos + (cfg-1)*(pos-neg) + stg_scale*(pos-perturbed). "
        "Each of cfg, stg_scale, stg_rescale can use either a per-step "
        "list (full control) or a sigma_curve mode (linear interpolation "
        "between min/max bounds — set once and adapts to any sigma schedule)."
    )

    def get_guider(self, model, positive, negative, sigmas,
                   cfg_per_step, stg_scale_per_step, stg_rescale_per_step,
                   cfg_mode="per_step_list", cfg_min=1.0, cfg_max=3.0,
                   stg_mode="per_step_list", stg_min=0.0, stg_max=2.0,
                   stg_rescale_mode="per_step_list",
                   stg_rescale_min=1.0, stg_rescale_max=1.0,
                   block_indices="14, 19",
                   skip_steps_sigma_threshold=1.0, debug=False):

        def _parse_floats(s, name):
            try:
                return [float(x.strip()) for x in s.split(",") if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"STG Guider: couldn't parse {name}: '{s}' "
                    f"({type(e).__name__}: {e})"
                )

        def _parse_ints(s, name):
            try:
                return [int(x.strip()) for x in s.split(",") if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"STG Guider: couldn't parse {name}: '{s}' "
                    f"({type(e).__name__}: {e})"
                )

        cfg_list = _parse_floats(cfg_per_step, "cfg_per_step")
        stg_scale_list = _parse_floats(stg_scale_per_step, "stg_scale_per_step")
        stg_rescale_list = _parse_floats(stg_rescale_per_step, "stg_rescale_per_step")
        block_idx_list = _parse_ints(block_indices, "block_indices")

        # Only validate lists for signals using list mode (curve mode
        # ignores the lists). Empty list is fine if the corresponding mode
        # is sigma_curve.
        if cfg_mode == "per_step_list" and not cfg_list:
            raise ValueError(
                "STG Guider: cfg_per_step cannot be empty when "
                "cfg_mode=per_step_list."
            )
        if stg_mode == "per_step_list" and not stg_scale_list:
            raise ValueError(
                "STG Guider: stg_scale_per_step cannot be empty when "
                "stg_mode=per_step_list."
            )
        if stg_rescale_mode == "per_step_list" and not stg_rescale_list:
            raise ValueError(
                "STG Guider: stg_rescale_per_step cannot be empty when "
                "stg_rescale_mode=per_step_list."
            )
        if not block_idx_list:
            raise ValueError("STG Guider: block_indices cannot be empty. "
                             "Use '9999' to disable STG perturbation.")

        # Provide dummy values for unused-mode lists so __init__ doesn't trip
        if not cfg_list:
            cfg_list = [1.0]
        if not stg_scale_list:
            stg_scale_list = [0.0]
        if not stg_rescale_list:
            stg_rescale_list = [1.0]

        guider = _LTX2STGGuider(
            model, sigmas,
            cfg_list, stg_scale_list, stg_rescale_list,
            block_indices=block_idx_list,
            cfg_mode=cfg_mode, cfg_min=cfg_min, cfg_max=cfg_max,
            stg_mode=stg_mode, stg_min=stg_min, stg_max=stg_max,
            stg_rescale_mode=stg_rescale_mode,
            stg_rescale_min=stg_rescale_min,
            stg_rescale_max=stg_rescale_max,
            skip_steps_sigma_threshold=skip_steps_sigma_threshold,
            debug=debug,
        )
        guider.set_conds(positive, negative)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "LTX2STGGuider": LTX2STGGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2STGGuider": "\U0001f3af STG Guider",
}
