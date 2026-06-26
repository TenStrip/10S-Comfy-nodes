"""
nodes_echo_dmd_sampler.py
─────────────────────────
ComfyUI custom nodes implementing Echo's DMD inference schedule and
a faithful deterministic euler sampler for JoyAI-Echo / LTX-2.3 DMD models.

Two nodes:
  • EchoDMDSigmas   — outputs a SIGMAS tensor from one of three presets or
                      a fully custom comma-separated list.  Handles the
                      two-zone structure (init cluster + DMD anchor points).
  • EchoDMDSampler  — outputs a SAMPLER that runs a pure deterministic euler
                      loop (no ancestral noise injection).  Optionally skips
                      the uncond pass (cfg=1.0 fast path) and supports
                      sigma-threshold gating of the init zone.

Add to your NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS as usual.
"""

from __future__ import annotations

import torch
import comfy.samplers
import comfy.sample
import comfy.model_patcher


# ─────────────────────────────────────────────────────────────────────────────
# Sigma presets
# ─────────────────────────────────────────────────────────────────────────────

# Official JoyAI-Echo DMD schedule (9 values → 8 steps)
# Two-zone: 5 tight init steps (σ 1.0→0.975) + 4 large anchor steps
SIGMAS_OFFICIAL = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0
]

# Bridge variant: drops one init step, adds midpoint at 0.817 between
# the 0.909→0.725 gap — the largest and most productive gap to subdivide.
# DMD anchor points (0.975, 0.909375, 0.725, 0.421875) are untouched.
SIGMAS_BRIDGE = [
    1.0, 0.99375, 0.9875, 0.975,
    0.909375, 0.817, 0.725, 0.421875, 0.0
]

# Minimal variant: collapses two init steps, adds bridge — least disruptive
# net change while gaining one productive mid-range step.
SIGMAS_MINIMAL = [
    1.0, 0.990, 0.981, 0.975,
    0.909375, 0.817, 0.725, 0.421875, 0.0
]

PRESET_MAP = {
    "official":  SIGMAS_OFFICIAL,
    "bridge":    SIGMAS_BRIDGE,
    "minimal":   SIGMAS_MINIMAL,
}

ANCHOR_SIGMAS = {0.975, 0.909375, 0.725, 0.421875, 0.0}


def parse_sigma_string(s: str) -> list[float]:
    """Parse a comma-separated sigma string into a sorted-descending list."""
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty sigma list")
    vals = sorted(set(vals), reverse=True)
    if vals[-1] != 0.0:
        vals.append(0.0)
    return vals


# ─────────────────────────────────────────────────────────────────────────────
# EchoDMDSigmas node
# ─────────────────────────────────────────────────────────────────────────────

class EchoDMDSigmas:
    """
    Outputs a SIGMAS tensor for use with SamplerCustomAdvanced.

    Presets
    ───────
    official  The exact JoyAI-Echo DMD schedule — 5 tight init steps then
              4 large anchor steps.  Faithful to the DMD training regime.

    bridge    One init step freed up, placed as a midpoint at σ=0.817
              between the 0.909→0.725 anchor gap.  All anchor sigmas
              untouched.  Gives the model one extra productive step.

    minimal   Two init steps collapsed, one bridge added.  Most different
              from official while still preserving all DMD anchors.

    custom    Comma-separated sigma list.  DMD anchor values
              (0.975, 0.909375, 0.725, 0.421875) should be included
              to preserve distillation quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    ["official", "bridge", "minimal", "custom"],
                    {"default": "official"},
                ),
                "custom_sigmas": (
                    "STRING",
                    {
                        "default": "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                                   "0.909375, 0.725, 0.421875, 0.0",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "10S Nodes/Sampling"

    def get_sigmas(self, preset: str, custom_sigmas: str):
        if preset == "custom":
            vals = parse_sigma_string(custom_sigmas)
        else:
            vals = PRESET_MAP[preset]

        # Warn if DMD anchor points are missing
        missing = [s for s in ANCHOR_SIGMAS if s > 0 and s not in vals]
        if missing:
            print(
                f"[EchoDMDSigmas] ⚠  Missing DMD anchor σ values: {missing}. "
                "Omitting anchor points may degrade distillation quality."
            )

        sigmas = torch.tensor(vals, dtype=torch.float32)
        return (sigmas,)


# ─────────────────────────────────────────────────────────────────────────────
# Pure deterministic euler step
# ─────────────────────────────────────────────────────────────────────────────

def _euler_dmd_step(
    x: torch.Tensor,
    sigma: float,
    sigma_next: float,
    denoised: torch.Tensor,
) -> torch.Tensor:
    """
    Rectified-flow euler step — no noise injection.

    In flow-matching terms:
        velocity  v = (x - denoised) / sigma
        x_next      = x + v * (sigma_next - sigma)
                    = (sigma_next / sigma) * x + (1 - sigma_next / sigma) * denoised

    When sigma_next == 0.0 the step collapses to x_next = denoised exactly,
    which is the correct terminal condition.
    """
    if sigma_next == 0.0 or sigma == 0.0:
        return denoised
    t = sigma_next / sigma
    return t * x + (1.0 - t) * denoised


# ─────────────────────────────────────────────────────────────────────────────
# Custom sampler function
# ─────────────────────────────────────────────────────────────────────────────

def _sample_echo_dmd(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    disable: bool = False,
) -> torch.Tensor:
    """
    Deterministic euler sampler matching Echo's DMD inference loop.

    Pure rectified-flow euler step at every sigma transition — no ancestral
    noise injection. The model() call goes through whatever guider is
    upstream of this sampler (CFGGuider, STGGuider, etc.); per-step CFG
    scheduling lives there, not here.

    To skip uncond on init-zone steps (the cheap 5-step σ≈1.0 cluster in
    the official Echo schedule), use the STG Guider node with cfg=1.0 for
    those steps in cfg_per_step (or sigma_curve with cfg_max=1.0). That
    achieves the same compute saving at the right architectural layer.
    """
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        sigma      = sigmas[i].item()
        sigma_next = sigmas[i + 1].item()

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({
                "x":         x,
                "i":         i,
                "sigma":     sigmas[i],
                "sigma_hat": sigmas[i],
                "denoised":  denoised,
            })

        x = _euler_dmd_step(x, sigma, sigma_next, denoised)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# EchoDMDSampler node
# ─────────────────────────────────────────────────────────────────────────────

class EchoDMDSampler:
    """
    SAMPLER node implementing Echo's DMD deterministic euler inference.

    Pure rectified-flow euler — no ancestral noise injection. Matches how
    DMD models are distilled (deterministic trajectory). CFG scheduling
    is upstream's responsibility (use STG Guider or CFGGuider with
    per-step cfg lists).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "10S Nodes/Sampling"

    def get_sampler(self):
        def sampler_fn(model, x, sigmas, extra_args=None, callback=None,
                       disable=False):
            return _sample_echo_dmd(
                model=model,
                x=x,
                sigmas=sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
            )

        sampler = comfy.samplers.KSAMPLER(sampler_fn)
        return (sampler,)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "EchoDMDSigmas":  EchoDMDSigmas,
    "EchoDMDSampler": EchoDMDSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EchoDMDSigmas":  "\U0001f300 Echo DMD Sigmas",
    "EchoDMDSampler": "\U0001f300 Echo DMD Sampler",
}
