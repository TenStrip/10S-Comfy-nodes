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
                      loop (no ancestral noise injection).  Supports sigma
                      remapping to keep model conditioning on trained anchor
                      values when using extended refinement schedules.

Add to your NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS as usual.
"""

from __future__ import annotations
from tqdm.auto import tqdm

import torch
import comfy.samplers
import comfy.sample
import comfy.model_patcher
import comfy.utils
import time


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


# Extended refinement presets — target the two large low-sigma gaps
# (0.725→0.422 and 0.422→0.0) where fine-motion ghosting originates.
# All DMD anchor points are preserved in every variant.

# 10 steps: minimal fix — one substep before terminal only.
# Splits the 0.422→0.0 gap. Lowest risk, try first.
SIGMAS_10STEP = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.14, 0.0
]

# 11 steps: drops one init step, adds 0.573 in the coarse-detail gap
# and 0.21 before terminal. Addresses both large low-sigma gaps.
SIGMAS_11STEP = [
    1.0, 0.99375, 0.9875, 0.975,
    0.909375, 0.725, 0.573, 0.421875, 0.21, 0.0
]

# 12 steps: bridge + full low-sigma coverage. Most complete refinement.
# Adds 0.817 (bridge), 0.573 (mid coarse-detail), 0.21 and 0.07
# (fine-detail substeps). Recommended if 10/11 step doesn't clear ghosting.
SIGMAS_12STEP = [
    1.0, 0.99375, 0.975,
    0.909375, 0.817, 0.725, 0.573, 0.421875, 0.21, 0.07, 0.0
]

PRESET_MAP = {
    "official": SIGMAS_OFFICIAL,
    "bridge":   SIGMAS_BRIDGE,
    "minimal":  SIGMAS_MINIMAL,
    "10step":   SIGMAS_10STEP,
    "11step":   SIGMAS_11STEP,
    "12step":   SIGMAS_12STEP,
}

ANCHOR_SIGMAS = {0.975, 0.909375, 0.725, 0.421875, 0.0}

# All 9 official DMD anchor sigmas as a tensor — used for sigma remapping.
# The model's adaln timestep embedder was trained on exactly these values.
# When using non-standard sigma schedules, remap_sigma() maps arbitrary
# denoising sigmas to these anchors for model conditioning, while the
# euler step still uses the actual sigma value.
DMD_ANCHORS = torch.tensor(
    [1.0, 0.99375, 0.9875, 0.98125, 0.975,
     0.909375, 0.725, 0.421875, 0.0],
    dtype=torch.float32,
)


def remap_sigma(sigma: float, method: str) -> float:
    """
    Map a denoising sigma to an effective sigma for model timestep conditioning.

    Decouples the euler-step sigma from the conditioning sigma so non-official
    schedules (bridge/minimal/10step/11step/12step/custom) still receive
    well-formed timestep embeddings the model was trained on.

    none          Pass sigma through unchanged.  Use with the official preset
                  or when testing schedules very close to the anchor values.

    interpolate   Linear interpolation between the two bounding DMD anchors.
                  Smooth embeddings.  Recommended for all extended presets.

    nearest       Hard snap to the closest DMD anchor.  Guarantees a seen
                  training embedding but can produce slight staircase artefacts
                  at anchor boundaries.
    """
    if method == "none":
        return sigma

    anchors = DMD_ANCHORS
    if method == "nearest":
        idx = int((anchors - sigma).abs().argmin().item())
        return float(anchors[idx].item())

    # interpolate: find the two anchors bracketing sigma
    above = anchors[anchors >= sigma]
    below = anchors[anchors <= sigma]
    if len(above) == 0:
        return float(anchors[0].item())
    if len(below) == 0:
        return float(anchors[-1].item())
    s_hi = float(above[-1].item())
    s_lo = float(below[0].item())
    if s_hi == s_lo:
        return s_hi
    t = (sigma - s_lo) / (s_hi - s_lo)
    return s_lo + t * (s_hi - s_lo)


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
                    ["official", "bridge", "minimal", "10step", "11step", "12step", "custom"],
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

    steps = len(sigmas) - 1
    pbar = comfy.utils.ProgressBar(steps)
    pbar_tqdm = tqdm(total=steps, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for i in range(steps):
        sigma      = sigmas[i].item()
        sigma_next = sigmas[i + 1].item()

        t0 = time.perf_counter()
        denoised = model(x, sigma * s_in, **extra_args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if callback is not None:
            callback({
                "x":         x,
                "i":         i,
                "sigma":     sigmas[i],
                "sigma_hat": sigmas[i],
                "denoised":  denoised,
            })

        x = _euler_dmd_step(x, sigma, sigma_next, denoised)
        pbar_tqdm.update(1)
        if elapsed >= 1.0:
            pbar_tqdm.set_postfix({"s/it": f"{elapsed:.2f}"})
        else:
            pbar_tqdm.set_postfix({"it/s": f"{1/elapsed:.2f}"})
        pbar.update(1)

    pbar_tqdm.close()
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
# EchoDMDSigmaRemap node
# ─────────────────────────────────────────────────────────────────────────────

class EchoDMDSigmaRemap:
    """
    Remaps a SIGMAS tensor so every value is expressed in terms of the
    DMD anchor sigma space before it reaches either the guider or the sampler.

    Why this is a separate node:
        The STG guider and the sampler both receive the same SIGMAS tensor.
        If remapping happened inside the sampler, the guider would see the
        raw schedule (e.g. 0.817) while the sampler conditions on the
        interpolated anchor value — a mismatch that misaligns CFG weighting
        with denoising conditioning.  Remapping upstream ensures both nodes
        operate on consistent effective timesteps.

    Modes
    ─────
    interpolate   Each sigma is linearly interpolated between its two
                  bounding DMD anchor values.  Smooth conditioning.
                  Recommended for bridge / minimal / extended presets.

    nearest       Each sigma snaps to the closest DMD anchor.  Hard
                  quantisation — guarantees a seen training embedding
                  but can introduce slight staircase artefacts.

    none          Pass sigmas through unchanged.  Use with the official
                  preset or when sigma values are already at DMD anchors.

    The remapped tensor is used for model conditioning only.  If you need
    the original values for reference (e.g. logging), keep the pre-remap
    SIGMAS connected separately.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "method": (
                    ["interpolate", "nearest", "none"],
                    {
                        "default": "interpolate",
                        "tooltip": (
                            "interpolate: smooth linear blend between bounding "
                            "DMD anchors.  nearest: snap to closest anchor.  "
                            "none: pass through unchanged."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES  = ("SIGMAS",)
    RETURN_NAMES  = ("sigmas",)
    FUNCTION      = "remap"
    CATEGORY      = "10S Nodes/Sampling"

    def remap(self, sigmas: torch.Tensor, method: str):
        if method == "none":
            return (sigmas,)
        remapped = torch.tensor(
            [remap_sigma(float(s.item()), method) for s in sigmas],
            dtype=sigmas.dtype,
            device=sigmas.device,
        )
        return (remapped,)

# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "EchoDMDSigmas":      EchoDMDSigmas,
    "EchoDMDSigmaRemap":  EchoDMDSigmaRemap,
    "EchoDMDSampler":     EchoDMDSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EchoDMDSigmas":      "\U0001f300 Echo DMD Sigmas",
    "EchoDMDSigmaRemap":  "\U0001f300 Echo DMD Sigma Remap",
    "EchoDMDSampler":     "\U0001f300 Echo DMD Sampler",
}
