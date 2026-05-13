import torch
import torch.nn.functional as nnf
from comfy import nested_tensor

# ─────────────────────────────────────────────────────────────────────────────
# Recommended fast-motion workflow (24 fps sample → 30 fps output):
#
#   [KSampler @ 24 fps]
#       └─► [LatentMotionSharpener]
#               └─► [LatentTemporalUpsampler  auto_retime=True]
#                       └─► [LatentTemporalInpainter]
#                                   └─► [KSampler  denoise=0.30–0.45]
#                                               └─► [Upscale / Decode]
#
# LatentMotionRetime is an alternative to auto_retime=True for post-correction
# on an already-upsampled latent.
# ─────────────────────────────────────────────────────────────────────────────


def unwrap_tensor(x):
    if isinstance(x, nested_tensor.NestedTensor):
        return x.tensors[0] if x.tensors else torch.zeros(1, 8, 1)
    return x


def _dict_with(base, **updates):
    out = {k: v for k, v in base.items()}
    out.update(updates)
    return out


def safe_return(samples):
    if samples is None:
        samples = torch.zeros(1, 8, 1)
    if isinstance(samples, dict):
        return (samples,)
    return ({"samples": samples},)


def ltx_safe_frames(n):
    if n <= 1:
        return 1
    return int(round((n - 1) / 8)) * 8 + 1


def _velocities(t):
    F    = t.shape[2]
    prev = torch.cat([t.narrow(2, 0, 1),     t.narrow(2, 0, F - 1)  ], dim=2)
    nxt  = torch.cat([t.narrow(2, 1, F - 1), t.narrow(2, F - 1, 1)  ], dim=2)
    return (nxt - prev) * 0.5


def _stretch_temporal(samples, target_F, interp_mode="hermite", motion_scale=1.0):
    src_F   = samples.shape[2]
    tail    = samples.dim() - 3
    src_pos = torch.linspace(0, src_F - 1, target_F,
                             device=samples.device, dtype=samples.dtype)
    t0    = src_pos.floor().long().clamp(0, src_F - 1)
    t1    = (t0 + 1).clamp(0, src_F - 1)
    alpha = src_pos - t0.float()
    f0 = samples[:, :, t0]
    f1 = samples[:, :, t1]
    if interp_mode == "nearest":
        a = alpha.view(1, 1, -1, *([1] * tail))
        return torch.where(a < 0.5, f0, f1)
    if interp_mode == "linear":
        a = alpha.view(1, 1, -1, *([1] * tail))
        return f0 * (1 - a) + f1 * a
    vels = _velocities(samples) * motion_scale
    v0   = vels[:, :, t0]
    v1   = vels[:, :, t1]
    a    = alpha.view(1, 1, -1, *([1] * tail))
    h00  =  2*a**3 - 3*a**2 + 1
    h10  =    a**3 - 2*a**2 + a
    h01  = -2*a**3 + 3*a**2
    h11  =    a**3 -   a**2
    return h00*f0 + h10*v0 + h01*f1 + h11*v1


def _unsharp_5d(x, strength):
    B, C, Fv, H, W = x.shape
    flat = x.reshape(B * C * Fv, 1, H, W)
    k    = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]],
                        dtype=x.dtype, device=x.device).div(16.).view(1, 1, 3, 3)
    return (flat + strength * (flat - nnf.conv2d(flat, k, padding=1))).reshape(B, C, Fv, H, W)


class LatentCrossFadeAutoConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"frames_0": ("LATENT",), "frames_1": ("LATENT",)},
            "optional": {
                "match_spatial":     ("BOOLEAN", {"default": True}),
                "interpolate_mode":  (["nearest-exact", "bilinear", "bicubic"], {"default": "nearest-exact"}),
                "cross_fade_frames": ("INT", {"default": 6, "min": 0, "max": 24, "step": 1}),
                "fade_curve":        (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
            },
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "10S Nodes/Latents"
    DESCRIPTION  = "Auto spatial match + temporal concat with optional cross-fade."

    def main(self, frames_0, frames_1, match_spatial=True, interpolate_mode="nearest-exact",
             cross_fade_frames=6, fade_curve="linear"):
        s0 = unwrap_tensor(frames_0["samples"])
        s1 = unwrap_tensor(frames_1["samples"])
        print(f"\u2192 [10S] Concat input: {s0.shape} + {s1.shape}")
        if match_spatial and s0.shape[3:] != s1.shape[3:]:
            s1 = nnf.interpolate(s1, size=(s1.shape[2], s0.shape[3], s0.shape[4]),
                                 mode=interpolate_mode,
                                 align_corners=None if interpolate_mode == "nearest-exact" else False)
        if cross_fade_frames > 0:
            fade = min(cross_fade_frames, s0.shape[2], s1.shape[2])
            if s0.shape[2] >= fade:
                w = torch.linspace(0., 1., fade, device=s0.device, dtype=s0.dtype).view(1,1,fade,1,1)
                blended = s0[:,:,-fade:] * (1-w) + s1[:,:,:fade] * w
                out = torch.cat((s0[:,:,:-fade], blended, s1[:,:,fade:]), dim=2)
            else:
                out = torch.cat((s0, s1), dim=2)
        else:
            out = torch.cat((s0, s1), dim=2)
        print(f"\u2192 [10S] Concat output: {out.shape}\n")
        return safe_return(out)


class AudioLatentStretch:
    """
    Resample LTX audio latent [B,C,T,D] to target FPS via cubic Hermite.
    Hermite velocity field eliminates linear-blend staircase artifacts.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_latent": ("LATENT",),
                "source_fps":   ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "target_fps":   ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
            "optional": {
                "interp_mode": (["hermite", "linear", "nearest"], {"default": "hermite"}),
            },
        }
    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("audio_latent",)
    FUNCTION      = "main"
    CATEGORY      = "10S Nodes/Latents"
    DESCRIPTION   = "Resample audio latent to a target FPS. Hermite mode recommended."

    def main(self, audio_latent, source_fps=24.0, target_fps=30.0, interp_mode="hermite"):
        samples  = unwrap_tensor(audio_latent["samples"])
        squeezed = False
        if samples.dim() == 3:
            samples  = samples.unsqueeze(-1)
            squeezed = True
        T        = samples.shape[2]
        target_T = ltx_safe_frames(int(round(T * (target_fps / source_fps))))
        print(f"\u2192 [10S] AudioStretch: {T} -> {target_T} frames ({source_fps}->{target_fps} fps)")
        if target_T == T:
            if squeezed: samples = samples.squeeze(-1)
            return (_dict_with(audio_latent, samples=samples),)
        out = _stretch_temporal(samples, target_T, interp_mode, motion_scale=1.0)
        if squeezed: out = out.squeeze(-1)
        print(f"\u2192 [10S] AudioStretch output: {out.shape}\n")
        return (_dict_with(audio_latent, samples=out),)


class LatentTemporalUpsampler:
    """
    Motion-compensated temporal upsampling for LTX video latents [B,C,F,H,W].

    auto_retime (default True):
        Sets hermite velocity scale to (source_fps / target_fps) = 0.8 for 24->30.
        Reduces per-frame displacement so playing at 30fps gives the same apparent
        speed as the original 24fps. Without this, corrective resampling regenerates
        at the model's 24fps cadence, which at 30fps reads as ~25% speed increase.

    ltx_safe:
        Rounds target frame count to (N-1)%8==0. The VAE decoder accepts arbitrary
        counts; this is a sampler-side constraint only. Leave False unless needed.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":     ("LATENT",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "target_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
            "optional": {
                "interp_mode":     (["hermite", "linear", "nearest"], {"default": "hermite"}),
                "auto_retime":     ("BOOLEAN", {"default": True}),
                "motion_scale":    ("FLOAT",   {"default": 1.0,  "min": 0.0, "max": 3.0,  "step": 0.05}),
                "spatial_sharpen": ("FLOAT",   {"default": 0.15, "min": 0.0, "max": 1.0,  "step": 0.05}),
                "ltx_safe":        ("BOOLEAN", {"default": False}),
                "override_frames": ("INT",     {"default": 0,    "min": 0,   "max": 4096, "step": 1}),
            },
        }
    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "main"
    CATEGORY      = "10S Nodes/Latents"
    DESCRIPTION   = "Hermite temporal upsampling with auto pacing correction."

    def main(self, latent, source_fps=24.0, target_fps=30.0,
             interp_mode="hermite", auto_retime=True, motion_scale=1.0,
             spatial_sharpen=0.15, ltx_safe=False, override_frames=0):
        samples = unwrap_tensor(latent["samples"])
        if samples.dim() != 5:
            print(f"\u2192 [10S] TemporalUpsampler: expected 5D, got {samples.shape} - passthrough")
            return (_dict_with(latent, samples=samples),)
        B, C, Fv, H, W = samples.shape
        raw      = override_frames if override_frames > 0 else int(round(Fv * (target_fps / source_fps)))
        target_F = ltx_safe_frames(raw) if ltx_safe else raw
        eff      = (source_fps / target_fps) if auto_retime else motion_scale
        print(f"\u2192 [10S] TemporalUpsampler: {samples.shape} | {Fv}->{target_F} | vel={eff:.4f}")
        if target_F == Fv:
            return (_dict_with(latent, samples=samples),)
        result = _stretch_temporal(samples, target_F, interp_mode, eff)
        if spatial_sharpen > 0.0:
            result = _unsharp_5d(result, spatial_sharpen)
        print(f"\u2192 [10S] TemporalUpsampler output: {result.shape}\n")
        return (_dict_with(latent, samples=result),)


class LatentMotionSharpener:
    """
    Motion-adaptive spatial sharpening. Place BEFORE LatentTemporalUpsampler.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",)},
            "optional": {
                "base_sharpen":         ("FLOAT",   {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_sharpen":       ("FLOAT",   {"default": 0.55, "min": 0.0, "max": 2.0, "step": 0.05}),
                "motion_thresh":        ("FLOAT",   {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01}),
                "temporal_smooth_mask": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "main"
    CATEGORY      = "10S Nodes/Latents"
    DESCRIPTION   = "Adaptive sharpening proportional to inter-frame motion magnitude."

    def main(self, latent, base_sharpen=0.08, motion_sharpen=0.55,
             motion_thresh=0.04, temporal_smooth_mask=True):
        samples = unwrap_tensor(latent["samples"])
        if samples.dim() != 5:
            return (_dict_with(latent, samples=samples),)
        B, C, Fv, H, W = samples.shape
        print(f"\u2192 [10S] MotionSharpener: {samples.shape}")
        diff = torch.cat([
            torch.zeros(B, C, 1, H, W, device=samples.device, dtype=samples.dtype),
            (samples[:,:,1:] - samples[:,:,:-1]).abs()
        ], dim=2)
        mmap = diff.mean(dim=1, keepdim=True)
        mmap = mmap / mmap.amax(dim=(2,3,4), keepdim=True).clamp(min=1e-6)
        if temporal_smooth_mask and Fv >= 3:
            m = mmap.permute(0,3,4,1,2).reshape(B*H*W, 1, Fv)
            k = torch.tensor([0.25, 0.5, 0.25], dtype=samples.dtype, device=samples.device).view(1,1,3)
            mmap = nnf.conv1d(m, k, padding=1).reshape(B,H,W,1,Fv).permute(0,3,4,1,2)
        headroom = 1.0 - motion_thresh + 1e-6
        astr = base_sharpen + ((mmap - motion_thresh).clamp(min=0.0) / headroom) * motion_sharpen
        k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]],
                         dtype=samples.dtype, device=samples.device).div(16.).view(1,1,3,3)
        flat   = samples.reshape(B*C*Fv, 1, H, W)
        hf     = flat - nnf.conv2d(flat, k, padding=1)
        str_bc = astr.expand(B, C, Fv, H, W).reshape(B*C*Fv, 1, H, W)
        result = (flat + str_bc * hf).reshape(B, C, Fv, H, W)
        print(f"\u2192 [10S] MotionSharpener output: {result.shape}\n")
        return (_dict_with(latent, samples=result),)


class LatentMotionRetime:
    """
    Post-hoc pacing correction for an already-upsampled latent.

    Attenuates the per-frame velocity by (1 - source_fps/target_fps) to recover
    natural 30fps motion pace from a 24fps-amplitude upsampled latent.

        v[t]             = (f[t+1] - f[t-1]) / 2
        f_retimed[t]     = f[t] - v[t] * (1 - scale) * strength

    For 24->30fps: scale=0.8, each frame shifts 0.2 * velocity toward temporal
    mean, reducing apparent speed by 20%.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",)},
            "optional": {
                "source_fps":   ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "target_fps":   ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "strength":     ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0,   "step": 0.05}),
                "manual_scale": ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 2.0,   "step": 0.05}),
            },
        }
    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "main"
    CATEGORY      = "10S Nodes/Latents"
    DESCRIPTION   = "Attenuate motion amplitude to correct 24->30fps pacing on an upsampled latent."

    def main(self, latent, source_fps=24.0, target_fps=30.0, strength=1.0, manual_scale=0.0):
        samples = unwrap_tensor(latent["samples"])
        if samples.dim() != 5:
            return (_dict_with(latent, samples=samples),)
        scale = manual_scale if manual_scale > 0.0 else (source_fps / target_fps)
        atten = (1.0 - scale) * strength
        print(f"\u2192 [10S] MotionRetime: {samples.shape} | scale={scale:.4f} atten={atten:.4f}")
        if atten < 1e-4:
            return (_dict_with(latent, samples=samples),)
        result = samples - _velocities(samples) * atten
        print(f"\u2192 [10S] MotionRetime output: {result.shape}\n")
        return (_dict_with(latent, samples=result),)


class LatentTemporalInpainter:
    """
    Ghost-aware per-frame noise injection for targeted corrective resampling.

    WHY uniform 0.7 denoise fails:
        The model regenerates at its baked-in 24fps cadence. At 30fps playback
        this reads as ~25% speed increase because the model's priors, not the
        content, determine pacing.

    HOW this works:
        Ghost detection via interpolation residual:
            linear_pred[t] = (f[t-1] + f[t+1]) / 2
            residual[t]    = |f[t] - linear_pred[t]|.mean()
            ghost_score[t] = 1 - normalize(residual)   (1=ghost, 0=real)

        Per-frame noise:
            sigma[t] = anchor_sigma + (ghost_sigma - anchor_sigma) * ghost_score[t]
            noised[t] = sqrt(1 - sigma^2) * f[t] + sigma * noise[t]

        Real frames  -> sigma ~0.05 -> sampler barely touches them (anchors)
        Ghost frames -> sigma ~0.35 -> sampler corrects guided by anchor attention

        anchor_blend re-weights real frames back toward original post-injection
        to further reinforce the temporal locks.

    Corrective KSampler settings:
        denoise: 0.30-0.45  (NOT 0.7: too much freedom = model retimes content)
        steps:   10-20
        The noise_mask in the output latent dict guides LTX's inpainting sampler
        to focus on the ghost frames.

    score_gamma:
        Controls how concentrated the ghost-score distribution becomes
        before being mapped to per-frame sigma.
        >1 (default 2.0) = concentrate noise on worst ghosts only. Raise to
                           3-4 for heavily upsampled latents with severe
                           interpolation artifacts.
        =1               = use raw normalized residual scores directly.
        <1 (0.1-0.5)     = aggressively flatten the distribution so even
                           low-score frames get meaningful correction.
                           Empirically useful when ghost frames span a
                           wide range of severities and you want broad
                           treatment rather than peak-focused correction.
                           User reports 0.1, 0.2, 0.5 producing strong
                           effects for certain content. Minimum 0.01.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",)},
            "optional": {
                "anchor_sigma":  ("FLOAT",   {"default": 0.05, "min": 0.0,  "max": 0.3,  "step": 0.01}),
                "ghost_sigma":   ("FLOAT",   {"default": 0.35, "min": 0.05, "max": 0.80, "step": 0.01}),
                "score_gamma":   ("FLOAT",   {"default": 2.0,  "min": 0.01, "max": 5.0,  "step": 0.01}),
                "anchor_blend":  ("FLOAT",   {"default": 0.4,  "min": 0.0,  "max": 1.0,  "step": 0.05}),
                "seed":          ("INT",     {"default": 0,    "min": 0,    "max": 2**31, "step": 1}),
                "debug_scores":  ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "main"
    CATEGORY      = "10S Nodes/Latents"
    DESCRIPTION   = (
        "Per-frame ghost detection + targeted noise. Locks real anchor frames. "
        "Follow with KSampler denoise 0.30-0.45."
    )

    def main(self, latent, anchor_sigma=0.05, ghost_sigma=0.35,
             score_gamma=2.0, anchor_blend=0.4, seed=0, debug_scores=False):
        samples = unwrap_tensor(latent["samples"])
        if samples.dim() != 5:
            print(f"\u2192 [10S] TemporalInpainter: expected 5D, got {samples.shape} - passthrough")
            return (_dict_with(latent, samples=samples),)
        B, C, Fv, H, W = samples.shape
        print(f"\u2192 [10S] TemporalInpainter: {samples.shape}")

        # Ghost score: low interpolation residual = likely an interpolated frame
        prev = torch.cat([samples[:,:,:1],     samples[:,:,:-1]], dim=2)
        nxt  = torch.cat([samples[:,:,1:],     samples[:,:,-1:]], dim=2)
        pred = (prev + nxt) * 0.5
        residual = (samples - pred).abs().mean(dim=(1, 3, 4))   # [B, Fv]

        r_min  = residual.amin(dim=1, keepdim=True)
        r_max  = residual.amax(dim=1, keepdim=True).clamp(min=1e-6)
        r_norm = (residual - r_min) / (r_max - r_min + 1e-8)

        ghost = (1.0 - r_norm).clamp(0., 1.)
        ghost[:, 0]  = 0.0   # edge frames are always real
        ghost[:, -1] = 0.0
        ghost = ghost.pow(score_gamma)

        if debug_scores:
            top = ghost[0].topk(min(10, Fv)).indices.sort().values
            print(f"  -> Top ghost frames: {top.tolist()}")
            print(f"  -> Scores:           {ghost[0, top].tolist()}")

        sigma  = (anchor_sigma + (ghost_sigma - anchor_sigma) * ghost).to(samples.dtype)
        print(f"  -> sigma range: {sigma.min().item():.4f} - {sigma.max().item():.4f}"
              f" | mean ghost: {ghost.mean().item():.4f}")

        gen   = torch.Generator(device=samples.device)
        gen.manual_seed(seed)
        noise = torch.randn(B, C, Fv, H, W,
                            dtype=samples.dtype, device=samples.device,
                            generator=gen)
        sig5d  = sigma.view(B, 1, Fv, 1, 1).expand_as(samples)
        noised = (1.0 - sig5d**2).clamp(min=0.).sqrt() * samples + sig5d * noise

        if anchor_blend > 0.0:
            anchor_w = (1.0 - ghost).view(B, 1, Fv, 1, 1).expand_as(samples)
            noised   = noised + anchor_w * anchor_blend * (samples - noised)

        # noise_mask: [B,1,Fv,H,W] — 1=inpaint ghost, 0=preserve anchor
        noise_mask = ghost.view(B, 1, Fv, 1, 1).expand(B, 1, Fv, H, W).contiguous()

        print(f"\u2192 [10S] TemporalInpainter output: {noised.shape}\n")
        return (_dict_with(latent, samples=noised, noise_mask=noise_mask),)


NODE_CLASS_MAPPINGS = {
    "LatentCrossFadeAutoConcat": LatentCrossFadeAutoConcat,
    "AudioLatentStretch":        AudioLatentStretch,
    "LatentTemporalUpsampler":   LatentTemporalUpsampler,
    "LatentMotionSharpener":     LatentMotionSharpener,
    "LatentMotionRetime":        LatentMotionRetime,
    "LatentTemporalInpainter":   LatentTemporalInpainter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentCrossFadeAutoConcat": "\U0001f500 Latent Cross Fade Auto Concat",
    "AudioLatentStretch":        "\U0001f3b5 Audio Latent Stretch",
    "LatentTemporalUpsampler":   "\U0001f39e\ufe0f  Latent Temporal Upsampler",
    "LatentMotionSharpener":     "\u26a1 Latent Motion Sharpener",
    "LatentMotionRetime":        "\u23f1\ufe0f  Latent Motion Retime",
    "LatentTemporalInpainter":   "\U0001f47b Latent Temporal Inpainter",
}