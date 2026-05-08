DISCLAIMER: None of these really solve the problem set out to be fixed since my finetune is strictly bound to 24 frame motions and no matter of latent adjustment could overcome it, but they still provide useful tools and successful deep latent manipulation of LTX2.3 video/audio latents. 

10S Nodes — LTX 2.3 Latent Processing
Custom ComfyUI nodes for improving motion quality when working with LTX 2.3's combined audio/video latent pipeline. The primary goal is clean 30fps output from a model that must be sampled at 24fps.

The Core Problem
LTX 2.3 (and strict finetunes of it) bake 24fps temporal priors into their weights. Sampling at any other rate produces degraded output. The challenge is that 24fps latents decoded and played at 30fps show visible motion artifacts — ghosting on fast-moving objects, blurred edges, and unnatural pacing. These nodes address that problem in stages without touching the sampler itself.

Recommended Workflow
[KSampler @ 24 fps]
    └─► [LatentMotionSharpener]
            └─► [LatentTemporalUpsampler  auto_retime=True]
                    └─► [LatentTemporalInpainter]
                                └─► [KSampler  denoise=0.30–0.45]
                                            └─► [Upscale / Decode]

[AudioLatentStretch] → runs in parallel on the audio latent

Nodes

🔀 Latent Cross Fade Auto Concat
Category: 10S Nodes/Latents
Concatenates two video latents along the temporal dimension with optional cross-fade blending and automatic spatial resolution matching. The original node in this pack — untouched in all revisions.
ParameterDescriptionframes_0 / frames_1Latents to concatenate (frames_0 first)match_spatialResizes frames_1 to match frames_0 spatial dims if they differinterpolate_modeResize method: nearest-exact, bilinear, bicubiccross_fade_framesNumber of frames to blend at the join point (0 = hard cut)fade_curveShape of the blend curve (currently linear; others reserved)

🎵 Audio Latent Stretch
Category: 10S Nodes/Latents
Resamples an LTX audio latent [B, C, T, D] to match a new frame count derived from a target FPS. LTX's combined AV latent carries audio and video at the same temporal length; when the video is upsampled from 24 to 30fps the audio must be stretched to match or the decoder will receive mismatched lengths.
Uses cubic Hermite interpolation by default. The original implementation used floor-indexing with linear blending, which produced staircase discontinuities audible as phasing artifacts in stretched audio. Hermite uses a central-difference velocity field to produce smooth arcs through each interpolated sample.
The reference_video_latent input was removed after testing showed it read the upsampler's output frame count (e.g. 17) rather than the full video length (121) due to how the combined latent is structured. FPS ratio is the reliable path.
ParameterDescriptionaudio_latentInput audio latent from the LTX pipelinesource_fpsOriginal sample rate (typically 24.0)target_fpsTarget playback rate (typically 30.0)interp_modehermite (recommended), linear, nearest

⚡ Latent Motion Sharpener
Category: 10S Nodes/Latents
Applies spatially adaptive unsharp masking to a video latent, with sharpening strength proportional to inter-frame motion magnitude. Still regions are left untouched; fast-moving areas receive stronger edge enhancement.
Place this before LatentTemporalUpsampler. The upsampler's hermite velocity field tracks latent feature edges between frames — if those edges are soft going in, the velocity field follows blurry gradients and the interpolated frames inherit that blur. Sharpening first gives the velocity field crisp edges to track, which noticeably reduces ghosting on fast objects.
ParameterDescriptionbase_sharpenSharpening floor applied everywhere (0.05–0.10 recommended)motion_sharpenAdditional strength in high-motion regions (primary quality knob)motion_threshNormalised motion floor before sharpening activates; raise to ignore camera shaketemporal_smooth_mask1-2-1 blur along the motion mask's time axis to prevent per-frame sharpening flicker

🎞️ Latent Temporal Upsampler
Category: 10S Nodes/Latents
Motion-compensated temporal upsampling for video latents [B, C, F, H, W]. Inserts synthetic intermediate frames between the 24fps sampled frames before decoding, using cubic Hermite interpolation to follow the trajectory of latent features rather than blending their values.
The critical parameter here is auto_retime. Without it, every interpolated frame inherits the full 24fps-amplitude displacement between its neighbors. At 30fps playback those same displacements happen more frequently, reading as ~25% speed increase. With auto_retime=True the velocity scale is automatically set to source_fps / target_fps (0.8 for 24→30), reducing per-frame displacement to what genuine 30fps content would carry.
spatial_sharpen applies a post-interpolation unsharp mask to recover the slight softness that cubic blending introduces.
ltx_safe rounds the output frame count to (N-1) % 8 == 0. This is a sampler constraint, not a decoder constraint — leave it off unless the corrective KSampler pass errors on the frame count.
ParameterDescriptionsource_fps / target_fpsDefines the upsampling ratioauto_retimeScales velocity by src/tgt for natural pacing (strongly recommended)interp_modehermite (recommended), linear, nearestmotion_scaleManual velocity scale override when auto_retime=Falsespatial_sharpenPost-interpolation unsharp mask strength (0.10–0.20 typical)ltx_safeRound output to nearest LTX-valid frame countoverride_framesForce a specific output frame count instead of using FPS ratio

⏱️ Latent Motion Retime
Category: 10S Nodes/Latents
Standalone post-hoc pacing correction for an already-upsampled latent. Equivalent to what auto_retime does during upsampling, applied after the fact.
When to use this instead of auto_retime: when you have an existing upsampled latent from a previous run, or when you want to dial in pacing correction strength independently of the upsampling step.
Mechanism: computes a central-difference velocity field v[t] = (f[t+1] - f[t-1]) / 2 and shifts each frame toward its temporal mean by (1 - scale) × strength. For 24→30fps with strength=1.0 this shifts each frame by 20% of its local velocity — exactly cancelling the extra speed introduced by replaying 24fps-amplitude motion at 30fps.
ParameterDescriptionsource_fps / target_fpsUsed to compute attenuation scale automaticallystrengthBlend between no correction (0.0) and full correction (1.0)manual_scaleOverride auto scale if set > 0

👻 Latent Temporal Inpainter
Category: 10S Nodes/Latents
The most important node in the pipeline for corrective resampling quality. Detects which frames in an upsampled latent are interpolated ghosts vs original real frames, then injects calibrated per-frame noise before a second KSampler pass — giving the model more freedom to fix ghost frames while locking real frames as temporal anchors.
Why uniform high denoise (0.7) fails: the model gets equal creative freedom over every frame. It uses that freedom to regenerate content at its baked-in 24fps temporal cadence. At 30fps playback this reads as ~25% speed increase. The model's own priors, not the input content, end up determining the pacing.
How ghost detection works: an interpolated frame closely resembles the linear average of its neighbors — that is literally what hermite interpolation produces. Real frames have independent spatial content. The residual |f[t] - (f[t-1] + f[t+1]) / 2| is therefore low for ghost frames and high for real ones. This residual is normalised per clip and inverted to produce a per-frame ghost score in [0, 1].
How noise injection works:

Ghost frames receive sigma ≈ ghost_sigma (0.35) — enough freedom for the model to resolve blending artifacts
Real frames receive sigma ≈ anchor_sigma (0.05) — barely any noise; the sampler cannot meaningfully move them
anchor_blend further pushes real frames back toward their pre-noise values

Because the real frames are locked, the corrective sampler is forced to reconcile ghost frames within the motion context established by the anchors. It cannot retime the content globally because the anchors resist it. The noise_mask embedded in the output latent dict tells LTX's inpainting-aware sampler exactly which frames need attention.
Recommended corrective KSampler settings: denoise=0.30–0.45, steps=10–20. Do not use 0.7+ — that gives the model enough freedom to override the anchor locking.
ParameterDescriptionanchor_sigmaNoise for real/anchor frames. Lower = stronger temporal lock (0.02–0.08)ghost_sigmaNoise for ghost frames. Ceiling ~0.40 for pacing-safe correctionscore_gammaPower curve on ghost scores. >1 = concentrate on worst ghosts only; raise to 3–4 for heavily upsampled latentsanchor_blendRe-blend real frames toward original after injection (0.3–0.6 recommended)seedNoise generator seed for reproducibilitydebug_scoresPrint the top 10 ghost frame indices and their scores to console

Notes

All nodes preserve the full latent dict (all keys, including LTX metadata) when passing data downstream. Only the keys each node is responsible for are modified. This is critical for the LTX decoder which expects a complete dict structure beyond just "samples".
All nodes return proper (dict,) 1-tuples as required by ComfyUI's _async_map_node_over_list execution path.
The LatentAVSplit and LatentAVMerge nodes were removed from this pack as LTX 2.3's own node ecosystem provides these. Use the native LTX split/merge nodes to separate video and audio before feeding into AudioLatentStretch and the video processing chain.