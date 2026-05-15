"""
LTX Likeness Guide v1.4

Attaches a likeness reference image at the CONDITIONING level for identity
preservation. The Guide's value comes from its modifications to positive/
negative conditioning (which the model interprets during cross-attention)
and the reference_info metadata it emits (which LikenessAnchor reads for
attention pull). The latent extension/keyframe mechanism is optional.

================================================================================
CHANGES IN v1.4
================================================================================
- New emit_latent parameter (default 'passthrough'): the latent output is
  now the unmodified input latent by default. The Guide's conditioning
  modifications still flow to Guider, and reference_info still flows to
  LikenessAnchor — those are where the actual identity-preservation
  mechanisms live. The latent extension was causing keyframe-pattern
  composition loop even in silent_reference mode (the spatial pattern
  itself triggers learned keyframe behavior). Passthrough mode avoids
  this entirely.
- emit_latent='extended' preserves v1.3 behavior for users who explicitly
  want the in-latent reference mechanism (requires LikenessCrop downstream).
- LikenessCrop's reference_info is now optional with graceful passthrough,
  so workflows don't break validation when Guide is in passthrough mode.

================================================================================
ARCHITECTURE (v1.4)
================================================================================

  Input latent ────────────────────────────────────────────────┐
                                                                ↓
  positive, negative ──→ Guide(image) ──→ pos', neg' ──→ Guider → KSampler
                              ↓
                      reference_info ──→ LikenessAnchor → model patch

  No LikenessCrop needed. The Guide modifies conditioning and emits
  metadata; LikenessAnchor uses the metadata to pull attention toward
  identity features (frame_0 of the latent, by default, since that's
  the conditioning start image in i2v workflows).

================================================================================
CHANGES IN v1.3
================================================================================
- Fixed decoder periodic artifact on the last generated frame. Two causes:
  (1) silent_reference's hard noise_mask=0.0 created a temporal discontinuity
      between the preserved reference and the fully-denoised adjacent frame.
      The VAE decoder interpreted this as periodic structure (visible as
      blob-halos in image 2). Fixed by introducing a small noise_mask floor
      (0.10) so the reference can integrate smoothly with sampling noise
      while still being ~90% preserved. Identity preservation effectively
      unchanged.
  (2) bbox_softfade zeroed the non-face region of the reference latent.
      Zeros are not "absence" — they decode to specific (wrong) content,
      and the boundary between non-zero face content and zero non-face
      content was where the periodic artifact emerged. Fixed by replacing
      the zeroed region with the PREVIOUS frame's latent content. The
      non-face region of the reference is now a temporal continuation of
      the user's last generated frame — no content cliff for the decoder.

================================================================================
CHANGES IN v1.2
================================================================================
- New placement_mode parameter: 'silent_reference' (default) vs 'keyframe'.
  silent_reference encodes and places the reference in the latent and sets
  noise_mask to preserve it, but does NOT call LTXVAddGuide.append_keyframe.
  The model still attends to the reference via natural bidirectional self-
  attention (giving us identity preservation), but receives no keyframe
  metadata in the conditioning — no interpolation pressure toward end-frame,
  no composition loop back to first frame.
- keyframe mode preserves the original v1.1 behavior (uses LTXVAddGuide
  with its append_keyframe call). Available for users who specifically want
  end-keyframe interpolation behavior (e.g., scene transitions).
- Metadata now reports actual final latent length (post-causal-fix
  expansion) so LikenessCrop and LikenessAnchor see consistent values.

================================================================================
Built on the LTX-native conditioning mechanism (LTXVAddGuide) with extensions
for identity-preservation use cases.

================================================================================
CHANGES IN v1.1
================================================================================
- Added face_detect parameter: 'auto' uses MediaPipe (best) or OpenCV Haar
  cascade (fallback) to detect face bbox automatically. 'manual' uses the
  face_bbox_within_reference string. 'none' = whole frame (v1.0 behavior).
- Added reference_mask_mode parameter: controls how non-face regions of the
  reference latent are handled. 'whole_frame' = unchanged (default v1.0).
  'bbox_only' = zero outside bbox. 'bbox_softfade' (new default) = Gaussian
  fade outside bbox. Addresses the issue where guide_strength > 0 caused
  output to loop/interpolate toward the entire reference as an end-keyframe.
- Added face_padding parameter: expands detected bbox to capture hair/neck
  context (default 0.15 = 15% padding).
- Outside-bbox region of reference latent has noise_mask restored to 1.0
  (full denoise allowed) so model fills those regions naturally instead
  of preserving zero-valued latent there.

================================================================================
ARCHITECTURE
================================================================================

  1. Extends the latent by 1 latent frame at the end (so the reference doesn't
     consume any of the user's intended output frames)
  2. Places the encoded reference image in those new frames
  3. Detects face bbox (auto) or uses provided manual bbox
  4. Masks reference latent so only face region (with optional softfade)
     carries identity content; outside-bbox region zeroed and re-enabled
     for normal denoising
  5. Marks the face region preserved in noise_mask
  6. Attaches metadata to conditioning indicating reference location and bbox

================================================================================
WHY THIS HELPS WITH LIKENESS DRIFT
================================================================================

LTX2's video self-attention (attn1) is fully bidirectional across the entire
sequence. When a clean reference frame is included in the latent, generated
tokens naturally attend to its features during sampling. Because the reference
is preserved (no noise applied via noise_mask), it remains a stable identity
anchor throughout all sampling steps.

With v1.1's bbox masking, the reference's *identity content* is restricted
to the face region. Non-face regions of the reference are blanked, so the
model's natural attention doesn't pull whole-frame content (avoiding the
"looping" / end-keyframe-interpolation effect at higher guide strengths).

This works at the CONDITIONING level — the model sees the masked reference
as part of its input. It complements (rather than replaces) attn1-hook
approaches like LikenessAnchor, which can additionally amplify the pull
toward reference features.

================================================================================
INTEGRATION WITH LIKENESS ANCHOR
================================================================================

Metadata attached to conditioning includes:
  _10s_likeness_reference: {
    start_latent_frame: int,
    frame_count_latent: int,
    attention_strength: float,
    face_bbox: str,             # resolved bbox (auto or manual)
    spatial_dims: (H, W),
    reference_mask_mode: str,
  }

LikenessAnchor reads from the REFERENCE_INFO output (cleaner than walking
conditioning metadata) and uses face_bbox to restrict which reference
tokens contribute to attention pull.

================================================================================
COMPATIBILITY
================================================================================

- LTX2 with audio (NestedTensor wrapper): supported
- Plain video latent: supported
- Auto face detection: requires either mediapipe or opencv-python.
  Both are optional — install whichever is convenient:
      pip install mediapipe          # best quality
      pip install opencv-python      # widely available
  If neither installed, auto mode silently falls back to manual bbox or
  whole frame.
- Causal VAE asymmetry (the Comfy-Org PR #13625 issue): we append at
  end-of-latent, not mid-video, so we're not affected by the freeze/
  stutter the PR addresses. Our single-image guide encodes one latent
  frame from one pixel frame, which is the native causal behavior.
"""

import torch
import comfy
import comfy_extras.nodes_lt as nodes_lt

try:
    import comfy_extras.nodes_post_processing as post_processing
    _HAS_POST_PROCESSING = True
except ImportError:
    _HAS_POST_PROCESSING = False

# Import wrapper extraction/reconstruction from tiled sampler module
# (same package, shared helpers)
from .latent_tiled_sampler import _extract_components, _reconstruct_samples


METADATA_KEY = "_10s_likeness_reference"


# ─────────────────────────────────────────────────────────────────────────────
# Face detection (auto-bbox)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_face_bbox(image_np, padding=0.15, debug=False):
    """
    Detect the largest face in an HxWx3 uint8 image and return normalized
    (x1, y1, x2, y2) bbox, or None if no face found / no detection backend.

    Tries MediaPipe first (better, more accurate), then OpenCV Haar cascade.
    Both are optional dependencies — fails gracefully with informative log.

    Args:
        image_np   : numpy array HxWx3 uint8
        padding    : fraction to expand bbox by (0.15 = 15% padding around face,
                     captures hair/neck context)
    """
    H, W = image_np.shape[:2]

    # Try MediaPipe (more accurate, supports angles)
    try:
        import mediapipe as mp
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as detector:
            results = detector.process(image_np)
            if results.detections:
                # Find largest detection
                best = None
                best_area = 0
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    area = box.width * box.height
                    if area > best_area:
                        best_area = area
                        best = box
                if best is not None:
                    x1 = max(0.0, best.xmin)
                    y1 = max(0.0, best.ymin)
                    x2 = min(1.0, best.xmin + best.width)
                    y2 = min(1.0, best.ymin + best.height)
                    # Apply padding
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    half_w = (x2 - x1) / 2 * (1.0 + padding)
                    half_h = (y2 - y1) / 2 * (1.0 + padding)
                    x1 = max(0.0, cx - half_w)
                    y1 = max(0.0, cy - half_h)
                    x2 = min(1.0, cx + half_w)
                    y2 = min(1.0, cy + half_h)
                    if debug:
                        print(f"  \u00b7 [face_detect] MediaPipe found face: "
                              f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})")
                    return (x1, y1, x2, y2)
    except ImportError:
        if debug:
            print(f"  \u00b7 [face_detect] mediapipe not installed; trying OpenCV")
    except Exception as e:
        if debug:
            print(f"  \u00b7 [face_detect] MediaPipe error: {type(e).__name__}: {e}")

    # Fallback: OpenCV Haar cascade
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            if debug:
                print(f"  \u00b7 [face_detect] OpenCV cascade load failed")
            return None
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            if debug:
                print(f"  \u00b7 [face_detect] OpenCV: no face found")
            return None
        # Largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x1 = x / W
        y1 = y / H
        x2 = (x + w) / W
        y2 = (y + h) / H
        # Apply padding
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_w = (x2 - x1) / 2 * (1.0 + padding)
        half_h = (y2 - y1) / 2 * (1.0 + padding)
        x1 = max(0.0, cx - half_w)
        y1 = max(0.0, cy - half_h)
        x2 = min(1.0, cx + half_w)
        y2 = min(1.0, cy + half_h)
        if debug:
            print(f"  \u00b7 [face_detect] OpenCV Haar found face: "
                  f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})")
        return (x1, y1, x2, y2)
    except ImportError:
        if debug:
            print(f"  \u00b7 [face_detect] opencv-python not installed")
        return None
    except Exception as e:
        if debug:
            print(f"  \u00b7 [face_detect] OpenCV error: {type(e).__name__}: {e}")
        return None


def _bbox_str_to_tuple(s):
    """Parse 'x1,y1,x2,y2' string to (x1,y1,x2,y2) floats. Returns None on failure."""
    if not s or not s.strip():
        return None
    try:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 4:
            return None
        return tuple(parts)
    except (ValueError, TypeError):
        return None


def _tuple_to_bbox_str(t):
    if t is None:
        return ""
    return ",".join(f"{v:.4f}" for v in t)


def _build_reference_latent_mask(bbox, latent_H, latent_W, mode="bbox_softfade",
                                  fade_sigma_frac=0.10, device=None, dtype=None):
    """
    Build a per-pixel mask in latent space for the reference frame.
      mode='bbox_only'      → hard 1.0 inside bbox, 0.0 outside
      mode='bbox_softfade'  → 1.0 inside, Gaussian fade outside
      mode='whole_frame'    → all 1.0 (no masking)

    Returns (1, 1, 1, latent_H, latent_W) tensor.
    """
    if mode == "whole_frame" or bbox is None:
        m = torch.ones((1, 1, 1, latent_H, latent_W), dtype=dtype, device=device)
        return m

    x1, y1, x2, y2 = bbox
    h1 = max(0, int(y1 * latent_H))
    h2 = min(latent_H, max(h1 + 1, int(round(y2 * latent_H))))
    w1 = max(0, int(x1 * latent_W))
    w2 = min(latent_W, max(w1 + 1, int(round(x2 * latent_W))))

    if mode == "bbox_only":
        m = torch.zeros((1, 1, 1, latent_H, latent_W), dtype=dtype, device=device)
        m[:, :, :, h1:h2, w1:w2] = 1.0
        return m

    # bbox_softfade: build distance field to bbox, Gaussian fade
    yy = torch.arange(latent_H, dtype=torch.float32, device=device).view(-1, 1)
    xx = torch.arange(latent_W, dtype=torch.float32, device=device).view(1, -1)
    # Signed distance: 0 inside box, positive distance outside
    dy = torch.maximum(torch.maximum(h1 - yy, yy - (h2 - 1)),
                       torch.zeros(1, device=device))
    dx = torch.maximum(torch.maximum(w1 - xx, xx - (w2 - 1)),
                       torch.zeros(1, device=device))
    dist = torch.sqrt(dy * dy + dx * dx)
    # Gaussian fade
    fade_sigma = max(1.0, fade_sigma_frac * max(latent_H, latent_W))
    fade = torch.exp(-(dist ** 2) / (2 * fade_sigma ** 2))
    m = fade.view(1, 1, 1, latent_H, latent_W)
    if dtype is not None:
        m = m.to(dtype=dtype)
    return m


def _blur_internal(image, blur_radius):
    """Same blur as LTXVAddGuideAdvanced. No-op if blur_radius=0 or
    post_processing module unavailable."""
    if blur_radius > 0 and _HAS_POST_PROCESSING:
        sigma = 0.3 * blur_radius
        image = post_processing.Blur.execute(image, blur_radius, sigma)[0]
    return image


def _attach_metadata_to_conditioning(cond, metadata):
    """
    Attach metadata to each conditioning entry's dict.

    ComfyUI conditioning is a list of [tensor, dict] pairs. The dict
    flows through to transformer_options during sampling, so any keys we
    add here are visible to attention hooks at runtime.
    """
    new_cond = []
    for entry in cond:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            tensor, meta = entry[0], entry[1]
            new_meta = dict(meta) if isinstance(meta, dict) else {}
            new_meta[METADATA_KEY] = metadata
            new_entry = [tensor, new_meta]
            new_cond.append(new_entry)
        else:
            # Unexpected format — leave entry untouched
            new_cond.append(entry)
    return new_cond


class LTXLikenessGuide:
    """
    Attach a likeness reference image to the latent for identity preservation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "How strongly the reference frames are "
                                   "preserved via noise_mask. 1.0 = fully "
                                   "preserved (clean reference throughout "
                                   "sampling). With placement_mode="
                                   "silent_reference (default), strength=1 "
                                   "is safe and recommended — no looping.",
                    },
                ),
                "placement_mode": (
                    ["silent_reference", "keyframe"],
                    {
                        "default": "silent_reference",
                        "tooltip": "silent_reference (recommended): place "
                                   "reference in latent and preserve via "
                                   "noise_mask, but DO NOT register it as "
                                   "a keyframe in conditioning. The model "
                                   "still attends to it naturally for "
                                   "identity, without interpolation pressure "
                                   "that causes end-frame looping. "
                                   "keyframe: use LTXVAddGuide's full "
                                   "keyframe registration (original v1.1 "
                                   "behavior). May cause composition loop "
                                   "back to first frame.",
                    },
                ),
                "face_detect": (
                    ["auto", "manual", "none"],
                    {
                        "default": "auto",
                        "tooltip": "auto: detect face via MediaPipe/OpenCV. "
                                   "manual: use face_bbox_within_reference. "
                                   "none: whole reference treated as identity "
                                   "(matches v1.0 behavior).",
                    },
                ),
                "reference_mask_mode": (
                    ["whole_frame", "bbox_only", "bbox_softfade"],
                    {
                        "default": "bbox_softfade",
                        "tooltip": "How to mask the reference latent. "
                                   "whole_frame = unchanged (model attends to "
                                   "entire reference, may cause looping). "
                                   "bbox_only = zero outside bbox in latent. "
                                   "bbox_softfade (recommended) = Gaussian "
                                   "fade outside bbox.",
                    },
                ),
                "face_padding": (
                    "FLOAT",
                    {
                        "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                        "tooltip": "Padding around detected face bbox as "
                                   "fraction (0.15 = 15% expansion). Captures "
                                   "hair/neck context for stronger identity.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 29, "min": 0, "max": 51,
                        "tooltip": "CRF for reference image preprocessing. "
                                   "Higher = softer reference, less rigid pull.",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 7,
                        "tooltip": "Blur radius for reference image. Higher = "
                                   "softer reference.",
                    },
                ),
                "interpolation": (
                    ["lanczos", "bislerp", "nearest", "bilinear",
                     "bicubic", "area", "nearest-exact"],
                    {"default": "lanczos"},
                ),
                "crop": (["center", "disabled"], {"default": "center"}),
                "attention_strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                        "tooltip": "Metadata for downstream attention hooks "
                                   "(LTXLikenessAnchor). Doesn't affect this "
                                   "node's own output, only what downstream "
                                   "anchors read.",
                    },
                ),
                "face_bbox_within_reference": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Manual bbox 'x1,y1,x2,y2' normalized 0-1. "
                                   "Used when face_detect=manual, or as "
                                   "fallback when auto detection fails. "
                                   "Empty + auto failed = whole frame.",
                    },
                ),
                "emit_latent": (
                    ["passthrough", "extended"],
                    {
                        "default": "passthrough",
                        "tooltip": "passthrough (recommended): the latent "
                                   "output is the unmodified input latent. "
                                   "Guide's value comes from its conditioning "
                                   "modifications (which feed into Guider) "
                                   "and the reference_info metadata it emits "
                                   "(which feeds into LikenessAnchor). Wire "
                                   "the input latent directly to your sampler. "
                                   "extended: emits the extended latent with "
                                   "the reference frame appended. Requires "
                                   "LikenessCrop downstream to remove the "
                                   "extension before VAE decode. Note: may "
                                   "cause end-keyframe interpolation pressure "
                                   "even in silent_reference mode (the "
                                   "spatial pattern itself can trigger "
                                   "learned keyframe behavior).",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "REFERENCE_INFO")
    RETURN_NAMES = ("positive", "negative", "latent", "reference_info")
    FUNCTION = "attach_guide"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Attach likeness reference image to conditioning for identity "
        "preservation. Emits modified positive/negative conditioning (for "
        "Guider) and reference_info metadata (for LikenessAnchor). By default "
        "the latent output is a passthrough of the input — no extension, no "
        "crop needed. The Guide's value comes from its conditioning-level "
        "interaction with the model and its bbox metadata, not from latent "
        "modification."
    )

    def attach_guide(self, positive, negative, vae, latent, image,
                     strength=1.0,
                     placement_mode="silent_reference",
                     face_detect="auto",
                     reference_mask_mode="bbox_softfade",
                     face_padding=0.15,
                     crf=29, blur_radius=0,
                     interpolation="lanczos", crop="center",
                     attention_strength=1.0,
                     face_bbox_within_reference="",
                     emit_latent="passthrough",
                     debug=False):

        # Preserve the original raw input latent for passthrough mode
        original_input_latent = latent
        latent = dict(latent)  # shallow copy
        raw_samples = latent["samples"]

        # ─── Detect wrapper format (audio bundled?) ──────────────────────────
        is_wrapped = not isinstance(raw_samples, torch.Tensor)
        audio_tensor = None
        format_info = None
        if is_wrapped:
            try:
                video_tensor, audio_tensor, format_info = _extract_components(
                    raw_samples, debug=debug
                )
                if debug:
                    print(f"\u2192 [10S] LikenessGuide: wrapped input "
                          f"video={tuple(video_tensor.shape)} "
                          f"audio={tuple(audio_tensor.shape) if audio_tensor is not None else None}")
            except Exception as e:
                raise RuntimeError(
                    f"LTXLikenessGuide: cannot extract video from wrapped "
                    f"latent: {type(e).__name__}: {e}"
                )
        else:
            video_tensor = raw_samples
            if debug:
                print(f"\u2192 [10S] LikenessGuide: plain video latent "
                      f"shape={tuple(video_tensor.shape)}")

        if video_tensor.dim() != 5:
            raise RuntimeError(
                f"LTXLikenessGuide: expected 5D video latent (B,C,F,H,W), "
                f"got {video_tensor.dim()}D"
            )

        # ─── Compute spatial dims and resize reference image ─────────────────
        _, width_scale_factor, height_scale_factor = vae.downscale_index_formula
        B, C, latent_length, latent_height, latent_width = video_tensor.shape
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor

        if debug:
            print(f"  \u00b7 latent_length={latent_length} "
                  f"latent_height={latent_height} latent_width={latent_width}")
            print(f"  \u00b7 target image size: {target_width}x{target_height}")

        # Resize image to match latent's pixel dims
        image_processed = comfy.utils.common_upscale(
            image.movedim(-1, 1), target_width, target_height,
            interpolation, crop=crop,
        ).movedim(1, -1).clamp(0, 1)
        # LTX preprocessing (CRF)
        image_processed = nodes_lt.LTXVPreprocess().execute(
            image_processed, crf
        )[0]
        # Optional blur
        image_processed = _blur_internal(image_processed, blur_radius)

        # ─── Face detection (if requested) ───────────────────────────────────
        # Resolve the bbox using face_detect mode. Detection runs on the
        # *resized* image_processed so coordinates match the latent geometry.
        resolved_bbox = None
        if face_detect == "auto":
            try:
                # image_processed is (1, H, W, 3) in [0,1] — convert for detection
                img_for_detect = image_processed[0].cpu().clamp(0, 1)
                img_np = (img_for_detect * 255.0).to(torch.uint8).numpy()
                resolved_bbox = _detect_face_bbox(img_np, padding=face_padding,
                                                  debug=debug)
                if resolved_bbox is None:
                    # Auto failed — fall back to manual if provided
                    manual = _bbox_str_to_tuple(face_bbox_within_reference)
                    if manual is not None:
                        resolved_bbox = manual
                        if debug:
                            print(f"  \u00b7 auto detect failed; using manual "
                                  f"bbox {manual}")
                    else:
                        if debug:
                            print(f"  \u00b7 auto detect failed AND no manual "
                                  f"bbox provided; whole frame will be used")
            except Exception as e:
                print(f"\u2192 [10S] LikenessGuide: face detection error: "
                      f"{type(e).__name__}: {e}")
                resolved_bbox = _bbox_str_to_tuple(face_bbox_within_reference)
        elif face_detect == "manual":
            resolved_bbox = _bbox_str_to_tuple(face_bbox_within_reference)
            if resolved_bbox is None and debug:
                print(f"  \u00b7 face_detect=manual but no bbox provided; "
                      f"whole frame will be used")
        # face_detect == "none" → resolved_bbox stays None (whole frame mode)

        if debug:
            bbox_str = (f"({resolved_bbox[0]:.3f},{resolved_bbox[1]:.3f},"
                        f"{resolved_bbox[2]:.3f},{resolved_bbox[3]:.3f})"
                        if resolved_bbox is not None else "<none — whole frame>")
            print(f"  \u00b7 resolved bbox: {bbox_str}")
            print(f"  \u00b7 reference_mask_mode={reference_mask_mode}")

        # ─── Extend video latent by 1 latent frame at end ────────────────────
        extension_latent_frames = 1
        extension_pixel_frames = extension_latent_frames * 8  # LTX2 8× temporal
        new_latent_length = latent_length + extension_latent_frames

        if debug:
            print(f"  \u00b7 extending latent: {latent_length} \u2192 "
                  f"{new_latent_length} latent frames")

        # Build extended video latent — original content + zero pad at end
        extended_video = torch.zeros(
            (B, C, new_latent_length, latent_height, latent_width),
            dtype=video_tensor.dtype, device=video_tensor.device
        )
        extended_video[:, :, :latent_length] = video_tensor

        # Build / extend noise mask
        current_noise_mask = latent.get("noise_mask", None)
        # Detect wrapped mask
        if current_noise_mask is not None and not isinstance(
            current_noise_mask, torch.Tensor
        ):
            try:
                current_noise_mask, _, _ = _extract_components(
                    current_noise_mask, debug=False
                )
            except Exception:
                current_noise_mask = None

        # Build extended mask — same shape as extended video, 1.0 = noise, 0.0 = preserve
        new_mask = torch.ones(
            (B, 1, new_latent_length, latent_height, latent_width),
            dtype=video_tensor.dtype, device=video_tensor.device
        )
        if current_noise_mask is not None and current_noise_mask.dim() == 5:
            # Resize to spatial dims if needed
            if current_noise_mask.shape[2:] == (latent_length, latent_height, latent_width):
                new_mask[:, :, :latent_length] = current_noise_mask
            else:
                # Spatial dims don't match — interpolate
                try:
                    resized = torch.nn.functional.interpolate(
                        current_noise_mask.float(),
                        size=(latent_length, latent_height, latent_width),
                        mode="trilinear", align_corners=False,
                    ).to(dtype=video_tensor.dtype)
                    new_mask[:, :, :latent_length] = resized
                except Exception as e:
                    if debug:
                        print(f"  \u26a0  noise_mask resize failed: "
                              f"{type(e).__name__}: {e}; using all-ones for original")

        extended_dict = {
            "samples": extended_video,
            "noise_mask": new_mask,
        }

        # ─── Place the reference in the latent ───────────────────────────────
        # frame_idx in pixel-space — the start of the extended region
        target_frame_idx = latent_length * 8

        if debug:
            print(f"  \u00b7 placing reference at pixel frame_idx="
                  f"{target_frame_idx} (mode={placement_mode})")

        if placement_mode == "keyframe":
            # ─── Original v1.1 behavior: use LTXVAddGuide ───────────────────
            # This calls append_keyframe under the hood, which writes
            # structured keyframe metadata into the conditioning. The model
            # interprets this as "interpolate toward this position" — causes
            # end-frame composition loop.
            try:
                ltx_result = nodes_lt.LTXVAddGuide().generate(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=extended_dict,
                    image=image_processed,
                    frame_idx=target_frame_idx,
                    strength=strength,
                )
            except Exception as e:
                raise RuntimeError(
                    f"LTXLikenessGuide: LTXVAddGuide.generate failed: "
                    f"{type(e).__name__}: {e}"
                )
            # LTXVAddGuide returns (positive, negative, latent_dict)
            out_positive, out_negative, out_latent_dict = ltx_result

        else:
            # ─── silent_reference: encode + place + noise_mask, NO keyframe ─
            # The reference is in the latent (model attends to it naturally
            # via bidirectional self-attention) but the conditioning is
            # unmodified — no keyframe interpolation pressure.
            try:
                scale_factors = vae.downscale_index_formula
                # encode() handles the causal-fix logic internally based on
                # frame_idx and num_frames
                _, encoded_t = nodes_lt.LTXVAddGuide.encode(
                    vae, latent_width, latent_height,
                    image_processed, scale_factors
                )

                # Compute latent index from frame_idx (same logic LTXVAddGuide uses)
                resolved_frame_idx, latent_idx = (
                    nodes_lt.LTXVAddGuide.get_latent_index(
                        positive,
                        new_latent_length,
                        len(image_processed),
                        target_frame_idx,
                        scale_factors,
                    )
                )

                # Place encoded latent at latent_idx, update noise_mask there
                placed_latent = extended_video.clone()
                placed_mask = new_mask.clone()

                # encoded_t shape: (B, C, t_frames, H, W)
                t_frames = encoded_t.shape[2]
                end_idx = latent_idx + t_frames
                if end_idx > new_latent_length:
                    # Encoded reference is larger than our extension; we
                    # account for the causal-fix prepend that LTXVAddGuide.encode
                    # may have added. Expand the latent further.
                    extra_needed = end_idx - new_latent_length
                    if debug:
                        print(f"  \u00b7 silent_reference: encoded ref needs "
                              f"{extra_needed} extra latent frame(s); "
                              f"expanding")
                    extra_zeros = torch.zeros(
                        (B, C, extra_needed, latent_height, latent_width),
                        dtype=placed_latent.dtype, device=placed_latent.device,
                    )
                    placed_latent = torch.cat([placed_latent, extra_zeros], dim=2)
                    extra_mask = torch.ones(
                        (B, 1, extra_needed, latent_height, latent_width),
                        dtype=placed_mask.dtype, device=placed_mask.device,
                    )
                    placed_mask = torch.cat([placed_mask, extra_mask], dim=2)
                    new_latent_length = placed_latent.shape[2]

                placed_latent[:, :, latent_idx:end_idx] = encoded_t.to(
                    dtype=placed_latent.dtype, device=placed_latent.device
                )
                # noise_mask in placed region: scale by strength.
                # NOTE: we DO NOT use 0.0 (hard preserve) even at strength=1.0
                # because hard preserve next to fully-denoised generated
                # content creates a temporal discontinuity that decodes as
                # periodic VAE artifacts on the adjacent frame. A small floor
                # (0.10) lets the model integrate the reference smoothly with
                # the surrounding sampling noise, eliminating the boundary.
                # Identity preservation is essentially unaffected — the
                # reference still dominates this region by 90% throughout
                # sampling.
                NOISE_MASK_FLOOR = 0.10
                preserve_value = NOISE_MASK_FLOOR + (1.0 - NOISE_MASK_FLOOR) * \
                    (1.0 - max(0.0, min(1.0, strength)))
                placed_mask[:, :, latent_idx:end_idx] = preserve_value

                out_positive = positive  # NO keyframe modification
                out_negative = negative
                out_latent_dict = {
                    "samples": placed_latent,
                    "noise_mask": placed_mask,
                }

                if debug:
                    print(f"  \u00b7 silent_reference: placed encoded ref at "
                          f"latent[{latent_idx}:{end_idx}], "
                          f"noise_mask={preserve_value:.2f} in that region, "
                          f"conditioning unchanged (no keyframe)")
            except Exception as e:
                raise RuntimeError(
                    f"LTXLikenessGuide: silent_reference placement failed: "
                    f"{type(e).__name__}: {e}"
                )

        # ─── Apply reference latent mask ─────────────────────────────────────
        # If face_detect found / user provided a bbox AND mask mode is not
        # whole_frame, mask the reference latent so only the face region
        # carries identity features. This addresses the "looping" issue when
        # guide_strength > 0 (model interpolates toward end keyframe) by
        # making the keyframe identity-only rather than whole-scene.
        if (resolved_bbox is not None and reference_mask_mode != "whole_frame"
                and out_latent_dict is not None
                and "samples" in out_latent_dict):

            ref_samples = out_latent_dict["samples"]
            # The samples may already be wrapped if input was wrapped — extract
            ref_is_wrapped = not isinstance(ref_samples, torch.Tensor)
            ref_video_t = None
            ref_audio_t = None
            ref_fmt = None
            if ref_is_wrapped:
                try:
                    ref_video_t, ref_audio_t, ref_fmt = _extract_components(
                        ref_samples, debug=False
                    )
                except Exception as e:
                    if debug:
                        print(f"  \u26a0  mask: couldn't extract video from "
                              f"output wrapper ({type(e).__name__}: {e}); "
                              f"skipping mask")
                    ref_video_t = None
            else:
                ref_video_t = ref_samples

            if ref_video_t is not None and ref_video_t.dim() == 5:
                ref_dtype = ref_video_t.dtype
                ref_device = ref_video_t.device
                # Build mask in latent space (1, 1, 1, H, W)
                ref_mask = _build_reference_latent_mask(
                    bbox=resolved_bbox,
                    latent_H=latent_height, latent_W=latent_width,
                    mode=reference_mask_mode,
                    fade_sigma_frac=0.10,
                    device=ref_device, dtype=ref_dtype,
                )
                # Apply ONLY to the reference frame portion at the end.
                # Instead of zeroing non-face regions (which creates a sharp
                # content/zero boundary that decodes as periodic VAE
                # artifacts on the adjacent generated frame), blend with the
                # PREVIOUS frame's content. The non-face region of the
                # reference becomes a temporal continuation of the user's
                # last generated frame, so the decoder sees natural
                # frame-to-frame variation rather than a content cliff.
                ref_slice = ref_video_t[:, :, latent_length:, :, :]
                prev_frame = ref_video_t[:, :, latent_length - 1:latent_length, :, :]
                # ref_mask is per-pixel weight (1 inside bbox, 0 outside);
                # inverse weight is the prev-frame contribution.
                inv_ref_mask = (1.0 - ref_mask).to(dtype=ref_dtype)
                masked_ref_slice = ref_slice * ref_mask + prev_frame * inv_ref_mask
                ref_video_t = torch.cat([
                    ref_video_t[:, :, :latent_length, :, :],
                    masked_ref_slice,
                ], dim=2)

                # noise_mask outside-bbox region: with prev-frame continuation
                # in the latent there, we want a smooth preserve. The bbox
                # interior stays at its original preserve_value; outside the
                # bbox we use the same value (no sharp denoise-vs-preserve
                # boundary that would also produce decoder periodicity).
                # The non-face region having moderate preserve means the
                # model can still adjust it during sampling, but it starts
                # from continuation content rather than zeros.
                nm = out_latent_dict.get("noise_mask")
                if nm is not None and isinstance(nm, torch.Tensor) and nm.dim() == 5:
                    new_nm = nm.clone()
                    # Leave reference region's noise_mask as-is (already set
                    # to preserve_value in silent_reference placement).
                    # No outside-bbox override — the whole reference region
                    # gets the same preserve treatment.
                    out_latent_dict["noise_mask"] = new_nm

                # Re-wrap if needed
                if ref_is_wrapped and ref_audio_t is not None and ref_fmt is not None:
                    try:
                        out_latent_dict["samples"] = _reconstruct_samples(
                            ref_video_t, ref_audio_t, ref_fmt, debug=False
                        )
                    except Exception as e:
                        if debug:
                            print(f"  \u26a0  mask: re-wrap failed "
                                  f"({type(e).__name__}: {e}); "
                                  f"outputting plain video")
                        out_latent_dict["samples"] = ref_video_t
                else:
                    out_latent_dict["samples"] = ref_video_t

                if debug:
                    pct_active = (ref_mask > 0.5).float().mean().item() * 100
                    print(f"  \u00b7 reference mask applied: "
                          f"{pct_active:.1f}% of latent area active "
                          f"(rest faded/zeroed)")

        # Determine actual final latent length from the placed result
        # (silent_reference mode may have expanded beyond initial extension
        # to accommodate causal-fix prepend)
        try:
            placed_samples = out_latent_dict["samples"]
            if isinstance(placed_samples, torch.Tensor) and placed_samples.dim() == 5:
                actual_final_length = placed_samples.shape[2]
            else:
                # Wrapped — extract to inspect
                try:
                    pv, _, _ = _extract_components(placed_samples, debug=False)
                    actual_final_length = pv.shape[2] if pv.dim() == 5 else new_latent_length
                except Exception:
                    actual_final_length = new_latent_length
        except Exception:
            actual_final_length = new_latent_length

        # ─── Attach our metadata ─────────────────────────────────────────────
        metadata = {
            "start_latent_frame": latent_length,
            "frame_count_latent": actual_final_length - latent_length,
            "start_pixel_frame": target_frame_idx,
            "frame_count_pixel": (actual_final_length - latent_length) * 8,
            "original_latent_length": latent_length,
            "extended_latent_length": actual_final_length,
            "attention_strength": attention_strength,
            "face_bbox": _tuple_to_bbox_str(resolved_bbox),
            "spatial_dims_latent": [latent_height, latent_width],
            "guide_strength": strength,
            "reference_mask_mode": reference_mask_mode,
            "face_detect_mode": face_detect,
            "placement_mode": placement_mode,
        }

        out_positive = _attach_metadata_to_conditioning(out_positive, metadata)
        out_negative = _attach_metadata_to_conditioning(out_negative, metadata)

        # ─── Re-wrap if input was wrapped ────────────────────────────────────
        # Audio doesn't get extended — its temporal dim is independent
        if is_wrapped and audio_tensor is not None and format_info is not None:
            try:
                reconstructed = _reconstruct_samples(
                    out_latent_dict["samples"],  # extended video
                    audio_tensor,                 # original audio (unchanged)
                    format_info,
                    debug=debug,
                )
                out_latent_dict["samples"] = reconstructed
                if debug:
                    print(f"  \u00b7 re-wrapped with original audio "
                          f"shape={tuple(audio_tensor.shape)}")
            except Exception as e:
                if debug:
                    print(f"  \u26a0  wrapper reconstruction failed "
                          f"({type(e).__name__}: {e}); outputting plain video")

        if debug:
            print(f"  \u00b7 metadata attached: start_latent_frame="
                  f"{metadata['start_latent_frame']} "
                  f"count={metadata['frame_count_latent']}")

        # ─── Choose latent output based on emit_latent mode ──────────────────
        # passthrough (default): return the user's original input latent.
        # The Guide's value flows through conditioning (positive/negative)
        # and metadata (reference_info). The user wires the input latent
        # directly to their sampler, avoiding any keyframe-pattern issues
        # from the extended latent.
        # extended: return the latent with reference appended. User must
        # crop downstream before decode. Available for users who want to
        # use the in-latent reference mechanism.
        if emit_latent == "passthrough":
            output_latent = original_input_latent
            if debug:
                print(f"  \u00b7 emit_latent=passthrough: returning original "
                      f"input latent unchanged")
        else:
            output_latent = out_latent_dict
            if debug:
                print(f"  \u00b7 emit_latent=extended: returning extended "
                      f"latent (use LikenessCrop before decode)")

        return (out_positive, out_negative, output_latent, metadata)


class LTXLikenessCrop:
    """
    Companion node — crops likeness reference frames off the latent before
    VAE decode. Use after sampling, before VAE Decode, on the latent that
    was processed through LTXLikenessGuide.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "reference_info": ("REFERENCE_INFO",),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "crop_reference"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Crop likeness reference frames off latent before VAE decode. "
        "Inverts LTXLikenessGuide's emit_latent=extended mode. Use only if "
        "Guide is emitting an extended latent. With Guide in passthrough "
        "mode (default), this node is unnecessary — wire the latent "
        "directly to VAE Decode."
    )

    def crop_reference(self, latent, reference_info=None, debug=False):
        # If reference_info isn't wired, this node is a no-op passthrough.
        # Common when Guide is in emit_latent=passthrough mode and the user
        # left LikenessCrop in the workflow for organizational reasons.
        if reference_info is None or not isinstance(reference_info, dict):
            if debug:
                print(f"\u2192 [10S] LikenessCrop: no reference_info wired; "
                      f"passing latent through unchanged")
            return (latent,)

        latent = dict(latent)
        raw_samples = latent["samples"]

        # Detect wrapper
        is_wrapped = not isinstance(raw_samples, torch.Tensor)
        audio_tensor = None
        format_info = None
        if is_wrapped:
            try:
                video_tensor, audio_tensor, format_info = _extract_components(
                    raw_samples, debug=debug
                )
            except Exception as e:
                raise RuntimeError(
                    f"LTXLikenessCrop: cannot extract video: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            video_tensor = raw_samples

        original_length = reference_info.get("original_latent_length")
        current_length = video_tensor.shape[2]

        if original_length is None:
            raise RuntimeError(
                "LTXLikenessCrop: reference_info missing 'original_latent_length' — "
                "is this from an LTXLikenessGuide node?"
            )

        if original_length >= current_length:
            if debug:
                print(f"\u2192 [10S] LikenessCrop: original_length="
                      f"{original_length} \u2265 current={current_length}; "
                      f"no crop needed (passing through)")
            return (latent,)

        if debug:
            print(f"\u2192 [10S] LikenessCrop: cropping {current_length} \u2192 "
                  f"{original_length} latent frames "
                  f"(removed last {current_length - original_length})")

        cropped_video = video_tensor[:, :, :original_length, :, :].contiguous()

        # Also crop noise_mask if present
        cropped_mask = None
        nm = latent.get("noise_mask", None)
        if nm is not None and isinstance(nm, torch.Tensor) and nm.dim() == 5:
            if nm.shape[2] == current_length:
                cropped_mask = nm[:, :, :original_length].contiguous()
            else:
                cropped_mask = nm  # leave as-is if dim mismatch

        # Re-wrap if needed
        if is_wrapped and audio_tensor is not None and format_info is not None:
            try:
                final = _reconstruct_samples(
                    cropped_video, audio_tensor, format_info, debug=debug
                )
            except Exception as e:
                if debug:
                    print(f"  \u26a0  re-wrap failed ({type(e).__name__}: {e}); "
                          f"outputting plain video")
                final = cropped_video
        else:
            final = cropped_video

        out = dict(latent)
        out["samples"] = final
        if cropped_mask is not None:
            out["noise_mask"] = cropped_mask
        elif "noise_mask" in out:
            # Mask shape mismatched and we didn't update — remove rather than
            # carry over an inconsistent mask
            try:
                if out["noise_mask"].shape[2] != original_length:
                    del out["noise_mask"]
            except Exception:
                pass

        if debug:
            print(f"  \u00b7 output video shape={tuple(cropped_video.shape)}")

        return (out,)


NODE_CLASS_MAPPINGS = {
    "LTXLikenessGuide": LTXLikenessGuide,
    "LTXLikenessCrop":  LTXLikenessCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLikenessGuide": "\U0001f9ec LTX Likeness Guide",
    "LTXLikenessCrop":  "\u2702 LTX Likeness Crop",
}
