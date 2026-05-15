"""
LTX Face Detector v1.0

Standalone face detection helper. Takes an IMAGE and emits a bbox string
suitable for wiring into:
  - LTXLikenessGuide's face_bbox_within_reference
  - LTXLikenessAnchor's override_face_bbox or frame_0_bbox
  - LTXFaceAttentionAnchor's face_bbox_norm

Uses MediaPipe Face Detection (best) or OpenCV Haar cascade (fallback).
If neither is installed, returns an empty string and logs a warning.

Decouples detection from any specific node so multiple nodes can share
the same detected bbox, and so users can detect on an image that's
separate from the reference image (e.g., the I2V conditioning frame
when using LikenessAnchor in latent_frame_0 mode).
"""

import torch
import math

# Reuse the detection helper from the guide module
from .latent_likeness_guide import _detect_face_bbox, _tuple_to_bbox_str


class LTXFaceDetector:
    """
    Detect the largest face in an image and emit a normalized bbox string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Padding around detected face bbox as fraction "
                               "(0.15 = 15% expansion). Captures hair/neck "
                               "context for stronger identity preservation.",
                }),
                "fallback_bbox": ("STRING", {
                    "default": "",
                    "tooltip": "Bbox to use if face detection fails or no "
                               "face is found. Format: 'x1,y1,x2,y2' "
                               "normalized 0-1. Empty = empty output on fail.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("face_bbox",)
    FUNCTION = "detect"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Detect face in image, emit normalized bbox string. Wire output to "
        "LikenessGuide's face_bbox_within_reference, LikenessAnchor's "
        "override_face_bbox / frame_0_bbox, or face_anchor's face_bbox_norm. "
        "Decouples detection from per-node logic so multiple nodes can share "
        "the same bbox."
    )

    def detect(self, image, padding=0.15, fallback_bbox="", debug=False):
        # image shape: (B, H, W, 3) in [0, 1]
        if not isinstance(image, torch.Tensor):
            if debug:
                print("\u2192 [10S] FaceDetector: input not a tensor; "
                      "returning fallback")
            return (fallback_bbox,)

        try:
            img = image[0].cpu().clamp(0, 1)
            img_np = (img * 255.0).to(torch.uint8).numpy()
            if debug:
                print(f"\u2192 [10S] FaceDetector: input "
                      f"{img_np.shape[1]}x{img_np.shape[0]} (W x H)")

            bbox = _detect_face_bbox(img_np, padding=padding, debug=debug)
            if bbox is not None:
                bbox_str = _tuple_to_bbox_str(bbox)
                if debug:
                    print(f"\u2192 [10S] FaceDetector: bbox={bbox_str}")
                return (bbox_str,)

            if debug:
                print(f"\u2192 [10S] FaceDetector: no face found; "
                      f"using fallback='{fallback_bbox}'")
            return (fallback_bbox,)

        except Exception as e:
            print(f"\u2192 [10S] FaceDetector: error "
                  f"({type(e).__name__}: {e}); returning fallback")
            return (fallback_bbox,)


NODE_CLASS_MAPPINGS = {
    "LTXFaceDetector": LTXFaceDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXFaceDetector": "\U0001f50d LTX Face Detector",
}
