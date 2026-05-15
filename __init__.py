from .nodes                          import NODE_CLASS_MAPPINGS as _NCM_NODES,         NODE_DISPLAY_NAME_MAPPINGS as _NDM_NODES
from .latent_anchor                  import NODE_CLASS_MAPPINGS as _NCM_LATENT_ANCHOR, NODE_DISPLAY_NAME_MAPPINGS as _NDM_LATENT_ANCHOR
from .latent_anchor_aware            import NODE_CLASS_MAPPINGS as _NCM_AWARE,         NODE_DISPLAY_NAME_MAPPINGS as _NDM_AWARE
from .latent_tiled_sampler           import NODE_CLASS_MAPPINGS as _NCM_TILED_SAMP,    NODE_DISPLAY_NAME_MAPPINGS as _NDM_TILED_SAMP
from .latent_upsampler_tiled         import NODE_CLASS_MAPPINGS as _NCM_UPSAMPLER,     NODE_DISPLAY_NAME_MAPPINGS as _NDM_UPSAMPLER
from .latent_text_amplifier          import NODE_CLASS_MAPPINGS as _NCM_TEXTAMP,       NODE_DISPLAY_NAME_MAPPINGS as _NDM_TEXTAMP
from .latent_likeness_guide          import NODE_CLASS_MAPPINGS as _NCM_LIKE_GUIDE,    NODE_DISPLAY_NAME_MAPPINGS as _NDM_LIKE_GUIDE
from .latent_likeness_anchor         import NODE_CLASS_MAPPINGS as _NCM_LIKE_ANCHOR,   NODE_DISPLAY_NAME_MAPPINGS as _NDM_LIKE_ANCHOR
from .latent_likeness_semantic_clamp import NODE_CLASS_MAPPINGS as _NCM_LIKE_SEM,      NODE_DISPLAY_NAME_MAPPINGS as _NDM_LIKE_SEM
from .latent_action_amplifier        import NODE_CLASS_MAPPINGS as _NCM_ACT_AMP,       NODE_DISPLAY_NAME_MAPPINGS as _NDM_ACT_AMP
from .latent_face_detector           import NODE_CLASS_MAPPINGS as _NCM_FACE_DETECT,   NODE_DISPLAY_NAME_MAPPINGS as _NDM_FACE_DETECT
from .model_inspector                import NODE_CLASS_MAPPINGS as _NCM_INSPECT,       NODE_DISPLAY_NAME_MAPPINGS as _NDM_INSPECT

NODE_CLASS_MAPPINGS = {
    **_NCM_NODES,
    **_NCM_LATENT_ANCHOR,
    **_NCM_AWARE,
    **_NCM_TILED_SAMP,
    **_NCM_UPSAMPLER,
    **_NCM_TEXTAMP,
    **_NCM_LIKE_GUIDE,
    **_NCM_LIKE_ANCHOR,
    **_NCM_LIKE_SEM,
    **_NCM_ACT_AMP,
    **_NCM_FACE_DETECT,
    **_NCM_INSPECT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_NDM_NODES,
    **_NDM_LATENT_ANCHOR,
    **_NDM_AWARE,
    **_NDM_TILED_SAMP,
    **_NDM_UPSAMPLER,
    **_NDM_TEXTAMP,
    **_NDM_LIKE_GUIDE,
    **_NDM_LIKE_ANCHOR,
    **_NDM_LIKE_SEM,
    **_NDM_ACT_AMP,
    **_NDM_FACE_DETECT,
    **_NDM_INSPECT,
}

__version__ = "1.6.0"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']
