from .nodes                    import NODE_CLASS_MAPPINGS as _NCM_NODES,         NODE_DISPLAY_NAME_MAPPINGS as _NDM_NODES
from .face_anchor              import NODE_CLASS_MAPPINGS as _NCM_FACE_ANCHOR,   NODE_DISPLAY_NAME_MAPPINGS as _NDM_FACE_ANCHOR
from .latent_anchor            import NODE_CLASS_MAPPINGS as _NCM_LATENT_ANCHOR, NODE_DISPLAY_NAME_MAPPINGS as _NDM_LATENT_ANCHOR
from .latent_anchor_aware      import NODE_CLASS_MAPPINGS as _NCM_AWARE,         NODE_DISPLAY_NAME_MAPPINGS as _NDM_AWARE
from .latent_upsampler_tiled   import NODE_CLASS_MAPPINGS as _NCM_UPSAMPLER,     NODE_DISPLAY_NAME_MAPPINGS as _NDM_UPSAMPLER
from .latent_color_restore     import NODE_CLASS_MAPPINGS as _NCM_COLOR,         NODE_DISPLAY_NAME_MAPPINGS as _NDM_COLOR
from .latent_outlier_suppress  import NODE_CLASS_MAPPINGS as _NCM_OUTLIER,       NODE_DISPLAY_NAME_MAPPINGS as _NDM_OUTLIER
from .latent_text_amplifier    import NODE_CLASS_MAPPINGS as _NCM_TEXTAMP,       NODE_DISPLAY_NAME_MAPPINGS as _NDM_TEXTAMP
from .model_inspector          import NODE_CLASS_MAPPINGS as _NCM_INSPECT,       NODE_DISPLAY_NAME_MAPPINGS as _NDM_INSPECT

NODE_CLASS_MAPPINGS = {
    **_NCM_NODES,
    **_NCM_FACE_ANCHOR,
    **_NCM_LATENT_ANCHOR,
    **_NCM_AWARE,
    **_NCM_UPSAMPLER,
    **_NCM_COLOR,
    **_NCM_OUTLIER,
    **_NCM_TEXTAMP,
    **_NCM_INSPECT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_NDM_NODES,
    **_NDM_FACE_ANCHOR,
    **_NDM_LATENT_ANCHOR,
    **_NDM_AWARE,
    **_NDM_UPSAMPLER,
    **_NDM_COLOR,
    **_NDM_OUTLIER,
    **_NDM_TEXTAMP,
    **_NDM_INSPECT,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


__version__ = "1.0.0"