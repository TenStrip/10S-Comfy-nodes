"""
LTX Model Inspector — diagnostic for finding attention modules.

Use when ComfyUI's patch protocol is bypassed by the model's attention
implementation (registration succeeds but hook never fires). Walks the
diffusion model, prints structure, and (optionally) attaches one-shot
forward-trace hooks so the next sampling pass prints input/output shapes
for chosen modules.

Three things this can do, controlled by parameters:
  • Top-level dump of the diffusion backbone's children (param counts +
    container lengths). Set print_top_level=True.
  • Name-filter listing of all modules whose dotted path or class name
    contains a substring. Use for finding candidate modules.
  • Focus mode: print just the *immediate children* of one specific
    module path (e.g. "transformer_blocks.0"). Cleaner than a recursive
    name-filter dump when you want to see a single block's structure.
  • Trace hooks on modules whose path contains a target string,
    auto-quieting after N calls.

Pass-through: returns the model unmodified (clones + adds hooks that
auto-disable after their call budget).
"""

import torch


def _resolve_diffusion_model(m):
    """Find the actual nn.Module backbone, trying several conventions."""
    for path in ("diffusion_model", "model", "transformer", "dit", "net"):
        obj = getattr(m.model, path, None)
        if obj is not None and hasattr(obj, "named_modules"):
            return obj, path
    if hasattr(m.model, "named_modules"):
        return m.model, "model"
    return None, None


def _resolve_path(root, dotted_path):
    """Walk a dotted path from root; return None if any segment missing."""
    cur = root
    if not dotted_path:
        return cur
    for part in dotted_path.split("."):
        if part.isdigit() and hasattr(cur, "__getitem__"):
            try:
                cur = cur[int(part)]
                continue
            except Exception:
                return None
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _safe_shape(x):
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            return f"{type(x).__name__}[0]"
        head = x[0]
        if torch.is_tensor(head):
            return f"{tuple(head.shape)} (+{len(x)-1})"
        return f"{type(x).__name__}[{len(x)}]"
    if isinstance(x, dict):
        return f"dict({list(x.keys())[:4]})"
    return type(x).__name__


class LTXModelInspector:
    """
    Diagnostic node. Inspects the diffusion model and (optionally) traces
    forward shapes. Use to find modules to attach forward hooks to.

    Inputs:
      name_filter:               Substring filter for the listing block.
                                 Empty disables listing.
      focus_path:                Dotted path under the backbone (e.g.
                                 'transformer_blocks.0'). If set, prints the
                                 *immediate children* of that path with
                                 param counts and container lengths.
      max_modules_listed:        Cap on listed name-filter matches.
      trace_path_contains:       If non-empty, attach forward-trace hooks
                                 to every module whose dotted path contains
                                 this substring. Empty disables tracing.
      max_modules_traced:        Cap on number of trace hooks attached.
      trace_calls_per_module:    Number of forward calls per module to log
                                 (then auto-quiets).
      print_top_level:           Print top-level children of the backbone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "name_filter":             ("STRING",  {"default": ""}),
                "focus_path":              ("STRING",  {"default": "transformer_blocks.0"}),
                "max_modules_listed":      ("INT",     {"default": 100, "min": 1, "max": 4000, "step": 1}),
                "trace_path_contains":     ("STRING",  {"default": ""}),
                "max_modules_traced":      ("INT",     {"default": 8,   "min": 0, "max": 200,  "step": 1}),
                "trace_calls_per_module":  ("INT",     {"default": 1,   "min": 1, "max": 10,   "step": 1}),
                "print_top_level":         ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inspect"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = "Inspect diffusion model: top-level / name-filter / focus_path / trace hooks."

    def inspect(self, model,
                name_filter="", focus_path="transformer_blocks.0",
                max_modules_listed=100,
                trace_path_contains="", max_modules_traced=8,
                trace_calls_per_module=1,
                print_top_level=False):

        m = model.clone()
        backbone, backbone_path = _resolve_diffusion_model(m)
        if backbone is None:
            print("\u2192 [10S] Inspector: could not locate diffusion backbone.")
            try:
                print(f"  m.model attrs: {[a for a in dir(m.model) if not a.startswith('_')][:30]}")
            except Exception:
                pass
            return (m,)

        print(f"\u2192 [10S] Inspector: backbone at m.model.{backbone_path} "
              f"({type(backbone).__name__})")

        # 1. Top-level dump (optional)
        if print_top_level:
            print(f"\u2192 [10S] Inspector: top-level children:")
            for name, child in backbone.named_children():
                try:
                    n_params = sum(p.numel() for p in child.parameters())
                except Exception:
                    n_params = -1
                ext = ""
                if hasattr(child, "__len__"):
                    try:
                        ext = f" len={len(child)}"
                    except Exception:
                        pass
                print(f"    {name:30s} {type(child).__name__:36s} params={n_params:>14,}{ext}")

        # 2. Focus path: immediate children of one specific module
        if focus_path:
            target = _resolve_path(backbone, focus_path)
            if target is None:
                print(f"\u2192 [10S] Inspector: focus_path '{focus_path}' not found under backbone.")
            else:
                print(f"\u2192 [10S] Inspector: immediate children of '{focus_path}' "
                      f"({type(target).__name__}):")
                children = list(target.named_children())
                if not children:
                    print(f"    (no named children — leaf module)")
                for name, child in children:
                    try:
                        n_params = sum(p.numel() for p in child.parameters())
                    except Exception:
                        n_params = -1
                    ext = ""
                    if hasattr(child, "__len__"):
                        try:
                            ext = f" len={len(child)}"
                        except Exception:
                            pass
                    print(f"    {name:30s} {type(child).__name__:36s} params={n_params:>14,}{ext}")

        # 3. Name-filter listing (optional)
        if name_filter:
            f = name_filter.lower()
            matched = []
            for name, module in backbone.named_modules():
                if not name:
                    continue
                if f in name.lower() or f in type(module).__name__.lower():
                    matched.append((name, module))
            print(f"\u2192 [10S] Inspector: {len(matched)} module(s) match name_filter '{name_filter}'")
            for name, module in matched[:max_modules_listed]:
                try:
                    np = sum(p.numel() for p in module.parameters(recurse=False))
                except Exception:
                    np = -1
                print(f"    {name:80s} {type(module).__name__:24s} own_params={np}")
            if len(matched) > max_modules_listed:
                print(f"    ... ({len(matched) - max_modules_listed} more truncated)")

        # 4. Trace hooks (optional)
        if trace_path_contains and max_modules_traced > 0:
            tf = trace_path_contains.lower()
            trace_targets = []
            for name, module in backbone.named_modules():
                if not name:
                    continue
                if tf in name.lower():
                    trace_targets.append((name, module))
                    if len(trace_targets) >= max_modules_traced:
                        break

            print(f"\u2192 [10S] Inspector: attaching trace hooks on {len(trace_targets)} "
                  f"module(s) matching '{trace_path_contains}' "
                  f"(quiet after {trace_calls_per_module} call(s) each):")

            trace_state = {}

            def make_hook(mod_name):
                trace_state[mod_name] = 0
                def hook(module, inputs, output):
                    if trace_state[mod_name] >= trace_calls_per_module:
                        return None
                    in_str = ", ".join(_safe_shape(x) for x in inputs) if inputs else "()"
                    out_str = _safe_shape(output)
                    print(f"  TRACE {mod_name}: in=({in_str}) out={out_str}")
                    trace_state[mod_name] += 1
                    return None
                return hook

            for name, module in trace_targets:
                try:
                    module.register_forward_hook(make_hook(name))
                    print(f"    + hook on {name} ({type(module).__name__})")
                except Exception as e:
                    print(f"    ! failed to hook {name}: {type(e).__name__}: {e}")
            print(f"\u2192 [10S] Inspector: hooks live; will print on next sampling pass.")
            print(f"  NOTE: hooks persist on the model object until ComfyUI reloads it.")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXModelInspector": LTXModelInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXModelInspector": "\U0001f50d LTX Model Inspector",
}
