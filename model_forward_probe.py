"""
model_forward_probe.py

Diagnostic node that inspects a MODEL's forward signature, class hierarchy,
and runtime kwarg handling. Built to verify whether Echo's memory_video /
paired_audio_memory kwargs flow through Comfy's model wrapper to the
actual diffusion module's forward — but useful for any kwargs investigation.

Outputs:
  - the MODEL unchanged (passthrough so it can be wired mid-graph)
  - a STRING report with the probe results

Probes performed:
  1. Class hierarchy from the ModelPatcher down to the innermost nn.Module,
     descending via .model and .diffusion_model attributes
  2. inspect.signature() on the innermost module's forward() method
  3. Presence check for user-specified kwarg names
  4. Listing of any forward kwargs containing case-insensitive substrings
     ("memory", "shot", "prior", "bank")
  5. Detection of **kwargs catch-all in the signature
  6. Existence of model.model_options and Comfy's transformer_options
     conventional path
  7. Attempt to read source file path for the forward method (helps
     identify whether it's stock LTX2 or Echo-aware)
"""

import inspect


class ModelForwardProbe:
    """Inspects a model's forward signature for diagnostic purposes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "kwargs_to_check": ("STRING", {
                    "default": "memory_video, paired_audio_memory, "
                               "memory_audio, prior_video, prior_audio",
                    "multiline": False,
                    "tooltip": "Comma-separated kwarg names to check for "
                               "in the innermost forward() signature. The "
                               "defaults cover the JoyAI-Echo memory bank "
                               "kwargs and likely alternative names.",
                }),
                "substring_search": ("STRING", {
                    "default": "memory, shot, prior, bank, paired",
                    "multiline": False,
                    "tooltip": "Comma-separated substrings to search for "
                               "in forward signature kwarg names (case-"
                               "insensitive). Any kwarg containing one of "
                               "these is reported.",
                }),
                "max_depth": ("INT", {
                    "default": 6, "min": 1, "max": 12,
                    "tooltip": "How many .model / .diffusion_model nesting "
                               "levels to traverse before stopping.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "probe"
    CATEGORY = "10S Nodes/Debug"
    DESCRIPTION = (
        "Inspects a MODEL's class hierarchy and forward signature. "
        "Designed to verify whether memory_video / paired_audio_memory "
        "(JoyAI-Echo memory bank kwargs) are accepted by the diffusion "
        "module's forward — but works for any kwargs check. Output is the "
        "input MODEL unchanged + a STRING report. Also prints to console."
    )

    def probe(self, model, kwargs_to_check="memory_video, paired_audio_memory",
              substring_search="memory, shot, prior, bank, paired",
              max_depth=6):
        lines = []
        lines.append("=" * 70)
        lines.append("[10S] Model Forward Probe")
        lines.append("=" * 70)

        # ── 1. Walk hierarchy ─────────────────────────────────────────────
        lines.append("\nHierarchy (descending via .model / .diffusion_model):")
        obj = model
        path_taken = []
        for depth in range(max_depth):
            cls = obj.__class__
            cls_name = cls.__name__
            mod_name = cls.__module__
            lines.append(f"  [{depth}] {mod_name}.{cls_name}")
            path_taken.append((depth, mod_name, cls_name))

            # Try .diffusion_model first (more specific), then .model
            if hasattr(obj, "diffusion_model") and obj.diffusion_model is not None:
                next_obj = obj.diffusion_model
                if next_obj is obj:
                    break
                lines.append(f"      \u2193 .diffusion_model")
                obj = next_obj
            elif hasattr(obj, "model") and obj.model is not None:
                next_obj = obj.model
                if next_obj is obj:
                    break
                lines.append(f"      \u2193 .model")
                obj = next_obj
            else:
                break

        innermost = obj
        lines.append(f"\nInnermost: {innermost.__class__.__module__}."
                     f"{innermost.__class__.__name__}")

        # ── 2. Inspect forward signature ──────────────────────────────────
        forward_fn = getattr(innermost, "forward", None)
        if forward_fn is None:
            lines.append("\n\u26a0 Innermost object has no .forward() method.")
        else:
            # Try to get source file
            try:
                src_file = inspect.getsourcefile(forward_fn)
                src_line = inspect.getsourcelines(forward_fn)[1]
                lines.append(f"forward() defined at: {src_file}:{src_line}")
            except (TypeError, OSError):
                lines.append("forward() source location unavailable "
                             "(possibly C extension)")

            # Try signature
            try:
                sig = inspect.signature(forward_fn)
                params = sig.parameters
                lines.append(f"\nforward() parameters ({len(params)} total):")
                for name, p in params.items():
                    kind_str = {
                        inspect.Parameter.POSITIONAL_ONLY: "pos-only",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos/kw",
                        inspect.Parameter.VAR_POSITIONAL: "*args",
                        inspect.Parameter.KEYWORD_ONLY: "kw-only",
                        inspect.Parameter.VAR_KEYWORD: "**kwargs",
                    }.get(p.kind, str(p.kind))
                    default_str = ""
                    if p.default is not inspect.Parameter.empty:
                        try:
                            default_str = f" = {p.default!r}"
                        except Exception:
                            default_str = " = <unprintable>"
                        if len(default_str) > 80:
                            default_str = default_str[:77] + "..."
                    annot_str = ""
                    if p.annotation is not inspect.Parameter.empty:
                        annot_str = f": {p.annotation}"
                        if len(annot_str) > 60:
                            annot_str = annot_str[:57] + "..."
                    lines.append(f"  [{kind_str:10s}] {name}{annot_str}{default_str}")

                param_names_lower = {n.lower() for n in params.keys()}
                has_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in params.values()
                )

                # ── 3. Check requested kwargs ─────────────────────────────
                requested = [k.strip() for k in kwargs_to_check.split(",")
                             if k.strip()]
                lines.append("\nRequested kwarg presence:")
                for k in requested:
                    present = k.lower() in param_names_lower
                    marker = "\u2713" if present else "\u2717"
                    lines.append(f"  {marker} {k}")

                # ── 4. Substring search ───────────────────────────────────
                substrings = [s.strip().lower() for s in
                               substring_search.split(",") if s.strip()]
                if substrings:
                    lines.append(f"\nKwargs matching substrings "
                                 f"{substrings}:")
                    matches = []
                    for name in params.keys():
                        lower = name.lower()
                        for sub in substrings:
                            if sub in lower:
                                matches.append((name, sub))
                                break
                    if matches:
                        for name, sub in matches:
                            lines.append(f"  \u2713 {name}  (matched '{sub}')")
                    else:
                        lines.append("  (none)")

                # ── 5. **kwargs catch-all note ────────────────────────────
                if has_var_kwargs:
                    lines.append("\n\u26a0 forward has **kwargs catch-all.")
                    lines.append("  Memory kwargs MAY be accepted at runtime "
                                 "without erroring,")
                    lines.append("  but they're only meaningful if the body "
                                 "explicitly handles them.")
                    lines.append("  Check the source file above to confirm.")
                else:
                    lines.append("\nNo **kwargs catch-all in signature.")
                    lines.append("  Unrecognized kwargs would raise TypeError.")

            except (ValueError, TypeError) as e:
                lines.append(f"\nCouldn't inspect signature: "
                             f"{type(e).__name__}: {e}")

        # ── 6. ModelPatcher / model_options check ─────────────────────────
        lines.append("\n" + "-" * 70)
        lines.append("ModelPatcher conventions:")
        if hasattr(model, "model_options"):
            mo = model.model_options
            lines.append(f"  model.model_options: present "
                         f"(type={type(mo).__name__})")
            if isinstance(mo, dict):
                lines.append(f"    keys: {list(mo.keys())}")
                if "transformer_options" in mo:
                    to = mo["transformer_options"]
                    if isinstance(to, dict):
                        lines.append(f"    transformer_options keys: "
                                     f"{list(to.keys())}")
        else:
            lines.append("  model.model_options: not present")

        if hasattr(model, "add_object_patch"):
            lines.append("  model.add_object_patch: present "
                         "(can patch attributes at runtime)")
        if hasattr(model, "set_model_patch_replace"):
            lines.append("  model.set_model_patch_replace: present "
                         "(can replace internal blocks)")

        lines.append("=" * 70)

        report = "\n".join(lines)
        print(report)
        return (model, report)


NODE_CLASS_MAPPINGS = {
    "ModelForwardProbe": ModelForwardProbe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelForwardProbe": "\U0001f52c Model Forward Probe",
}
