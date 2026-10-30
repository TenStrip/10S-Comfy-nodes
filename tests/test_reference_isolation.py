"""CPU-only reference isolation regressions; no model weights or generation.

Run with ComfyUI's Python: python -B -m unittest discover -s tests -v
"""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn


NODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_DIR.parents[1]))

# Select CPU before model_patcher imports ComfyUI's device management.
import comfy.cli_args

comfy.cli_args.args.cpu = True
from comfy.model_patcher import ModelPatcher
from comfy.ldm.lightricks.symmetric_patchifier import (
    SymmetricPatchifier,
    latent_to_pixel_coords,
)


def load_module(filename):
    # Import only the module under test, not the custom-node pack's __init__.
    spec = importlib.util.spec_from_file_location(
        "reference_isolation_" + Path(filename).stem, NODE_DIR / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conditioning = load_module("ltx_reference_conditioning.py")
enable = load_module("ltx_reference_enable.py")


class FakeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchifier = SymmetricPatchifier(1, start_end=True)
        self.patchify_proj = nn.Identity()
        self.vae_scale_factors = (8, 32, 32)
        self.causal_temporal_positioning = True


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = FakeDiffusion()

    def process_latent_in(self, latent):
        return latent


class FakeVAE:
    def encode(self, images):
        # Synthetic frame values stand in for encoded video latents.
        values = images.mean(dim=(1, 2, 3)).reshape(1, 1, -1, 1, 1)
        return values.expand(
            1, 128, images.shape[0], images.shape[1] // 32, images.shape[2] // 32
        ).clone()


def native_process_input(model, x, keyframe_idxs, denoise_mask, **kwargs):
    tokens, coords = model.patchifier.patchify(x)
    coords = latent_to_pixel_coords(
        coords, model.vae_scale_factors, model.causal_temporal_positioning
    )
    return [model.patchify_proj(tokens)], [coords], {"orig_shape": tuple(x.shape)}


class ReferenceIsolationTests(unittest.TestCase):
    def setUp(self):
        cpu = torch.device("cpu")
        self.base = ModelPatcher(FakeBase(), cpu, cpu)
        self.dm = self.base.model.diffusion_model
        self.vae = FakeVAE()
        self.target = torch.zeros(1, 128, 2, 2, 2)
        for name, replacement in (
            ("_ORIGINAL_PROCESS_INPUT", native_process_input),
            ("_import_comfy", lambda: (None, None, latent_to_pixel_coords)),
        ):
            mocked = patch.object(enable, name, replacement)
            mocked.start()
            self.addCleanup(mocked.stop)

    @staticmethod
    def image(value):
        return torch.full((1, 64, 64, 3), value)

    @staticmethod
    def reference(value, frames=1):
        return torch.full((1, 128, frames, 2, 2), value)

    @staticmethod
    def options(model):
        return model.model_options["transformer_options"]

    def attach(self, model, value=0.25, strength=1.0):
        return conditioning.LTXReferenceConditioning().attach(
            model, self.vae, self.image(value), strength=strength
        )[0]

    def run_input(self, options):
        # ComfyUI's LTXBaseModel._forward unpacks transformer_options this way.
        return enable._patched_process_input(
            self.dm, self.target, None, None, **options
        )

    def assert_no_reference(self, result):
        native = native_process_input(self.dm, self.target, None, None)
        torch.testing.assert_close(result[0][0], native[0][0], rtol=0, atol=0)
        torch.testing.assert_close(result[1][0], native[1][0], rtol=0, atol=0)
        self.assertNotIn("reference_seq_len", result[2])
        self.assertEqual(self.dm._pending_ref_seq_len, 0)

    def assert_reference(self, result, reference):
        ref_tokens, _ = self.dm.patchifier.patchify(reference)
        target_tokens, _ = self.dm.patchifier.patchify(self.target)
        expected = torch.cat((ref_tokens, target_tokens), dim=1)
        torch.testing.assert_close(result[0][0], expected, rtol=0, atol=0)
        self.assertEqual(result[2]["reference_seq_len"], ref_tokens.shape[1])
        self.assertEqual(self.dm._pending_ref_seq_len, ref_tokens.shape[1])
        self.assertEqual(result[1][0].shape[2], expected.shape[1])

    def assert_no_shared_reference(self):
        self.assertFalse(hasattr(self.dm, "_ltx_reference_latent"))
        self.assertFalse(hasattr(self.dm, "_echo_memory_video"))

    def test_cached_strength_zero_cannot_inherit_later_reference(self):
        cached_zero = self.attach(self.base, strength=0.0)
        active = self.attach(self.base)

        self.assert_no_reference(self.run_input(self.options(cached_zero)))
        for key in ("reference_latent", "memory_video"):
            self.assertIn(key, self.options(cached_zero))
            self.assertIsNone(self.options(cached_zero)[key])
        self.assert_reference(
            self.run_input(self.options(active)), self.options(active)["reference_latent"]
        )
        self.assertEqual(self.options(self.base), {})
        self.assert_no_shared_reference()

    def test_sibling_references_are_clone_local(self):
        first = self.attach(self.base, 0.25)
        first_ref = self.options(first)["reference_latent"]
        second = self.attach(self.base, 0.75)
        second_ref = self.options(second)["reference_latent"]

        self.assertIs(first.model, second.model)
        self.assertIsNot(self.options(first), self.options(second))
        self.assertIs(self.options(first)["reference_latent"], first_ref)
        self.assertIs(self.options(first)["memory_video"], first_ref)
        self.assertIs(self.options(second)["memory_video"], second_ref)
        self.assertFalse(torch.equal(first_ref, second_ref))
        self.assert_reference(self.run_input(self.options(first)), first_ref)
        self.assert_reference(self.run_input(self.options(second)), second_ref)
        self.assertEqual(self.options(self.base), {})
        self.assert_no_shared_reference()

    def test_zero_and_bypass_preserve_active_sibling_and_shared_state(self):
        active = self.attach(self.base)
        reference = self.options(active)["reference_latent"]
        legacy = self.reference(0.9)
        actions = (
            ("strength_zero", lambda: self.attach(active, strength=0.0)),
            ("bypass", lambda: conditioning.LTXReferenceBypass().bypass(active)[0]),
        )
        for name, action in actions:
            with self.subTest(action=name):
                self.dm._ltx_reference_latent = legacy
                self.dm._echo_memory_video = legacy
                self.dm._pending_ref_seq_len = 123
                self.dm._pending_ref_frames = 7
                disabled = action()
                self.assertIs(self.dm._ltx_reference_latent, legacy)
                self.assertIs(self.dm._echo_memory_video, legacy)
                self.assertEqual(self.dm._pending_ref_seq_len, 123)
                self.assertEqual(self.dm._pending_ref_frames, 7)
                self.assertIs(self.options(active)["reference_latent"], reference)
                self.assert_no_reference(self.run_input(self.options(disabled)))
                self.assert_reference(self.run_input(self.options(active)), reference)

    def test_sequence_cached_zero_and_active_paths_are_isolated(self):
        node = conditioning.LTXReferenceSequenceConditioning()
        cached_zero = node.attach(
            self.base, self.vae, None, strength=0.0
        )[0]
        frames = torch.cat([self.image(value) for value in (0.2, 0.4, 0.6)])
        active = node.attach(
            self.base, self.vae, frames, start_frame=1, num_frames=2
        )[0]
        reference = self.options(active)["reference_latent"]

        self.assertEqual(tuple(reference.shape), (1, 128, 2, 2, 2))
        self.assertIs(self.options(active)["memory_video"], reference)
        self.assert_no_reference(self.run_input(self.options(cached_zero)))
        self.assert_reference(self.run_input(self.options(active)), reference)
        disabled = node.attach(active, self.vae, None, strength=0.0)[0]
        self.assertIs(self.options(active)["reference_latent"], reference)
        self.assert_no_reference(self.run_input(self.options(disabled)))
        self.assert_reference(self.run_input(self.options(active)), reference)
        self.assertEqual(self.options(self.base), {})
        self.assert_no_shared_reference()

    def test_explicit_none_blocks_shared_attribute_fallbacks(self):
        reference = self.reference(0.75)
        self.dm._ltx_reference_latent = reference
        self.dm._echo_memory_video = reference
        cases = (
            {"reference_latent": None},
            {"memory_video": None},
            {"transformer_options": {"reference_latent": None}},
            {"transformer_options": {"memory_video": None}},
            {"reference_latent": None, "memory_video": None,
             "transformer_options": {"reference_latent": None,
                                     "memory_video": None}},
        )
        for index, options in enumerate(cases):
            with self.subTest(priority=index):
                self.assert_no_reference(self.run_input(options))

    def test_missing_keys_still_allow_legacy_attributes(self):
        modern = self.reference(0.25)
        legacy = self.reference(0.75)
        self.dm._ltx_reference_latent = modern
        self.dm._echo_memory_video = legacy
        self.assert_reference(self.run_input({}), modern)
        self.dm._ltx_reference_latent = None
        self.assert_reference(
            self.run_input({"transformer_options": {"unrelated": None}}), legacy
        )
        del self.dm._ltx_reference_latent
        self.assert_reference(self.run_input({}), legacy)

    def test_explicit_reference_precedence(self):
        expected = self.reference(0.25)
        lower_priority = self.reference(0.75)
        self.dm._ltx_reference_latent = lower_priority
        self.dm._echo_memory_video = lower_priority
        cases = (
            {"reference_latent": expected, "memory_video": lower_priority,
             "transformer_options": {"reference_latent": lower_priority,
                                     "memory_video": lower_priority}},
            {"memory_video": expected,
             "transformer_options": {"reference_latent": lower_priority,
                                     "memory_video": lower_priority}},
            {"transformer_options": {"reference_latent": expected,
                                     "memory_video": lower_priority}},
            {"transformer_options": {"memory_video": expected}},
            {"reference_latent": None, "memory_video": expected},
            {"transformer_options": {"reference_latent": None,
                                     "memory_video": expected}},
            {"reference_latent": None, "memory_video": None,
             "transformer_options": {"reference_latent": expected}},
        )
        for index, options in enumerate(cases):
            with self.subTest(priority=index):
                self.assert_reference(self.run_input(options), expected)

    def test_legacy_key_can_reenable_a_bypassed_clone(self):
        active = self.attach(self.base)
        disabled = conditioning.LTXReferenceBypass().bypass(active)[0]
        reenabled = disabled.clone()
        reference = self.reference(0.75)
        # Older companion nodes may set only the legacy key.
        self.options(reenabled)["memory_video"] = reference
        self.assertIsNone(self.options(reenabled)["reference_latent"])
        self.assert_reference(self.run_input(self.options(reenabled)), reference)
        self.assert_no_reference(self.run_input(self.options(disabled)))
        self.assert_reference(
            self.run_input(self.options(active)), self.options(active)["reference_latent"]
        )

    def test_conditioning_nodes_can_reenable_a_bypassed_clone(self):
        disabled = conditioning.LTXReferenceBypass().bypass(self.base)[0]
        single = self.attach(disabled, 0.25)
        frames = torch.cat((self.image(0.5), self.image(0.75)))
        sequence = conditioning.LTXReferenceSequenceConditioning().attach(
            disabled, self.vae, frames, num_frames=2
        )[0]
        for active in (single, sequence):
            self.assert_reference(
                self.run_input(self.options(active)),
                self.options(active)["reference_latent"],
            )
            self.assert_no_reference(self.run_input(self.options(disabled)))
        self.assert_no_shared_reference()

    def test_active_disabled_active_resets_pending_prefix_state(self):
        reference = self.reference(0.25, frames=2)
        active = {"reference_latent": reference, "reference_source_id": 2.0,
                  "reference_phase_scale": 0.5}
        self.assert_reference(self.run_input(active), reference)
        self.assertEqual(self.dm._pending_source_id, 2.0)
        self.assertEqual(self.dm._pending_phase_scale, 0.5)
        self.assertEqual(len(self.dm._pending_ref_segments), 2)

        self.dm._ltx_reference_latent = reference
        disabled = {"transformer_options": {"reference_latent": None}}
        self.assert_no_reference(self.run_input(disabled))
        self.assertEqual(self.dm._pending_source_id, 0.0)
        self.assertEqual(self.dm._pending_phase_scale, 0.0)
        self.assertIsNone(self.dm._pending_ref_segments)

        next_reference = self.reference(0.75)
        self.assert_reference(
            self.run_input({"reference_latent": next_reference}), next_reference
        )
        self.assertEqual(self.dm._pending_source_id, 0.0)
        self.assertIsNone(self.dm._pending_ref_segments)


if __name__ == "__main__":
    unittest.main()
