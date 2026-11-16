"""CPU-only tests of immutable adapter-only checkpoint integrity."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from iq_transfer.glm_first_overlay_checkpoint import (
    GLMOverlayCheckpointError,
    load_glm_first_overlay,
    save_glm_first_overlay,
)
from test_glm_first_mamba_hybrid import (
    CONTRACT,
    ToyGLM, ToyMambaModel, wrap_preloaded_official_models,
)

GLM_REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAMBA_REV = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
TOKENIZER_SHA = "a" * 64


class GLMFirstOverlayCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.teacher = ToyGLM().eval()
        self.donor = ToyMambaModel().eval()
        self.hybrid = wrap_preloaded_official_models(
            self.teacher, self.donor, contract=CONTRACT
        )
        self.original_embedding = self.teacher.model.embed_tokens.weight.detach().clone()
        self.original_mixer = self.donor.backbone.layers[0].mixer.proj.weight.detach().clone()

    def save(self, dest):
        return save_glm_first_overlay(
            self.hybrid, dest,
            glm_revision=GLM_REV,
            mamba_revision=MAMBA_REV,
            tokenizer_mapping_sha256=TOKENIZER_SHA,
            training_steps=3,
        )

    def load(self, dest, **overrides):
        args = {
            "glm_revision": GLM_REV,
            "mamba_revision": MAMBA_REV,
            "tokenizer_mapping_sha256": TOKENIZER_SHA,
        }
        args.update(overrides)
        return load_glm_first_overlay(self.hybrid, dest, **args)

    def test_round_trip_and_freezes_original_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "overlay"
            with torch.no_grad():
                for branch in self.hybrid.branches:
                    branch.gate.fill_(0.125)
            original = [
                {key: value.detach().clone() for key, value in (
                    ("gate", b.gate),
                    ("in_adapter.weight", b.in_adapter.weight),
                    ("out_adapter.weight", b.out_adapter.weight),
                )}
                for b in self.hybrid.branches
            ]
            manifest = self.save(target)
            self.assertFalse(manifest["inference_ready"])
            self.assertEqual(len(manifest["shards"]), CONTRACT.mamba_layers)
            with torch.no_grad():
                for b in self.hybrid.branches:
                    b.gate.zero_()
                    b.in_adapter.weight.zero_()
                    b.out_adapter.weight.zero_()
            self.load(target)
            for branch, expected in zip(self.hybrid.branches, original):
                torch.testing.assert_close(branch.gate, expected["gate"], atol=0, rtol=0)
                torch.testing.assert_close(
                    branch.in_adapter.weight, expected["in_adapter.weight"], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    branch.out_adapter.weight, expected["out_adapter.weight"], atol=0, rtol=0
                )
            torch.testing.assert_close(
                self.teacher.model.embed_tokens.weight, self.original_embedding, atol=0, rtol=0
            )
            torch.testing.assert_close(
                self.donor.backbone.layers[0].mixer.proj.weight, self.original_mixer,
                atol=0, rtol=0
            )

    def test_reject_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "overlay"
            self.save(target)
            with self.assertRaisesRegex(GLMOverlayCheckpointError, "overwrite refused"):
                self.save(target)

    def test_reject_tampered_shard_before_any_adapter_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "overlay"
            self.save(target)
            with torch.no_grad():
                self.hybrid.branches[0].gate.fill_(0.5)
            with (target / "overlay-002.safetensors").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaisesRegex(GLMOverlayCheckpointError, "corrupt"):
                self.load(target)
            self.assertEqual(float(self.hybrid.branches[0].gate), 0.5)

    def test_reject_wrong_donor_or_tokenizer_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "overlay"
            self.save(target)
            with self.assertRaisesRegex(GLMOverlayCheckpointError, "provenance"):
                self.load(target, mamba_revision="e" * 40)
            with self.assertRaisesRegex(GLMOverlayCheckpointError, "provenance"):
                self.load(target, tokenizer_mapping_sha256="e" * 64)

    def test_manifest_absence_blocks_incomplete_package(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "overlay"
            self.save(target)
            (target / "overlay_manifest.json").unlink()
            with self.assertRaisesRegex(GLMOverlayCheckpointError, "manifest"):
                self.load(target)


if __name__ == "__main__":
    unittest.main()
