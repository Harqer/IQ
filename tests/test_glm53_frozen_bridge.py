"""Small-CPU contract tests for exact frozen GLM expert weights and trainable IQ adapters."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from iq_transfer.glm53_frozen_bridge import (
    FrozenExpertBridgeError,
    FrozenGLMExpertBridge,
    train_frozen_bridge,
)
from scripts.probe_glm53_frozen_bridge import _load_document_ids


class FrozenExpertBridgeTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(135)
        self.donor = {
            "gate": torch.randn(5, 6, generator=self.generator).to(torch.bfloat16),
            "up": torch.randn(5, 6, generator=self.generator).to(torch.bfloat16),
            "down": torch.randn(6, 5, generator=self.generator).to(torch.bfloat16),
        }
        self.basis = torch.eye(6, dtype=torch.float32)[:, :3].contiguous()

    def test_original_donor_tensors_frozen_and_no_checkpoint_copies(self):
        model = FrozenGLMExpertBridge(self.donor, self.basis, rank=2, seed=17)
        z = torch.randn(8, 3, generator=self.generator)
        expected = model.donor_forward(z @ self.basis.T) @ self.basis
        self.assertTrue(torch.allclose(model(z), expected, atol=1e-6))
        self.assertEqual(model.trainable_count, 2 * 2 * (3 + 6))
        self.assertEqual(set(model.adapter_state()), {
            "input_a", "input_b", "output_a", "output_b"
        })
        self.assertTrue(all(not getattr(model, name).requires_grad
                            for name in ("gate", "up", "down", "basis")))
        self.assertTrue(all(torch.equal(getattr(model, name), self.donor[name])
                            for name in self.donor))

        loss = model(z).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(getattr(model, name).grad is None
                            for name in ("gate", "up", "down")))

    def test_training_updates_adapters_but_does_not_change_donor(self):
        model = FrozenGLMExpertBridge(self.donor, self.basis, rank=3, seed=17)
        inputs = torch.randn(24, 3, generator=self.generator) * 0.7
        full_inputs = torch.cat(
            [inputs, torch.stack([inputs[:, 0] + inputs[:, 1],
                                  inputs[:, 1] - inputs[:, 2],
                                  inputs[:, 2] + inputs[:, 0]], dim=1)], dim=1,
        )
        with torch.no_grad():
            targets = model.donor_forward(full_inputs) @ self.basis
        original = {k: getattr(model, k).clone() for k in self.donor}
        output = train_frozen_bridge(
            model, inputs[:18], targets[:18],
            inputs[18:], targets[18:],
            steps=35, batch_size=6, learning_rate=0.015, seed=2026,
        )
        self.assertTrue(np.isfinite(output["heldout_loss_best"]))
        self.assertLessEqual(
            output["heldout_loss_best"],
            output["heldout_loss_before"] + 1e-9,
        )
        self.assertTrue(output["original_expert_frozen"])
        self.assertEqual(output["evaluation_samples"], 6)
        self.assertTrue(all(torch.equal(getattr(model, k), original[k])
                            for k in original))
        self.assertEqual(
            set(model.adapter_state()),
            {"input_a", "input_b", "output_a", "output_b"},
        )

    def test_shape_and_dtype_guards(self):
        with self.assertRaises(FrozenExpertBridgeError):
            FrozenGLMExpertBridge(self.donor, self.basis, rank=0)
        broken = dict(self.donor)
        broken["up"] = self.donor["up"].float()
        with self.assertRaises(FrozenExpertBridgeError):
            FrozenGLMExpertBridge(broken, self.basis, rank=2)
        model = FrozenGLMExpertBridge(self.donor, self.basis, rank=2)
        with self.assertRaises(FrozenExpertBridgeError):
            model(torch.ones(5, 4))

    def test_document_boundaries_are_rederived_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixture.txt"
            fixture.write_text("first\nsecond\n", encoding="utf-8")
            captured = root / "teacher.npy"
            ids = torch.tensor([[11, 12, 0], [21, 22, 23]])
            mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
            captured.with_suffix(".json").write_text(
                json.dumps({
                    "token_ids_sha256": hashlib.sha256(
                        ids.numpy().tobytes()
                    ).hexdigest()
                }),
                encoding="utf-8",
            )
            class Tokenizer:
                def __call__(self, texts, **kwargs):
                    self_texts = list(texts)
                    if self_texts != ["first", "second"]:
                        raise AssertionError("wrong texts")
                    return {"input_ids": ids, "attention_mask": mask}

            with patch(
                "scripts.probe_glm53_frozen_bridge.AutoTokenizer.from_pretrained",
                return_value=Tokenizer(),
            ):
                doc_ids, counts = _load_document_ids(
                    fixture, captured, max_tokens=128, token_count=5,
                )
            self.assertEqual(doc_ids.tolist(), [0, 0, 1, 1, 1])
            self.assertEqual(counts, {0: 2, 1: 3})

            meta = captured.with_suffix(".json")
            meta.write_text(json.dumps({"token_ids_sha256": "wrong"}))
            with patch(
                "scripts.probe_glm53_frozen_bridge.AutoTokenizer.from_pretrained",
                return_value=Tokenizer(),
            ):
                with self.assertRaisesRegex(ValueError, "do not match"):
                    _load_document_ids(
                        fixture, captured, max_tokens=128, token_count=5,
                    )


if __name__ == "__main__":
    unittest.main()
