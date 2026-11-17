"""CPU-only checks of phase selection, telemetry wrapping and full traversal."""
from __future__ import annotations
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from iq_cloud.phase_probe import check_phase, PhaseError, full_forward_probe, schema_digest


class DummyTokenizer:
    eos_token_id = 5
    def __call__(self, text, *, return_tensors):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(x) for x in ids)
    def get_vocab(self):
        return {"a": 1, "b": 2, "c": 3}


class DummyFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DummyTokenizer()
        self.core = nn.Module()
        self.core.backbone = nn.Module()
        self.core.backbone.embedding = nn.Embedding(16, 4)
        self.core.backbone.layers = nn.ModuleList(nn.Identity() for _ in range(24))
    def validate_input(self, ids, new_tokens=1):
        pass
    def forward(self, ids, num_last_tokens=1):
        x = self.core.backbone.embedding(ids)
        for block in self.core.backbone.layers:
            x = block(x)
        logits = torch.full((1, 1, 154880), -100., device=ids.device)
        logits[..., 6] = 1.
        return SimpleNamespace(logits=logits)


class TestCloudPhase(unittest.TestCase):
    def test_strict_phases(self):
        self.assertEqual(check_phase("architecture_only"), "architecture_only")
        self.assertEqual(check_phase("injected"), "injected")
        with self.assertRaises(PhaseError):
            check_phase("legacy")
    def test_full_24_layer_walk(self):
        model = DummyFull()
        result = full_forward_probe(model, "test", phase="architecture_only", max_new_tokens=2)
        self.assertEqual(result.completed_layers, 24)
        self.assertEqual(result.generated_ids, [6, 6])
        self.assertEqual(result.schema_sha256, schema_digest(model))
    def test_missing_layer_rejected(self):
        model = DummyFull()
        model.core.backbone.layers = nn.ModuleList(nn.Identity() for _ in range(23))
        with self.assertRaises(PhaseError):
            full_forward_probe(model, "test", phase="architecture_only")


if __name__ == "__main__":
    unittest.main()
