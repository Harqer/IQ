"""CPU-only safety checks for IQ correction-delta loading and promotion gates."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from iq_training.transfer_overlay import CandidateDeltaError, apply_candidate_delta


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class SmallIQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(2, 2, bias=False) for _ in range(4)
        ])
        self.norm = torch.nn.LayerNorm(2)


class TestCandidateOverlay(unittest.TestCase):
    def fixture(self, root: Path):
        base = root / "base"
        candidate = root / "candidate"
        base.mkdir()
        candidate.mkdir()
        (base / "model.safetensors.index.json").write_text(
            '{"weight_map": {"demo": "demo.safetensors"}}'
        )
        model = SmallIQ()
        tensor = torch.ones_like(model.layers[2].weight) * 7
        save_file({"layers.2.weight": tensor},
                  str(candidate / "stage2_delta.safetensors"))
        manifest = {
            "schema_version": 1,
            "status": "candidate_not_promoted",
            "language_quality_verified": False,
            "base_index_sha256": digest(base / "model.safetensors.index.json"),
            "target_iq_layer": 2,
            "delta_sha256": digest(candidate / "stage2_delta.safetensors"),
            "delta_tensors": 1,
        }
        (candidate / "manifest.json").write_text(json.dumps(manifest))
        return model, base, candidate

    def test_overlay_changes_only_allowed_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, base, candidate = self.fixture(Path(tmp))
            original = model.layers[0].weight.detach().clone()
            updated = apply_candidate_delta(
                model, checkpoint_dir=base, delta_dir=candidate
            )
            self.assertEqual(updated, 1)
            self.assertTrue(torch.equal(model.layers[0].weight, original))
            self.assertTrue(torch.equal(
                model.layers[2].weight, torch.full((2, 2), 7.)
            ))

    def test_reject_wrong_parent_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, base, candidate = self.fixture(Path(tmp))
            (base / "model.safetensors.index.json").write_text("different")
            with self.assertRaisesRegex(CandidateDeltaError, "different immutable"):
                apply_candidate_delta(model, checkpoint_dir=base, delta_dir=candidate)

    def test_reject_unauthorized_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, base, candidate = self.fixture(Path(tmp))
            save_file({"layers.0.weight": torch.ones((2, 2))},
                      str(candidate / "stage2_delta.safetensors"))
            manifest = json.loads((candidate / "manifest.json").read_text())
            manifest["delta_sha256"] = digest(candidate / "stage2_delta.safetensors")
            (candidate / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(CandidateDeltaError, "unexpected"):
                apply_candidate_delta(model, checkpoint_dir=base, delta_dir=candidate)


if __name__ == "__main__":
    unittest.main()
