"""Offline tests for complete-checkpoint sparse expert paging. No HF Jobs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.iq_hf_full78_sparse_inference import (
    hydrate_expert, partition_weight_map,
)


def fake_full78_map():
    # The full index contains 315 baseline shards + 19,968 expert shards.
    weights = {
        f"base.{i}": f"base-{i:03d}.safetensors" for i in range(315)
    }
    for physical in range(2, 235, 3):
        for slot in range(256):
            filename = f"model-moe-{(physical - 2) // 3:03d}-expert-{slot:03d}.safetensors"
            prefix = f"layers.{physical}.moe.routed_experts.{slot}."
            for part in ("gate_proj", "up_proj", "down_proj"):
                weights[prefix + part + ".weight"] = filename
    return weights


class SparseFull78Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = fake_full78_map()

    def test_all_78_times_256_experts_partition_strictly(self):
        base, routed = partition_weight_map(self.mapping)
        self.assertEqual(len(base), 315)
        self.assertEqual(len(routed), 78)
        self.assertEqual(sum(len(v) for v in routed.values()), 78 * 256)
        self.assertEqual(routed[11][61], "model-moe-003-expert-061.safetensors")

    def test_missing_expert_key_rejected(self):
        bad = dict(self.mapping)
        del bad["layers.11.moe.routed_experts.61.down_proj.weight"]
        with self.assertRaisesRegex(RuntimeError, "Incomplete GLM routed expert"):
            partition_weight_map(bad)

    def test_mixed_file_for_same_expert_rejected(self):
        bad = dict(self.mapping)
        bad["layers.11.moe.routed_experts.61.down_proj.weight"] = "elsewhere.safetensors"
        with self.assertRaisesRegex(RuntimeError, "unexpectedly split"):
            partition_weight_map(bad)

    def test_real_safetensors_hydrated_expert_forward_and_shape(self):
        prefix = "layers.11.moe.routed_experts.61."
        tensors = {
            prefix + "gate_proj.weight": torch.ones(3, 4, dtype=torch.bfloat16),
            prefix + "up_proj.weight": torch.full((3, 4), .5, dtype=torch.bfloat16),
            prefix + "down_proj.weight": torch.ones(4, 3, dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.safetensors"
            save_file(tensors, str(path))
            layer = hydrate_expert(
                path, prefix=prefix, hidden=4, intermediate=3,
                beta=1_000_000., linear_beta=1_000_000.,
                device=torch.device("cpu"), dtype=torch.bfloat16, bias=False,
            )
            self.assertEqual(layer(torch.ones(2, 4, dtype=torch.bfloat16)).shape, (2, 4))
            self.assertFalse(any(x.is_meta for x in layer.parameters()))

    def test_corrupted_selected_expert_weights_fail_closed(self):
        prefix = "layers.11.moe.routed_experts.61."
        tensors = {
            prefix + "gate_proj.weight": torch.ones(3, 4),
            prefix + "up_proj.weight": torch.ones(3, 4),
            prefix + "down_proj.weight": torch.ones(5, 3),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.safetensors"
            save_file(tensors, str(path))
            with self.assertRaisesRegex(RuntimeError, "geometry mismatch"):
                hydrate_expert(
                    path, prefix=prefix, hidden=4, intermediate=3,
                    beta=1_000_000., linear_beta=1_000_000.,
                    device=torch.device("cpu"), dtype=torch.bfloat16, bias=False,
                )


if __name__ == "__main__":
    unittest.main()
