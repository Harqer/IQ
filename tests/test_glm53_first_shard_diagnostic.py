"""Free first-shard diagnostic tests; synthetic weights only, no HF billing."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file

from scripts.diagnose_glm53_first_shard import find_first_base_shard, read_shard


class FirstShardTests(unittest.TestCase):
    def test_reads_first_base_shard_and_validates_weights(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            # Filename sorting matches the real CUDA loader, not JSON order.
            save_file({"layers.0.norm.weight": torch.ones(3)},
                      str(root / "model-context-000.safetensors"))
            save_file({"layers.1.norm.weight": torch.ones(3)},
                      str(root / "model-context-001.safetensors"))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "layers.1.norm.weight": "model-context-001.safetensors",
                    "layers.0.moe.routed_experts.5.gate_proj.weight": "moe-shard.safetensors",
                    "layers.0.norm.weight": "model-context-000.safetensors",
                }
            }), encoding="utf-8")
            name, keys = find_first_base_shard(root / "model.safetensors.index.json")
            self.assertEqual(name, "model-context-000.safetensors")
            self.assertEqual(keys, {"layers.0.norm.weight"})
            report = read_shard(root)
            self.assertEqual(report["tensor_count"], 1)
            self.assertGreater(report["bytes"], 0)

    def test_nonfinite_tensor_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            save_file({"x.weight": torch.tensor([float("nan")])},
                      str(root / "bad.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x.weight": "bad.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Nonfloating/nonfinite"):
                read_shard(root)

    def test_path_traversal_not_accepted(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "model.safetensors.index.json"
            p.write_text(json.dumps({
                "weight_map": {"x.weight": "../secret.safetensors"}
            }))
            with self.assertRaisesRegex(ValueError, "Unsafe shard filename"):
                find_first_base_shard(p)


if __name__ == "__main__":
    unittest.main()
