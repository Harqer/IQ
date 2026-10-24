"""Regression tests: failed saves never register incomplete IQ tensor keys."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from iq_transfer.complete_transplant import _save_shard, CompleteTransplantError
from iq_transfer.glm53_incremental_moe import (
    _check_runtime_budget, GLM53IncrementalMoEError,
)


class DurableShardTests(unittest.TestCase):
    def test_persist_before_index_and_forbid_overwrite(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            weight_map = {}
            name = "model-moe-003-expert-002.safetensors"
            tensor = {"layers.11.moe.routed_experts.2.gate_proj.weight":
                      torch.ones(3, 4, dtype=torch.bfloat16)}
            _save_shard(root, name, tensor, weight_map)
            self.assertEqual(set(weight_map), set(tensor))
            self.assertTrue((root / name).stat().st_size > 16)
            with self.assertRaises(CompleteTransplantError):
                _save_shard(root, name, tensor, {})
            self.assertEqual(len(list(root.glob("*.partial"))), 0)

    def test_interrupted_serializer_has_no_final_shard_or_mapping(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            mapping = {}
            data = {"layers.11.norm.weight": torch.ones(16)}
            with patch("safetensors.torch.save_file",
                       side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    _save_shard(root, "model-context-003.safetensors",
                                data, mapping)
            self.assertEqual(mapping, {})
            self.assertFalse((root / "model-context-003.safetensors").exists())
            self.assertFalse(list(root.glob("*.partial")))

    def test_nonfinite_never_published(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(CompleteTransplantError):
                _save_shard(root, "model-native.safetensors",
                            {"bad.weight": torch.tensor([float("nan")])}, {})
            self.assertEqual(list(root.iterdir()), [])

    def test_budget_fail_closed_before_new_expert(self):
        with patch.dict("os.environ", {"IQ_TRANSFER_DEADLINE_EPOCH": "1"}):
            with self.assertRaisesRegex(GLM53IncrementalMoEError,
                                        "time budget reached"):
                _check_runtime_budget(3, 41)
        with patch.dict("os.environ", {"IQ_TRANSFER_DEADLINE_EPOCH": "n/a"}):
            with self.assertRaisesRegex(GLM53IncrementalMoEError, "invalid"):
                _check_runtime_budget(3, 41)


if __name__ == "__main__":
    unittest.main()
