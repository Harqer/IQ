"""CPU-only tests for the read-only IQ device-map validator."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_iq_device_map import run


class MappedInferencePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.checkpoint = root / "checkpoint"
        self.checkpoint.mkdir()
        self.plan_path = root / "plan.json"
        self.config = {
            "model": {"num_hidden_layers": 2},
            "schedule": {"layers": ["mamba3", "moe"]},
            "mamba3": {"num_layers": 1, "mimo_rank": 4, "chunk_size": 16},
            "attnres": {"num_layers": 2},
            "stable_moe": {"num_experts": 256},
            "moe": {"num_experts": 256},
        }
        self.index = {
            "metadata": {"total_size": 20},
            "weight_map": {
                "layers.0.mamba.core.D": "mamba.safetensors",
                "layers.1.moe.routed_experts.0.down_proj.weight": "moe.safetensors",
            },
        }
        self._write_json(self.checkpoint / "iq_config.json", self.config)
        self._write_json(self.checkpoint / "model.safetensors.index.json", self.index)
        for name in ("mamba.safetensors", "moe.safetensors"):
            (self.checkpoint / name).write_bytes(b"1234567890")
        self.plan = {
            "plan_only": True,
            "inference_ready": False,
            "device_count": 2,
            "layer_to_gpu": [0, 1],
            "shard_to_gpu": {"mamba.safetensors": 0, "moe.safetensors": 1},
            "total_weight_bytes": 20,
            "indexed_parameters": 2,
            "config_sha256": self._hash(self.checkpoint / "iq_config.json"),
            "index_sha256": self._hash(self.checkpoint / "model.safetensors.index.json"),
        }
        self._write_json(self.plan_path, self.plan)

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data))

    def test_valid_metadata_passes_without_tokenizer_or_weight_payloads(self) -> None:
        report = run(self.checkpoint, self.plan_path, tokenizer_dir=None)
        self.assertTrue(report["metadata_passed"], report["errors"])
        self.assertFalse(report["inference_ready"])
        self.assertFalse(report["tensor_values_read"])
        self.assertFalse(report["tokenizer_found"])

    def test_stale_checkpoint_index_rejected(self) -> None:
        self.plan["index_sha256"] = "0" * 64
        self._write_json(self.plan_path, self.plan)
        report = run(self.checkpoint, self.plan_path, None)
        self.assertFalse(report["metadata_passed"])
        self.assertTrue(any("SHA-256 mismatch" in x for x in report["errors"]))

    def test_wrong_layer_assignment_rejected(self) -> None:
        self.plan["shard_to_gpu"]["moe.safetensors"] = 0
        self._write_json(self.plan_path, self.plan)
        report = run(self.checkpoint, self.plan_path, None)
        self.assertFalse(report["metadata_passed"])
        self.assertTrue(any("incorrect assignments" in x for x in report["errors"]))

    def test_false_inference_ready_rejected(self) -> None:
        self.plan["inference_ready"] = True
        self._write_json(self.plan_path, self.plan)
        report = run(self.checkpoint, self.plan_path, None)
        self.assertFalse(report["metadata_passed"])
        self.assertTrue(any("plan_only" in x for x in report["errors"]))

    def test_missing_shard_rejected(self) -> None:
        (self.checkpoint / "moe.safetensors").unlink()
        report = run(self.checkpoint, self.plan_path, None)
        self.assertFalse(report["metadata_passed"])
        self.assertTrue(any("missing shards" in x for x in report["errors"]))


if __name__ == "__main__":
    unittest.main()
