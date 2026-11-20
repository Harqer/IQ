"""No-GPU IQ235 readiness decisions and six synthetic shard-header checks."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import unittest

from iq_cloud.bulk_readiness import (
    choose_representative_shards, compare_representative_headers,
    static_release_blockers,
)
from iq_cloud.checkpoint_metadata import IQ235MetadataError
from iq_cloud.full235_weight_loader import PINNED_SOURCE_SHARD_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]


def tiny_blob(key):
    header = {key: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    data = json.dumps(header).encode()
    return struct.pack("<Q", len(data)) + data + b"\0" * 8


class FakeHeaders:
    def __init__(self, data):
        self.data = data

    def info(self, uri):
        return {"size": len(self.data[uri.rsplit("/", 1)[-1]])}

    def open(self, uri, mode, *, block_size):
        return BytesIO(self.data[uri.rsplit("/", 1)[-1]])


class ReadinessTests(unittest.TestCase):
    def test_repo_not_ready_even_when_cpu_tests_pass(self):
        report = static_release_blockers(
            ROOT, approved_digest_pin=PINNED_SOURCE_SHARD_MANIFEST_SHA256,
        )
        self.assertFalse(report["production_inference_ready"])
        self.assertFalse(report["readiness_to_collect_inference_api_key"])
        self.assertFalse(report["runtime_restricts_architecture_only"])
        self.assertTrue(report["default_deployment_is_architecture_only"])
        self.assertTrue(report["source_vocab_experimental_path_integrated"])
        self.assertTrue(report["source_vocab_verified_mode_blocked_on_manifest_pin"])
        self.assertTrue(report["single_pass_exploratory_integrated"])
        self.assertFalse(report["single_pass_exploratory_executed"])
        self.assertTrue(any(
            step.startswith("Stream/sha256/load each source shard exactly once")
            for step in report["steps_before_real_pretrained_inference"]
        ))
        self.assertIn("Pin the fully reviewed source manifest",
                      report["steps_before_verified_production_inference"])
        self.assertTrue(report["runtime_injects_checkpoint_weights"])
        self.assertTrue(report["baseten_stage_includes_loader"])
        self.assertTrue(report["lexical_production_training_blocked"])

    def test_six_disjoint_synthetic_header_categories(self):
        names = [
            "model-global.safetensors", "model-native.safetensors",
            "model-mamba-000.safetensors", "model-context-000.safetensors",
            "model-moe-000-common.safetensors",
            "model-moe-000-expert-000.safetensors",
        ]
        index = {"weight_map": {f"k.{i}": name for i, name in enumerate(names)}}
        selected = choose_representative_shards(index["weight_map"])
        self.assertEqual(len(selected), 6)
        details = compare_representative_headers(
            FakeHeaders({name: tiny_blob(f"k.{i}") for i, name in enumerate(names)}),
            "hf://buckets/a/b/run", index,
        )
        self.assertEqual(details["representative_shard_count"], 6)
        self.assertTrue(all(
            value["verified_header_tensors"] == 1
            for value in details["samples"].values()
        ))
        self.assertFalse(details["all_source_payloads_verified"])

    def test_missing_shard_type_fails_closed(self):
        with self.assertRaisesRegex(IQ235MetadataError, "missing"):
            choose_representative_shards({"x": "model-global.safetensors"})

    def test_missing_repository_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(IQ235MetadataError, "required source file"):
                static_release_blockers(Path(root), approved_digest_pin=None)


if __name__ == "__main__":
    unittest.main()
