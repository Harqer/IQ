"""Metadata-only full235 migration and hostile index tests; no GPU/weights."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from iq_cloud.checkpoint_migration import build_migration_plan
from iq_cloud.checkpoint_metadata import IQ235MetadataError

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = json.loads((ROOT / "configs/iq235_checkpoint_source.json").read_text())
TARGET = json.loads((ROOT / "configs/iq235_glm53_tokenizer_phase_a.json").read_text())

MARKER = {
    "status": "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED",
    "inference_ready": False, "tokenizer_packaged": False,
    "end_to_end_knowledge_retention_proven": False,
    "mapped_layers": 78, "mapped_routed_expert_slots": 19200,
    "routed_experts_per_layer": 256,
}


def index_fixture():
    return {
        "metadata": {"total_size": 12345},
        "weight_map": {
            "embed_tokens.weight": "model-global.safetensors",
            "lm_head.weight": "model-global.safetensors",
            "norm.weight": "model-global.safetensors",
            "layers.0.norm.weight": "model-mamba-000.safetensors",
            "layers.1.norm.weight": "model-context-001.safetensors",
            "layers.2.norm.weight": "model-moe-002.safetensors",
        },
    }


def check(index=None, source=None, target=None, marker=None):
    return build_migration_plan(
        ORIGINAL if source is None else source,
        TARGET if target is None else target,
        index_fixture() if index is None else index,
        MARKER if marker is None else marker,
        test_counts=(6, 4, 12345),
    )


class MetadataMigrationTests(unittest.TestCase):
    def test_preserves_original_index_and_distinct_lexical_rows(self):
        original = index_fixture()
        saved = deepcopy(original)
        report = check(index=original)
        self.assertEqual(report["source_index_tensor_keys"], 6)
        self.assertEqual(report["reused_nonlexical_source_key_count"], 4)
        self.assertEqual(report["target_tensor_key_count"], 6)
        self.assertEqual(report["replacement_embedding_shape"], [154880, 4096])
        self.assertEqual(report["replacement_head_shape"], [154880, 4096])
        self.assertEqual(report["body_tensor_keys_by_planned_gpu"]["0"], 2)
        self.assertEqual(report["body_tensor_keys_by_planned_gpu"]["1"], 1)
        self.assertEqual(report["body_tensor_keys_by_planned_gpu"]["2"], 1)
        self.assertFalse(report["all_source_tensors_injected"])
        self.assertFalse(report["readiness_for_training"])
        self.assertFalse(report["inference_ready"])
        self.assertEqual(original, saved)

    def test_lexical_source_tensors_cannot_be_implicitly_reused(self):
        index = index_fixture()
        index["weight_map"]["embed_tokens.weight"] = "model-mamba-000.safetensors"
        with self.assertRaises(IQ235MetadataError):
            check(index=index)

    def test_bad_physical_depth_refused(self):
        index = index_fixture()
        index["weight_map"]["layers.2.norm.weight"] = "model-moe-002.safetensors"
        index["weight_map"]["layers.235.norm.weight"] = index["weight_map"].pop(
            "layers.0.norm.weight"
        )
        with self.assertRaisesRegex(IQ235MetadataError, "out of range"):
            check(index=index)

    def test_wrong_target_profile_refused(self):
        target = deepcopy(TARGET)
        target["model"]["hidden_size"] = 6144
        with self.assertRaises(ValueError):
            check(target=target)

    def test_wrong_source_marker_refused(self):
        marker = dict(MARKER, inference_ready=True)
        with self.assertRaisesRegex(IQ235MetadataError, "provenance"):
            check(marker=marker)

    def test_unsafe_index_paths_refused(self):
        index = index_fixture()
        index["weight_map"]["norm.weight"] = "../weights.safetensors"
        with self.assertRaises(IQ235MetadataError):
            check(index=index)

    def test_malformed_layer_key_refused(self):
        index = index_fixture()
        index["weight_map"]["layers.nope.norm.weight"] = index["weight_map"].pop(
            "layers.0.norm.weight"
        )
        with self.assertRaisesRegex(IQ235MetadataError, "malformed"):
            check(index=index)

    def test_counts_checked_strictly(self):
        with self.assertRaisesRegex(IQ235MetadataError, "count"):
            build_migration_plan(ORIGINAL, TARGET, index_fixture(), MARKER)


if __name__ == "__main__":
    unittest.main()
