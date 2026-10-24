"""Metadata-only donor completeness checks; no weight downloads or CUDA."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[1] / "scripts/check_glm53_source_inventory.py"
spec = importlib.util.spec_from_file_location("glm53_source_inventory", FILE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestGLM53SourceInventory(unittest.TestCase):
    def test_full_snapshot_matches(self):
        mapping = {"x": "shard0.safetensors", "y": "shard1.safetensors"}
        result = module.compare_inventory(
            {"weight_map": mapping}, {"weight_map": mapping},
            {"shard0.safetensors", "shard1.safetensors"},
        )
        self.assertEqual(result["status"], "complete_pinned_index_and_shards")
        self.assertEqual(result["official_shards"], 2)

    def test_subset_index_does_not_pass_as_complete_donor(self):
        local = {"weight_map": {"x": "shard0.safetensors"}}
        official = {"weight_map": {
            "x": "shard0.safetensors",
            "y": "shard1.safetensors",
            "z": "shard1.safetensors",
        }}
        result = module.compare_inventory(local, official, {"shard0.safetensors"})
        self.assertEqual(result["present_local_index_shards"], 1)
        self.assertEqual(result["local_index_shards"], 1)
        self.assertEqual(result["official_shards"], 2)
        self.assertEqual(result["missing_official_tensors"], 2)
        self.assertEqual(result["status"], "incomplete_or_modified_donor_inventory")

    def test_changed_or_missing_shard_is_rejected(self):
        a = {"weight_map": {"x": "a.safetensors"}}
        b = {"weight_map": {"x": "b.safetensors"}}
        result = module.compare_inventory(a, b, {"a.safetensors"})
        self.assertEqual(result["changed_shard_assignments"], 1)
        self.assertEqual(result["status"], "incomplete_or_modified_donor_inventory")

    def test_invalid_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            module.compare_inventory({}, {"weight_map": {"a": "x"}}, set())


if __name__ == "__main__":
    unittest.main()
