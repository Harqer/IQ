"""CPU-only checks of GLM coverage accounting and frozen Mamba expansion plans."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plan_glm53_weight_extension.py"
spec = importlib.util.spec_from_file_location("glm53_expansion_planner", SCRIPT)
assert spec and spec.loader
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


SOURCE = {
    "model_type": "glm_moe_dsa",
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "n_routed_experts": 256,
}
MAPPED = [0, 3, 6, 10, 14, 17, 22, 23, 26, 30, 34, 37,
          42, 44, 46, 50, 54, 57, 62, 64, 66, 70, 74, 77]
MANIFEST = {
    "capability_source": {"repo": "zai-org/GLM-5.3-BF16", "revision": "abc"},
    "foundation": {"repo": "state-spaces/mamba3-mimo-1.5b"},
    "recipient_config_fingerprint": "hash",
    "source_layer_map": {str(i): v for i, v in enumerate(MAPPED)},
}
SCHEDULE = []
for i in range(24):
    SCHEDULE.extend(("mamba3", "csa" if i % 2 == 0 else "hca", "moe"))
    if (i + 1) % 3 == 0:
        SCHEDULE.append("mamba3")
CONFIG = {
    "model": {"vocab_size": 128256},
    "stable_moe": {"num_experts": 32},
    "schedule": {"layers": SCHEDULE},
}


class TestGLM53WeightExpansion(unittest.TestCase):
    def test_known_checkpoint_exact_layer_coverage(self):
        actual = planner.plan_coverage(SOURCE, MANIFEST, CONFIG, target_experts=32)
        self.assertEqual(len(actual["selected_source_layers"]), 24)
        self.assertEqual(len(actual["not_yet_mapped_source_layers"]), 54)
        self.assertEqual(actual["covered_dense_source_layers"], [0])
        self.assertEqual(len(actual["unmapped_dense_source_layers"]), 2)
        self.assertEqual(len(actual["covered_sparse_source_layers"]), 23)
        self.assertEqual(actual["routed_expert_slots"]["original_mapped_max"], 736)
        self.assertEqual(actual["routed_expert_slots"]["source_total"], 19200)
        self.assertFalse(actual["expert_ids_in_existing_checkpoint_verified"])

    def test_staged_new_expert_counts_preserve_mamba_schedule(self):
        for n in (64, 128, 256):
            with self.subTest(experts=n):
                result = planner.plan_coverage(SOURCE, MANIFEST, CONFIG, target_experts=n)
                self.assertEqual(result["routed_expert_slots"]["planned_mapped_max"], 23 * n)
                self.assertEqual(result["routed_expert_slots"]["new_slots"], 23 * (n - 32))
                self.assertTrue(result["recipient"]["mamba_weights_must_remain_identical"])
                self.assertEqual(result["recipient"]["mamba_layers"], 32)
                self.assertEqual(result["status"], "plan_only_no_weights_transferred")

    def test_reject_invalid_expansion_and_source(self):
        for n in (0, 33, 512):
            with self.subTest(experts=n):
                with self.assertRaises(planner.CoverageError):
                    planner.plan_coverage(SOURCE, MANIFEST, CONFIG, target_experts=n)
        bad = {**MANIFEST, "foundation": {"repo": "unexpected"}}
        with self.assertRaisesRegex(planner.CoverageError, "Mamba"):
            planner.plan_coverage(SOURCE, bad, CONFIG)

    def test_expert_identity_schema_must_be_complete(self):
        identities = {
            str(i): None if source < 3 else list(range(32))
            for i, source in enumerate(MAPPED)
        }
        with_ids = {**MANIFEST, "selected_source_expert_ids": identities}
        result = planner.plan_coverage(SOURCE, with_ids, CONFIG)
        self.assertTrue(result["expert_ids_in_existing_checkpoint_verified"])
        identities["1"] = [0] * 32
        with self.assertRaisesRegex(planner.CoverageError, "expert identity"):
            planner.plan_coverage(SOURCE, with_ids, CONFIG)

    def test_wrong_source_mapping_and_changed_schedule_fail_closed(self):
        bad_map = {**MANIFEST, "source_layer_map": {
            **MANIFEST["source_layer_map"], "3": MAPPED[2]
        }}
        with self.assertRaisesRegex(planner.CoverageError, "monotonic"):
            planner.plan_coverage(SOURCE, bad_map, CONFIG)
        altered = {**CONFIG, "schedule": {"layers": SCHEDULE[:-1]}}
        with self.assertRaisesRegex(planner.CoverageError, "schedule"):
            planner.plan_coverage(SOURCE, MANIFEST, altered)


if __name__ == "__main__":
    unittest.main()
