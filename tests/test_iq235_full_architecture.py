"""CPU-only structural checks for the actual 235-layer IQ experiment."""
from __future__ import annotations
import copy
from pathlib import Path
import unittest

from iq_cloud.full_iq235 import (
    EXPECTED_TYPES, EXPECTED_VOCAB, FullIQ235Error, GPU_COUNT,
    physical_layer_placement, read_architecture_contract,
)


class FullIQ235StructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_architecture_contract()

    def test_full_physical_schedule_not_student(self):
        schedule = self.config["schedule"]["layers"]
        self.assertEqual(len(schedule), 235)
        for kind, count in EXPECTED_TYPES.items():
            self.assertEqual(schedule.count(kind), count)
        self.assertEqual(self.config["stable_moe"]["num_experts"], 256)
        self.assertEqual(self.config["mamba3"]["mimo_rank"], 4)
        self.assertEqual(self.config["model"]["vocab_size"], EXPECTED_VOCAB)

    def test_distribution_covers_every_physical_layer(self):
        ids = physical_layer_placement(self.config)
        self.assertEqual(len(ids), 235)
        self.assertEqual(set(ids), set(range(GPU_COUNT)))
        for layer_type in ("mamba3", "moe", "csa", "hca"):
            occurrences = [d for d, lt in zip(ids, self.config["schedule"]["layers"])
                           if lt == layer_type]
            self.assertEqual(len(occurrences), EXPECTED_TYPES[layer_type])

    def test_non_eight_gpu_topology_fails(self):
        with self.assertRaisesRegex(FullIQ235Error, "exactly eight"):
            physical_layer_placement(self.config, gpu_count=1)

    def test_student_model_is_not_referenced_by_full_deployment(self):
        model_path = (Path(__file__).resolve().parents[1]
                      / "deploy" / "baseten_full_iq" / "model" / "model.py")
        source = model_path.read_text(encoding="utf-8")
        self.assertIn("load_phase_a", source)
        self.assertNotIn("load_glm_student", source)
        self.assertNotIn("iq_glm53_mamba3", source)

    def test_config_requests_full_h200_eight(self):
        path = (Path(__file__).resolve().parents[1]
                / "deploy" / "baseten_full_iq" / "config.yaml")
        content = path.read_text(encoding="utf-8")
        self.assertIn("accelerator: H200:8", content)
        self.assertIn("IQ_PHASE: architecture_only", content)
        self.assertIn("model_name: iq235-glm53-architecture-only", content)
        self.assertNotIn("gpu_count:", content)

    def test_contract_mismatch_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["model"]["vocab_size"] = 128256
        with self.assertRaises(FullIQ235Error):
            physical_layer_placement(changed, gpu_count=4)


if __name__ == "__main__":
    unittest.main()
