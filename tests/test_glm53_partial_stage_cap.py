"""Small free-GPU sessions select bounded real WARM source-stage groups."""
from __future__ import annotations

import unittest

from iq_transfer.cli import _parser
from iq_transfer.glm53_partial_repair import PartialCalibrationStageWriter


class FreeGpuStageBatchOptions(unittest.TestCase):
    def test_arg_parser_allows_bounded_repair_batch(self):
        parsed = _parser().parse_args([
            "glm53-bootstrap-calibration",
            "--checkpoint", "checkpoint",
            "--checkpoint-revision", "pinned",
            "--donor-license", "GLM-5.3",
            "--output", "staged",
            "--routed-experts", "256",
            "--full-78",
            "--repair-overlay",
            "--repair-max-stages", "4",
            "--warm-device", "cuda",
            "--streaming-source",
        ])
        self.assertTrue(parsed.repair_overlay)
        self.assertEqual(parsed.repair_max_stages, 4)
        self.assertEqual(parsed.routed_experts, 256)
        self.assertTrue(parsed.full_78)
        self.assertEqual(parsed.warm_device, "cuda")

    def test_max_stage_selection_preserves_all_unselected_source_stages(self):
        missing = [1, 2, 4, 5, 7, 8, 9, 11]
        selected = frozenset(missing[:4])
        skipped = frozenset(range(78)).difference(selected)
        self.assertEqual(selected, {1, 2, 4, 5})
        self.assertTrue(set(missing[4:]).issubset(skipped))
        self.assertEqual(len(selected) + len(skipped), 78)


if __name__ == "__main__":
    unittest.main()
