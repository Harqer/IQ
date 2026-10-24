"""Full GLM-5.3 78-layer transfer geometry must not collapse to 24 stages."""
from __future__ import annotations

import unittest

from iq_model import HybridLayerType
from iq_transfer.complete_transplant import (
    CompleteTransplantError,
    canonical_glm53_config,
    canonical_glm53_full78_config,
    donor_layer_positions,
)
from iq_transfer.glm53_calibration import _bootstrap_source_mapping


class Full78ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.types = tuple(
            "full" if layer % 3 == 0 else "shared"
            for layer in range(78)
        )

    def test_all_78_glm_layers_have_distinct_iq_context_and_moe(self):
        config = canonical_glm53_full78_config(
            source_indexer_types=self.types,
            routed_experts=256,
        )
        pairs = donor_layer_positions(config)
        self.assertEqual(len(pairs), 78)
        self.assertEqual(len(config.schedule.layers), 235)
        self.assertEqual(config.schedule.count(HybridLayerType.MAMBA3), 79)
        self.assertEqual(config.schedule.count(HybridLayerType.MOE), 78)
        self.assertEqual(config.mamba3.num_layers, 79)
        self.assertEqual(config.attnres.num_layers, 235)
        self.assertEqual(config.stable_moe.num_experts, 256)
        self.assertEqual(config.stable_moe.top_k, 8)
        self.assertEqual(
            [config.schedule.layers[c] for c, _ in pairs],
            [
                HybridLayerType.CSA if t == "full" else HybridLayerType.HCA
                for t in self.types
            ],
        )
        mapping = _bootstrap_source_mapping(
            source_layers=78,
            stage_positions=pairs,
            target_config=config,
            source_indexer_types=self.types,
        )
        self.assertEqual(mapping, {i: i for i in range(78)})
        self.assertEqual(len(set(pairs)), 78)

    def test_original_24_stage_config_is_not_changed(self):
        old = canonical_glm53_config(routed_experts=256)
        self.assertEqual(len(donor_layer_positions(old)), 24)
        self.assertEqual(old.mamba3.num_layers, 32)
        self.assertNotEqual(
            old.fingerprint,
            canonical_glm53_full78_config(
                source_indexer_types=self.types, routed_experts=256
            ).fingerprint,
        )

    def test_missing_or_bad_official_layer_types_fail(self):
        for entries in (self.types[:24], self.types + ("full",),
                        ("other",) + self.types[1:]):
            with self.assertRaises(CompleteTransplantError):
                canonical_glm53_full78_config(source_indexer_types=entries)


if __name__ == "__main__":
    unittest.main()
