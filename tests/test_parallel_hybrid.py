from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from iq_model import HybridLayerType, HybridSchedule, IQHybridConfig, HybridModelError
from iq_model.parallel import ParallelBranchFusion, ParallelFusionError
from iq_transfer.complete_transplant import (
    canonical_glm53_config,
    nvidia_parallel_78_config,
)


class ParallelFusionTests(unittest.TestCase):
    def test_initial_mamba_function_is_preserved(self):
        torch.manual_seed(5)
        fusion = ParallelBranchFusion(8)
        mamba = torch.randn(2, 3, 8)
        attention = torch.randn(2, 3, 8)
        self.assertTrue(torch.equal(fusion(mamba, attention), mamba))
        self.assertTrue(torch.equal(fusion(mamba, -attention), mamba))

    def test_fusion_learns_attention_weights_without_mamba_drift(self):
        fusion = ParallelBranchFusion(4)
        mamba = torch.ones(1, 2, 4, requires_grad=True)
        attention = torch.full((1, 2, 4), 2.0, requires_grad=True)
        result = fusion(mamba, attention)
        result.sum().backward()
        gradient = fusion.projection.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient[:, 4:].abs().sum()), 0.0)
        self.assertTrue(torch.equal(mamba.grad, torch.ones_like(mamba)))
        self.assertTrue(torch.equal(attention.grad, torch.zeros_like(attention)))

    def test_invalid_branch_shape_fails_closed(self):
        fusion = ParallelBranchFusion(4)
        with self.assertRaises(ParallelFusionError):
            fusion(torch.ones(1, 2, 4), torch.ones(1, 3, 4))
        with self.assertRaises(ParallelFusionError):
            fusion(torch.ones(2, 4), torch.ones(2, 4))


class ParallelHybridScheduleTests(unittest.TestCase):
    def test_78_composite_blocks_count_each_operator(self):
        config = nvidia_parallel_78_config()
        schedule = config.schedule
        self.assertEqual(len(schedule.layers), 78)
        self.assertEqual(len(schedule.parallel_positions), 78)
        self.assertEqual(len(schedule.mamba_positions), 78)
        self.assertEqual(len(schedule.attention_positions), 78)
        self.assertEqual(len(schedule.moe_positions), 78)
        self.assertEqual(config.mamba3.num_layers, 78)
        self.assertEqual(config.model.num_hidden_layers, 78)
        self.assertEqual(config.model.hidden_size, 4096)
        self.assertEqual(config.model.vocab_size, 131072)
        self.assertEqual(config.attnres.num_layers, 78)
        self.assertEqual(schedule.layers[0], HybridLayerType.PARALLEL_CSA)
        self.assertEqual(schedule.layers[1], HybridLayerType.PARALLEL_HCA)

    def test_config_round_trip_is_exact(self):
        config = nvidia_parallel_78_config()
        loaded = IQHybridConfig.from_dict(config.to_dict())
        self.assertEqual(loaded.fingerprint, config.fingerprint)
        self.assertEqual(loaded.schedule.fingerprint, config.schedule.fingerprint)

    def test_legacy_interleaved_configuration_is_unchanged(self):
        config = canonical_glm53_config()
        self.assertEqual(len(config.schedule.layers), 80)
        self.assertEqual(config.mamba3.num_layers, 32)
        self.assertFalse(config.schedule.parallel_positions)
        self.assertEqual(len(config.schedule.attention_positions), 24)
        self.assertEqual(len(config.schedule.moe_positions), 24)

    def test_config_cannot_miscount_recurrent_slots(self):
        config = nvidia_parallel_78_config()
        with self.assertRaisesRegex(HybridModelError, "number of Mamba layers"):
            replace(config, mamba3=replace(config.mamba3, num_layers=32))

    def test_new_schedule_serializes_typed_composite_blocks(self):
        schedule = HybridSchedule.parse("parallel_csa parallel_hca pc ph")
        self.assertEqual(
            schedule.to_tokens(),
            ("parallel_csa", "parallel_hca", "parallel_csa", "parallel_hca"),
        )
        self.assertEqual(
            HybridSchedule.from_dict(schedule.to_dict()).fingerprint,
            schedule.fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
