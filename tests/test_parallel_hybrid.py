from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from iq_model import HybridLayerType, HybridSchedule, IQHybridConfig, HybridModelError
from iq_model.hybrid import unsharded_parallel_expert_weight_bytes
from iq_model.parallel import ParallelBranchFusion, ParallelFusionError, ExpertResidualGate
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

    def test_uncalibrated_expert_does_not_change_pretrained_output(self):
        gate = ExpertResidualGate()
        pretrained = torch.randn(2, 3, 4)
        random_expert = torch.randn_like(pretrained)
        self.assertTrue(torch.equal(gate(pretrained, random_expert), pretrained))
        gate(pretrained, random_expert).sum().backward()
        self.assertIsNotNone(gate.scale.grad)
        self.assertTrue(torch.isfinite(gate.scale.grad).all())
        with torch.no_grad():
            gate.scale.fill_(1.0)
        torch.testing.assert_close(gate(pretrained, random_expert), random_expert)

    def test_expert_gate_rejects_incompatible_output(self):
        gate = ExpertResidualGate()
        with self.assertRaises(ParallelFusionError):
            gate(torch.ones(1, 2, 4), torch.ones(1, 3, 4))

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
        self.assertEqual(len(schedule.moe_positions), 75)
        self.assertEqual(schedule.parallel_positions[:3], (0, 1, 2))
        self.assertEqual(config.mamba3.num_layers, 78)
        self.assertEqual(config.model.num_hidden_layers, 78)
        self.assertEqual(config.model.hidden_size, 4096)
        self.assertEqual(config.model.vocab_size, 131072)
        self.assertEqual(config.attnres.num_layers, 78)
        self.assertEqual(schedule.layers[0], HybridLayerType.PARALLEL_DENSE_CSA)
        self.assertEqual(schedule.layers[1], HybridLayerType.PARALLEL_DENSE_CSA)
        self.assertEqual(schedule.layers[3], HybridLayerType.PARALLEL_CSA)
        self.assertEqual(config.parallel_dense_layers, (0, 1, 2))
        self.assertEqual(len(config.parallel_index_sources), 78)
        self.assertEqual(config.parallel_index_sources[3:7], (2, 2, 2, 6))
        self.assertEqual(config.moe.num_experts, 256)
        self.assertEqual(config.stable_moe.num_experts, 256)
        self.assertEqual(config.stable_moe.top_k, 8)
        self.assertEqual(config.model.intermediate_size, 12288)

    def test_full_256_expert_capacity_is_not_a_single_gpu_checkpoint(self):
        config = nvidia_parallel_78_config()
        size = unsharded_parallel_expert_weight_bytes(config)
        expected = 75 * 256 * 3 * 2048 * 2048 * 2
        self.assertEqual(size, expected)
        self.assertGreater(size, 400 * (2 ** 30))

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

    def test_cannot_erase_three_dense_ffn_stages_or_create_invalid_indexshare(self):
        config = nvidia_parallel_78_config()
        with self.assertRaisesRegex(HybridModelError, "sorted, unique"):
            replace(config, parallel_dense_layers=(1, 0, 2))
        invalid = list(config.parallel_index_sources)
        invalid[4] = 3
        with self.assertRaisesRegex(HybridModelError, "full-indexer"):
            replace(config, parallel_index_sources=tuple(invalid))
        with self.assertRaisesRegex(HybridModelError, "length must match"):
            replace(config, parallel_index_sources=(0, 1))

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
