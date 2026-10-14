"""Real tensor transport and EP ownership tests, no forged donor checkpoints."""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np
import torch

from iq_model import IQHybridConfig, HybridModelError
from iq_transfer.complete_transplant import nvidia_parallel_78_config
from iq_transfer.glm53_expert_shards import (
    GLM53ExpertShardError, expert_range, _linear, _projection_pair
)
from iq_transfer.transport import CoordinateMap, transport_linear


class FullExpertShardingContractTests(unittest.TestCase):
    def test_every_global_expert_id_is_owned_once(self):
        for size in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            shards = [list(expert_range(rank, size)) for rank in range(size)]
            self.assertEqual([n for group in shards for n in group], list(range(256)))
            self.assertTrue(all(len(group) == 256 // size for group in shards))
        for bad in ((1, 3), (8, 8), (-1, 8), (0, 0)):
            with self.assertRaises(GLM53ExpertShardError):
                expert_range(*bad)

    def test_config_serializes_real_ep_size_and_shards(self):
        cfg = nvidia_parallel_78_config(expert_parallel_size=8)
        self.assertEqual(cfg.expert_parallel_size, 8)
        self.assertEqual(cfg.stable_moe.num_experts, 256)
        self.assertEqual(cfg.schedule.moe_positions, tuple(range(3, 78)))
        self.assertEqual(cfg.schedule.parallel_positions, tuple(range(78)))
        self.assertEqual(
            IQHybridConfig.from_dict(cfg.to_dict()).fingerprint,
            cfg.fingerprint,
        )
        with self.assertRaisesRegex(HybridModelError, "divisible"):
            nvidia_parallel_78_config(expert_parallel_size=3)

    def test_ridge_bases_transport_actual_small_dense_and_sparse_matrices(self):
        torch.manual_seed(12)
        source_width, target_width = 6, 4
        original = torch.randn(source_width, target_width)
        original += torch.eye(source_width, target_width)
        out = torch.randn(7, 3)
        out += torch.eye(7, 3)
        donor = torch.randn(7, 6)
        p, pinv_transposed = _projection_pair(
            original.numpy(), device=torch.device("cpu"), ridge=1e-8,
        )
        dest = _linear(
            donor, left=out, right=pinv_transposed,
            expected=(7, 6),
        )
        source_map = CoordinateMap(original.numpy(), 1e-8)
        output_map = CoordinateMap(out.numpy(), 1e-8)
        reference = transport_linear(donor.numpy(), source_map, output_map)
        torch.testing.assert_close(
            dest.float(), torch.tensor(reference, dtype=torch.float32),
            atol=8e-3, rtol=8e-3,
        )
        self.assertEqual(tuple(dest.shape), (3, 4))
        with self.assertRaisesRegex(GLM53ExpertShardError, "geometry"):
            _linear(donor, left=out, right=pinv_transposed,
                    expected=(7, 5))

    def test_fp8_requires_nvidia_bridge_dequantization(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch float8 is unavailable")
        donor = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(GLM53ExpertShardError, "FP8"):
            _linear(donor, left=None, right=torch.eye(2),
                    expected=(2, 2))


if __name__ == "__main__":
    unittest.main()
