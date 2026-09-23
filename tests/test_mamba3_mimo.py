from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from iq_model import (
    MAMBA3_UPSTREAM_COMMIT,
    Mamba3MIMOConfig,
    Mamba3MIMORuntimeError,
    recommended_mamba3_chunk_size,
    require_mamba3_mimo_runtime,
)


class Mamba3MIMOContractTests(unittest.TestCase):
    def test_production_config_is_rank4_mimo(self):
        config = Mamba3MIMOConfig.production_4096x32()
        self.assertEqual(config.d_model, 4096)
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.d_state, 128)
        self.assertEqual(config.headdim, 64)
        self.assertEqual(config.mimo_rank, 4)
        self.assertEqual(config.chunk_size, 16)
        self.assertGreaterEqual(config.mimo_rank, 2)
        self.assertEqual(len(MAMBA3_UPSTREAM_COMMIT), 40)

    def test_siso_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            Mamba3MIMOConfig(
                d_model=4096,
                num_layers=32,
                mimo_rank=1,
            )

    def test_chunk_size_matches_upstream_mimo_guidance(self):
        self.assertEqual(
            recommended_mamba3_chunk_size(
                mimo_rank=4,
                dtype=torch.bfloat16,
            ),
            16,
        )
        self.assertEqual(
            recommended_mamba3_chunk_size(
                mimo_rank=4,
                dtype=torch.float32,
            ),
            8,
        )
        with self.assertRaises(Mamba3MIMORuntimeError):
            recommended_mamba3_chunk_size(
                mimo_rank=3,
                dtype=torch.bfloat16,
            )

    def test_cpu_target_fails_instead_of_falling_back(self):
        with self.assertRaises(Mamba3MIMORuntimeError):
            require_mamba3_mimo_runtime(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
