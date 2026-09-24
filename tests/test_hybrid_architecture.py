from __future__ import annotations

import unittest

import torch

from iq_model import (
    ArchitectureError,
    DenseContextAttention,
    HeadRMSNorm,
    HybridLayerType,
    HybridSchedule,
    IQModelConfig,
)
from iq_model.position import (
    RotaryEmbedding,
    apply_inverse_partial_rotary_at_end,
    apply_partial_rotary_at_end,
)


class HybridArchitectureTests(unittest.TestCase):
    def config(self) -> IQModelConfig:
        return IQModelConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )

    def test_explicit_schedule_is_mamba_dominant(self):
        schedule = HybridSchedule.parse(
            "M E M C E M E M A E M H X"
        )
        self.assertEqual(schedule.count(HybridLayerType.MAMBA3), 5)
        self.assertEqual(
            schedule.attention_positions,
            (3, 8, 11),
        )
        self.assertEqual(
            schedule.layers[-1],
            HybridLayerType.EXECUTIVE,
        )

        with self.assertRaises(ArchitectureError):
            HybridSchedule.parse("M A C H")
        with self.assertRaises(ArchitectureError):
            HybridSchedule.parse("M X E")

    def test_head_rmsnorm_normalizes_per_head_dimension(self):
        norm = HeadRMSNorm(4, eps=1e-6)
        x = torch.randn(2, 3, 5, 4)
        y = norm(x)
        rms = y.float().pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=2e-4, rtol=2e-4))

    def test_dense_context_attention_is_causal_and_uses_qk_norm(self):
        torch.manual_seed(3)
        attention = DenseContextAttention(self.config()).eval()
        self.assertEqual(tuple(attention.q_norm.weight.shape), (4,))
        self.assertEqual(tuple(attention.k_norm.weight.shape), (4,))

        a = torch.randn(1, 5, 16)
        b = a.clone()
        b[:, -1] = torch.randn_like(b[:, -1])
        with torch.no_grad():
            out_a = attention(a)
            out_b = attention(b)
        self.assertTrue(
            torch.allclose(
                out_a[:, :-1],
                out_b[:, :-1],
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_partial_rope_inverse_round_trip(self):
        torch.manual_seed(5)
        rotary = RotaryEmbedding(4, 32, 10000.0)
        positions = torch.tensor([[0, 1, 2, 3]])
        cos, sin = rotary.cos_sin(
            positions,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 2, 4, 8)
        rotated = apply_partial_rotary_at_end(x, cos, sin, 4)
        restored = apply_inverse_partial_rotary_at_end(rotated, cos, sin, 4)
        self.assertTrue(torch.allclose(x, restored, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
