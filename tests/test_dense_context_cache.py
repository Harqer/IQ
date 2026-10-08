from __future__ import annotations

import unittest

import torch

from iq_model import IQModelConfig
from iq_model.attention.context_dense import DenseContextAttention, DenseContextCache


class DenseContextCacheTests(unittest.TestCase):
    def config(self) -> IQModelConfig:
        return IQModelConfig(
            vocab_size=97,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )

    def test_incremental_dense_kv_matches_causal_full_forward(self):
        torch.manual_seed(47)
        model = DenseContextAttention(self.config()).eval()
        x = torch.randn(2, 11, 16)
        with torch.no_grad():
            full = model(x)
            for sizes in ((11,), (1,) * 11, (3, 5, 3)):
                cache = DenseContextCache()
                chunks = []
                offset = 0
                for length in sizes:
                    chunks.append(
                        model(
                            x[:, offset:offset + length],
                            past_key_values=cache,
                        )
                    )
                    offset += length
                actual = torch.cat(chunks, dim=1)
                torch.testing.assert_close(actual, full, atol=2e-5, rtol=2e-5)
                self.assertEqual(cache.cumulative_length, 11)
                self.assertEqual(cache.keys.shape[2], 11)
                self.assertEqual(cache.values.shape[2], 11)
                cache.reset()
                self.assertIsNone(cache.keys)
                self.assertEqual(cache.cumulative_length, 0)

    def test_incremental_dense_kv_rejects_offset_reuse(self):
        model = DenseContextAttention(self.config()).eval()
        cache = DenseContextCache()
        x = torch.randn(1, 3, 16)
        with torch.no_grad():
            model(x, past_key_values=cache)
        with self.assertRaisesRegex(ValueError, "position_ids"):
            model(x[:, :1], position_ids=torch.zeros(1, 1, dtype=torch.long), past_key_values=cache)


if __name__ == "__main__":
    unittest.main()
