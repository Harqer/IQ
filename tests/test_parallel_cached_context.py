"""CPU cache-contract tests; production Mamba CUDA decode requires GPU acceptance."""
from __future__ import annotations

from dataclasses import replace
import unittest
import torch

from iq_model import CompressedContextConfig, CompressedSparseContextAttention, HeavilyCompressedContextAttention
from iq_model.attention.compressed import DeepseekV4CSACache, DeepseekV4HCACache
from iq_transfer.complete_transplant import nvidia_parallel_78_config


def tiny_context(*, direct: bool = False) -> CompressedContextConfig:
    return CompressedContextConfig(
        hidden_size=16, num_attention_heads=4, head_dim=8,
        q_lora_rank=8, partial_rotary_dim=4,
        max_position_embeddings=64, sliding_window=3,
        csa_compress_rate=2, hca_compress_rate=4,
        o_groups=2, o_lora_rank=4, index_n_heads=2,
        index_head_dim=8 if direct else 4, index_topk=2,
        compress_rope_theta=10000.0, rms_norm_eps=1e-6,
        direct_token_indexer=direct,
        normalize_query_output=not direct,
        normalize_candidate_nonrotary_only=direct,
        projection_bias=False,
    )


class CachedContextParityTests(unittest.TestCase):
    def check_prefill_and_single_step(self, *, csa: bool, direct: bool = False):
        torch.manual_seed(46)
        config = tiny_context(direct=direct)
        module = (
            CompressedSparseContextAttention(config)
            if csa else HeavilyCompressedContextAttention(config)
        ).eval()
        cache = DeepseekV4CSACache(config) if csa else DeepseekV4HCACache(config)
        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            full = module(x)
            prefill = module(x[:, :5], past_key_values=cache)
            steps = [
                module(x[:, i : i + 1], past_key_values=cache)
                for i in range(5, 8)
            ]
            joined = torch.cat([prefill, *steps], dim=1)
        self.assertEqual(cache.cumulative_length, 8)
        torch.testing.assert_close(joined, full, atol=3e-5, rtol=2e-4)

        cache.reset()
        self.assertEqual(cache.cumulative_length, 0)
        self.assertIsNone(cache.keys)
        if csa:
            self.assertIsNone(cache.all_token_kv)
            self.assertIsNone(cache.token_index_kv)
        with torch.no_grad():
            replayed = module(x, past_key_values=cache)
        torch.testing.assert_close(replayed, full, atol=3e-5, rtol=2e-4)

    def test_hca_prefill_step_and_reset(self):
        self.check_prefill_and_single_step(csa=False)

    def test_csa_compressed_indexer_prefill_step_and_reset(self):
        self.check_prefill_and_single_step(csa=True)

    def test_csa_direct_token_indexer_prefill_step_and_reset(self):
        self.check_prefill_and_single_step(csa=True, direct=True)

    def test_rejects_position_drift_and_document_cross_contamination(self):
        cfg = tiny_context(direct=True)
        a = CompressedSparseContextAttention(cfg).eval()
        cache = DeepseekV4CSACache(cfg)
        x = torch.randn(1, 2, 16)
        with torch.no_grad():
            a(x[:, :1], past_key_values=cache)
            with self.assertRaisesRegex(ValueError, "position_ids"):
                a(x[:, 1:], position_ids=torch.tensor([[9]]), past_key_values=cache)
            with self.assertRaisesRegex(ValueError, "packed"):
                a(x[:, 1:], document_ids=torch.ones(1, 1), past_key_values=cache)
        self.assertEqual(cache.cumulative_length, 1)


class ParallelCacheArchitectureTest(unittest.TestCase):
    def test_every_parallel_layer_has_distinct_csa_or_hca_cache_slot(self):
        config = nvidia_parallel_78_config()
        self.assertEqual(len(config.schedule.parallel_positions), 78)
        self.assertEqual(len(config.schedule.attention_positions), 78)
        self.assertEqual(len(config.schedule.mamba_positions), 78)
        self.assertEqual(config.mamba3.num_layers, 78)


if __name__ == "__main__":
    unittest.main()
