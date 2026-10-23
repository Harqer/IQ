"""Parity and bounded-memory regression tests for streamed CSA/HCA."""

from __future__ import annotations

import unittest

import torch

from iq_model import (
    CompressedContextConfig,
    CompressedSparseContextAttention,
    HeavilyCompressedContextAttention,
)


def tiny_config(*, direct: bool, topk: int = 64) -> CompressedContextConfig:
    return CompressedContextConfig(
        hidden_size=16,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=8,
        partial_rotary_dim=4,
        max_position_embeddings=128,
        sliding_window=3,
        csa_compress_rate=2,
        hca_compress_rate=4,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=topk,
        direct_token_indexer=direct,
        compress_rope_theta=10000.0,
    )


def slow_reference(
    attention, hidden: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Original per-token reference; intentionally retained for exact parity."""
    length = hidden.shape[0]
    q_residual = attention.q_a_norm(attention.q_a_proj(hidden))
    q = attention.q_b_proj(q_residual).view(
        length, attention.config.num_attention_heads, attention.config.head_dim
    )
    if attention.config.normalize_query_output:
        q = attention.q_b_norm(q)
    q = attention._rope(q, positions)
    local_kv = attention._normalize_main_candidate(
        attention.kv_proj(hidden), attention.kv_norm
    )
    local_kv = attention._rope(local_kv.unsqueeze(1), positions).squeeze(1)
    compressed = attention._compress_main(hidden, positions)
    selected = (
        attention._index_selection(hidden, q_residual, positions)
        if attention.mode == "csa"
        else None
    )
    outputs = []
    for token in range(length):
        start = max(0, token - attention.config.sliding_window + 1)
        local = local_kv[start : token + 1]
        if attention.mode == "hca":
            long_range = compressed[: (token + 1) // attention.config.hca_compress_rate]
        else:
            assert selected is not None
            valid = selected[token][selected[token] >= 0]
            source = local_kv if attention.config.direct_token_indexer else compressed
            long_range = source.index_select(0, valid) if valid.numel() else source[:0]
        candidates = torch.cat((local, long_range), dim=0)
        outputs.append(attention._attend_candidates(q[token], candidates))
    result = attention._rope(torch.stack(outputs), positions, inverse=True)
    return attention.output(result)


class StreamingAttentionTests(unittest.TestCase):
    def test_streaming_parity_against_original_reference(self):
        for direct in (False, True):
            for cls in (
                CompressedSparseContextAttention,
                HeavilyCompressedContextAttention,
            ):
                with self.subTest(direct=direct, mode=cls.__name__):
                    torch.manual_seed(40)
                    layer = cls(tiny_config(direct=direct)).eval()
                    hidden = torch.randn(21, 16)
                    positions = torch.arange(21)
                    with torch.no_grad():
                        reference = slow_reference(layer, hidden, positions)
                        streamed = layer._forward_segment(hidden, positions)
                    torch.testing.assert_close(
                        streamed, reference, atol=3e-5, rtol=3e-5
                    )

    def test_forward_step_matches_full_prefill(self):
        for direct in (False, True):
            for cls in (
                CompressedSparseContextAttention,
                HeavilyCompressedContextAttention,
            ):
                with self.subTest(direct=direct, mode=cls.__name__):
                    torch.manual_seed(41)
                    layer = cls(tiny_config(direct=direct)).eval()
                    hidden = torch.randn(1, 37, 16)
                    with torch.no_grad():
                        full = layer(hidden)
                        cache = layer.new_cache(
                            device=hidden.device, dtype=hidden.dtype
                        )
                        outputs = [
                            layer.forward_step(hidden[:, i : i + 1], cache)
                            for i in range(hidden.shape[1])
                        ]
                        decoded = torch.cat(outputs, dim=1)
                    self.assertEqual(cache.length, hidden.shape[1])
                    torch.testing.assert_close(decoded, full, atol=4e-5, rtol=4e-5)
                    if cls is HeavilyCompressedContextAttention or not direct:
                        self.assertEqual(
                            cache.compressed_count, 37 // layer.compress_rate
                        )
                    else:
                        self.assertEqual(cache.compressed_count, 0)
                        self.assertEqual(cache.index_count, 37)
                        self.assertEqual(cache.direct_count, 37)

    def test_streamed_topk_agrees_with_full_indexer_for_complete_selection(self):
        for direct in (False, True):
            torch.manual_seed(42)
            layer = CompressedSparseContextAttention(tiny_config(direct=direct)).eval()
            h = torch.randn(16, 16)
            pos = torch.arange(16)
            residual = layer.q_a_norm(layer.q_a_proj(h))
            keys = layer._index_keys(h, pos)
            with torch.no_grad():
                streaming = layer._streaming_index_selection(
                    h, residual, pos, pos, keys, key_chunk_size=3
                )
                baseline = layer._index_selection(h, residual, pos)
            for a, b in zip(streaming, baseline):
                self.assertEqual(set(a[a >= 0].tolist()), set(b[b >= 0].tolist()))

    def test_causal_and_packed_isolation(self):
        for cls in (
            CompressedSparseContextAttention,
            HeavilyCompressedContextAttention,
        ):
            layer = cls(tiny_config(direct=True)).eval()
            docs = torch.tensor([[0] * 12 + [1] * 12])
            mask = torch.tensor([[1] * 21 + [0] * 3])
            torch.manual_seed(43)
            a = torch.randn(1, 24, 16)
            b = a.clone()
            b[:, :12] = torch.randn_like(b[:, :12])
            with torch.no_grad():
                out_a = layer(a, document_ids=docs, attention_mask=mask)
                out_b = layer(b, document_ids=docs, attention_mask=mask)
            torch.testing.assert_close(out_a[:, 12:21], out_b[:, 12:21])
            self.assertTrue(bool((out_a[:, 21:] == 0).all()))

    def test_no_dense_index_score_materialization_in_forward(self):
        layer = CompressedSparseContextAttention(
            tiny_config(direct=True, topk=3)
        ).eval()

        def forbidden(*args, **kwargs):
            raise AssertionError("full score matrix must not be allocated")

        layer._index_scores = forbidden
        with torch.no_grad():
            result = layer(torch.randn(1, 64, 16))
        self.assertEqual(result.shape, (1, 64, 16))


if __name__ == "__main__":
    unittest.main()
