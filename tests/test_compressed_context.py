from __future__ import annotations

import unittest

import torch

from iq_model import (
    CompressedContextConfig,
    CompressedSparseContextAttention,
    HeavilyCompressedContextAttention,
)
from iq_model.attention.compressed import DeepseekV4CSACache, DeepseekV4HCACache
from iq_model.position import (
    InterleavedRotaryEmbedding,
    apply_inverse_partial_rotary_at_end,
    apply_partial_rotary_at_end,
)


class CompressedContextAttentionTests(unittest.TestCase):
    def config(self) -> CompressedContextConfig:
        return CompressedContextConfig(
            hidden_size=16,
            num_attention_heads=4,
            head_dim=8,
            q_lora_rank=8,
            partial_rotary_dim=4,
            max_position_embeddings=64,
            sliding_window=3,
            csa_compress_rate=2,
            hca_compress_rate=4,
            o_groups=2,
            o_lora_rank=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            compress_rope_theta=10000.0,
            rms_norm_eps=1e-6,
        )

    def test_csa_hca_shapes_gradients_and_causality(self):
        torch.manual_seed(31)
        x = torch.randn(1, 8, 16, requires_grad=True)
        for cls in (
            CompressedSparseContextAttention,
            HeavilyCompressedContextAttention,
        ):
            attention = cls(self.config())
            out = attention(x)
            self.assertEqual(tuple(out.shape), (1, 8, 16))
            self.assertTrue(torch.isfinite(out).all())

            changed = x.detach().clone()
            changed[:, -1] = torch.randn_like(changed[:, -1])
            with torch.no_grad():
                a = attention(x.detach())
                b = attention(changed)
            self.assertTrue(
                torch.allclose(
                    a[:, :-1],
                    b[:, :-1],
                    atol=1e-5,
                    rtol=1e-4,
                )
            )

            loss = out.square().mean()
            loss.backward(retain_graph=True)
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            core_parameters = [
                attention.q_a_proj.weight,
                attention.q_b_proj.weight,
                attention.kv_proj.weight,
                attention.output.weight,
                attention.output.out_proj.weight,
            ]
            for parameter in core_parameters:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            # CSA's hard top-k indexer is intentionally trained by a separate
            # dense-teacher/indexer objective; LM loss does not differentiate
            # through the discrete selected indices.
            x.grad.zero_()

    def test_packed_documents_are_isolated(self):
        torch.manual_seed(32)
        docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        x_a = torch.randn(1, 8, 16)
        x_b = x_a.clone()
        x_b[:, :4] = torch.randn_like(x_b[:, :4])

        for cls in (
            CompressedSparseContextAttention,
            HeavilyCompressedContextAttention,
        ):
            attention = cls(self.config()).eval()
            with torch.no_grad():
                out_a = attention(x_a, document_ids=docs)
                out_b = attention(x_b, document_ids=docs)
            self.assertTrue(
                torch.allclose(
                    out_a[:, 4:],
                    out_b[:, 4:],
                    atol=1e-6,
                    rtol=1e-5,
                )
            )

    def test_padding_is_zero_and_does_not_contaminate_valid_tokens(self):
        torch.manual_seed(33)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        x_a = torch.randn(1, 6, 16)
        x_b = x_a.clone()
        x_b[:, 4:] = torch.randn_like(x_b[:, 4:])

        attention = CompressedSparseContextAttention(self.config()).eval()
        with torch.no_grad():
            out_a = attention(x_a, attention_mask=mask)
            out_b = attention(x_b, attention_mask=mask)
        self.assertTrue(torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(out_a[:, 4:], torch.zeros_like(out_a[:, 4:])))

    def test_closed_window_visibility_matches_deepseek_v4_reference(self):
        # Official DeepseekV4HCACompressor: (position_ids + 1) // rate.
        # Official CSA indexer uses the same closed-window threshold.
        torch.manual_seed(35)
        config = self.config()
        x = torch.randn(1, 9, config.hidden_size)
        for cls, rate in (
            (CompressedSparseContextAttention, config.csa_compress_rate),
            (HeavilyCompressedContextAttention, config.hca_compress_rate),
        ):
            attention = cls(config).eval()
            with torch.no_grad():
                compressed = attention._compress_main(
                    x[0], torch.arange(x.shape[1])
                )
            self.assertEqual(compressed.shape[0], x.shape[1] // rate)
            if cls is CompressedSparseContextAttention:
                with torch.no_grad():
                    scores = attention.indexer_scores(x)[0]
                expected = (
                    torch.arange(compressed.shape[0]).unsqueeze(0)
                    < ((torch.arange(x.shape[1]) + 1) // rate).unsqueeze(1)
                )
                self.assertTrue(torch.equal(scores.valid_mask.cpu(), expected))
                self.assertTrue(
                    torch.equal(scores.selected_indices[~expected.any(dim=1)], 
                                torch.full_like(scores.selected_indices[~expected.any(dim=1)], -1))
                )

    def test_csa_overlap_uses_previous_ca_and_current_cb(self):
        # DeepseekV4CSACompressor uses previous-window Ca and current-window Cb.
        torch.manual_seed(36)
        config = self.config()
        model = CompressedSparseContextAttention(config).eval()
        hidden = torch.randn(2 * config.csa_compress_rate, config.hidden_size)
        positions = torch.arange(hidden.shape[0])
        rate = config.csa_compress_rate
        with torch.no_grad():
            kv = model.compressor_kv_proj(hidden).view(2, rate, 2 * config.head_dim)
            gate = (
                model.compressor_gate_proj(hidden).view(2, rate, 2 * config.head_dim)
                + model.compressor_position_bias
            )
            combined_kv = torch.cat(
                [kv[0, :, :config.head_dim], kv[1, :, config.head_dim:]], dim=0
            )
            combined_gate = torch.cat(
                [gate[0, :, :config.head_dim], gate[1, :, config.head_dim:]], dim=0
            )
            weights = combined_gate.softmax(dim=0, dtype=torch.float32).to(kv.dtype)
            expected = model.compressor_kv_norm(
                (combined_kv * weights).sum(dim=0)
            )
            expected = model._rope(
                expected.view(1, 1, -1), positions[rate:rate + 1]
            ).squeeze(0).squeeze(0)
            actual = model._compress_main(hidden, positions)[1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_incremental_cache_matches_full_forward_with_unaligned_boundaries(self):
        # DeepSeek-V4's buffer, overlap and indexer state must survive
        # prefill -> decode -> prefill without recompressing open windows.
        torch.manual_seed(37)
        x = torch.randn(2, 13, self.config().hidden_size)
        for attention_cls, cache_cls in (
            (CompressedSparseContextAttention, DeepseekV4CSACache),
            (HeavilyCompressedContextAttention, DeepseekV4HCACache),
        ):
            attention = attention_cls(self.config()).eval()
            with torch.no_grad():
                expected = attention(x)
                for sizes in ((13,), (1,) * 13, (3, 1, 5, 4), (5, 8)):
                    cache = cache_cls(self.config())
                    outputs = []
                    start = 0
                    for size in sizes:
                        outputs.append(
                            attention(
                                x[:, start : start + size],
                                past_key_values=cache,
                            )
                        )
                        start += size
                    actual = torch.cat(outputs, dim=1)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    self.assertEqual(cache.cumulative_length, x.shape[1])
                    self.assertEqual(
                        cache.entry_count["compressor"],
                        x.shape[1] // cache.compress_rate,
                    )
                    self.assertEqual(
                        cache.buffer_kv["compressor"].shape[1],
                        x.shape[1] % cache.compress_rate,
                    )
                    if isinstance(cache, DeepseekV4CSACache):
                        self.assertEqual(
                            cache.entry_count["indexer"],
                            cache.entry_count["compressor"],
                        )
                        self.assertIsNotNone(cache.overlap_kv["compressor"])
                        self.assertIsNotNone(cache.overlap_kv["indexer"])
                    cache.reset()
                    self.assertEqual(cache.cumulative_length, 0)
                    self.assertEqual(cache.entry_count["compressor"], 0)
                    torch.testing.assert_close(
                        attention(x, past_key_values=cache),
                        expected,
                        atol=2e-5,
                        rtol=2e-5,
                    )

    def test_cache_rejects_cross_sequence_or_packed_state_reuse(self):
        attention = CompressedSparseContextAttention(self.config()).eval()
        cache = DeepseekV4CSACache(self.config())
        x = torch.randn(1, 3, self.config().hidden_size)
        with torch.no_grad():
            attention(x, past_key_values=cache)
        with self.assertRaisesRegex(ValueError, "position_ids"):
            attention(
                x[:, :1],
                position_ids=torch.tensor([[0]]),
                past_key_values=cache,
            )
        with self.assertRaisesRegex(ValueError, "packed"):
            attention(x[:, :1], document_ids=torch.zeros(1, 1, dtype=torch.long), past_key_values=cache)
        with self.assertRaisesRegex(ValueError, "batch size"):
            attention(torch.randn(2, 1, self.config().hidden_size), past_key_values=cache)

    def test_interleaved_partial_rope_inverse_round_trip(self):
        torch.manual_seed(34)
        rope = InterleavedRotaryEmbedding(4, 32, 10000.0)
        positions = torch.tensor([[0, 1, 2, 3]])
        cos, sin = rope.cos_sin(
            positions,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 2, 4, 8)
        rotated = apply_partial_rotary_at_end(x, cos, sin, 4)
        restored = apply_inverse_partial_rotary_at_end(rotated, cos, sin, 4)
        self.assertTrue(torch.allclose(x, restored, atol=1e-6, rtol=1e-6))

    def test_config_rejects_invalid_grouping_and_rotary_width(self):
        with self.assertRaises(ValueError):
            CompressedContextConfig(
                hidden_size=16,
                num_attention_heads=3,
                head_dim=8,
                q_lora_rank=8,
                partial_rotary_dim=4,
                max_position_embeddings=32,
                o_groups=2,
            )
        with self.assertRaises(ValueError):
            CompressedContextConfig(
                hidden_size=16,
                num_attention_heads=4,
                head_dim=8,
                q_lora_rank=8,
                partial_rotary_dim=10,
                max_position_embeddings=32,
            )


if __name__ == "__main__":
    unittest.main()
