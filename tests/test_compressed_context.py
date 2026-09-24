from __future__ import annotations

import unittest

import torch

from iq_model import (
    CompressedContextConfig,
    CompressedSparseContextAttention,
    HeavilyCompressedContextAttention,
)
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
