from __future__ import annotations

import unittest

import torch

from iq_model import (
    CompressedContextConfig,
    CompressedSparseContextAttention,
    MHCConfig,
    MHCHead,
    ManifoldHyperConnection,
    expand_mhc_streams,
    lightning_indexer_kl_loss,
    lightning_indexer_topk_recall,
    sinkhorn_doubly_stochastic,
)


class MHCAndIndexerTrainingTests(unittest.TestCase):
    def test_sinkhorn_and_mhc_forward_backward(self):
        torch.manual_seed(41)
        raw = torch.rand(2, 3, 4, 4) + 0.1
        projected = sinkhorn_doubly_stochastic(
            raw,
            iterations=30,
            eps=1e-8,
        )
        rows = projected.sum(dim=-1)
        cols = projected.sum(dim=-2)
        self.assertTrue(
            torch.allclose(rows, torch.ones_like(rows), atol=2e-4, rtol=2e-4)
        )
        self.assertTrue(
            torch.allclose(cols, torch.ones_like(cols), atol=2e-4, rtol=2e-4)
        )

        config = MHCConfig(
            hidden_size=8,
            streams=4,
            sinkhorn_iters=30,
            eps=1e-8,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
        )
        hc = ManifoldHyperConnection(config)
        head = MHCHead(config)
        base = torch.randn(2, 5, 8, requires_grad=True)
        streams = expand_mhc_streams(base, streams=4)
        weights = hc(streams)
        self.assertEqual(tuple(weights.collapsed.shape), (2, 5, 8))
        self.assertEqual(tuple(weights.post.shape), (2, 5, 4))
        self.assertEqual(tuple(weights.comb.shape), (2, 5, 4, 4))

        comb_rows = weights.comb.sum(dim=-1)
        comb_cols = weights.comb.sum(dim=-2)
        self.assertTrue(
            torch.allclose(
                comb_rows,
                torch.ones_like(comb_rows),
                atol=3e-4,
                rtol=3e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                comb_cols,
                torch.ones_like(comb_cols),
                atol=3e-4,
                rtol=3e-4,
            )
        )

        block_output = torch.tanh(weights.collapsed)
        merged = hc.merge(streams, block_output, weights)
        final = head(merged)
        self.assertEqual(tuple(merged.shape), (2, 5, 4, 8))
        self.assertEqual(tuple(final.shape), (2, 5, 8))
        self.assertTrue(torch.isfinite(final).all())

        final.square().mean().backward()
        self.assertIsNotNone(base.grad)
        self.assertTrue(torch.isfinite(base.grad).all())
        for module in (hc, head):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def compressed_config(self) -> CompressedContextConfig:
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

    def test_indexer_distillation_drives_indexer_parameters(self):
        torch.manual_seed(42)
        attention = CompressedSparseContextAttention(
            self.compressed_config()
        )
        x = torch.randn(1, 8, 16, requires_grad=True)
        examples = attention.indexer_scores(x)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(tuple(example.scores.shape), (8, 4))
        self.assertEqual(tuple(example.valid_mask.shape), (8, 4))
        self.assertEqual(tuple(example.selected_indices.shape), (8, 2))

        teacher = torch.randn_like(example.scores)
        loss = lightning_indexer_kl_loss(
            example.scores,
            teacher,
            example.valid_mask,
            temperature=1.5,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        required = [
            attention.index_kv_proj,
            attention.index_gate_proj,
            attention.index_q_proj,
            attention.index_weight_proj,
        ]
        for module in required:
            self.assertIsNotNone(module)
            assert module is not None
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())

        recall = lightning_indexer_topk_recall(
            example.scores.detach(),
            teacher,
            example.valid_mask,
            topk=2,
        )
        self.assertGreaterEqual(float(recall), 0.0)
        self.assertLessEqual(float(recall), 1.0)

    def test_indexer_scores_preserve_packed_document_segments(self):
        torch.manual_seed(43)
        attention = CompressedSparseContextAttention(
            self.compressed_config()
        )
        x = torch.randn(1, 8, 16)
        docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        examples = attention.indexer_scores(
            x,
            document_ids=docs,
        )
        self.assertEqual(len(examples), 2)
        self.assertTrue(
            torch.equal(
                examples[0].token_indices,
                torch.tensor([0, 1, 2, 3]),
            )
        )
        self.assertTrue(
            torch.equal(
                examples[1].token_indices,
                torch.tensor([4, 5, 6, 7]),
            )
        )


if __name__ == "__main__":
    unittest.main()
