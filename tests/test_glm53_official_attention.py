"""Source-native HF GLM MLA+DSA CPU attention acceptance.

This test uses actual transformers==5.15.0 GlmMoeDsaAttention and its
rotary/indexer operators, with small randomly initialized *test parameters*.
It makes no claim about pretrained donor values or IQ CSA equivalence.
"""
from __future__ import annotations

import os
import unittest

import torch

from iq_transfer.glm53_official_attention import (
    _hf_attention_classes,
    GLM53OfficialAttentionError,
    official_attention_prefill,
)


class OfficialGLMAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.Config, cls.Attention, _ = _hf_attention_classes()
        except GLM53OfficialAttentionError as error:
            if os.environ.get("IQ_REQUIRE_HF_GLM_ATTENTION") == "1":
                raise
            raise unittest.SkipTest(str(error))

    def make_attention(self, layer_index: int):
        config = self.Config(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            head_dim=4,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=3,
            intermediate_size=32,
            first_k_dense_replace=0,
            n_routed_experts=4,
            num_experts_per_tok=2,
            indexer_types=["full", "shared"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
            max_position_embeddings=128,
            attention_dropout=0.0,
            attention_bias=False,
        )
        config._attn_implementation = "eager"
        return self.Attention(config, layer_idx=layer_index).eval()

    def test_actual_official_full_indexer_outputs_and_causal_mask(self):
        torch.manual_seed(1003)
        layer = self.make_attention(0)
        self.assertIsNotNone(layer.indexer)
        source_state = layer.state_dict()
        for key in (
            "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
            "kv_b_proj.weight", "o_proj.weight", "indexer.wq_b.weight",
            "indexer.wk.weight", "indexer.weights_proj.weight",
            "indexer.k_norm.weight", "indexer.k_norm.bias",
        ):
            self.assertIn(key, source_state)
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            output, indices = official_attention_prefill(layer, x)
            perturbed = x.clone()
            perturbed[:, -1] += 999
            reference, ref_indices = official_attention_prefill(layer, perturbed)
        self.assertEqual(output.shape, (2, 5, 16))
        self.assertEqual(indices.shape, (2, 5, 3))
        torch.testing.assert_close(
            output[:, :-1], reference[:, :-1], rtol=1e-5, atol=1e-5,
        )
        self.assertTrue(torch.equal(indices[:, :-1], ref_indices[:, :-1]))
        # For the first query only key 0 is causal. Top-k can still return
        # masked future positions when k>1; the official additive causal
        # mask must remove their contribution from output, not the indexer.
        self.assertTrue(torch.any(indices[:, 0] == 0, dim=-1).all())
        self.assertTrue(torch.isfinite(output).all())

    def test_actual_official_shared_indexer_uses_previous_indices(self):
        torch.manual_seed(1012)
        full = self.make_attention(0)
        shared = self.make_attention(1)
        self.assertIsNone(shared.indexer)
        x = torch.randn(1, 5, 16)
        with torch.no_grad():
            _, topk = official_attention_prefill(full, x)
            output, reused = official_attention_prefill(
                shared, x, previous_indices=topk,
            )
        self.assertEqual(tuple(output.shape), (1, 5, 16))
        self.assertTrue(torch.equal(topk, reused))
        with self.assertRaisesRegex(ValueError, "Shared DSA"):
            official_attention_prefill(shared, x)


if __name__ == "__main__":
    unittest.main()
