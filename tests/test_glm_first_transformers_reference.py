"""Real Transformers GLM decoder API smoke test (tiny random CPU weights only).

This validates overlay hooks against GlmMoeDsaDecoderLayer's *actual* API:
its forward returns (hidden_states, topk_indices). This is not a proof
of GLM-5.3 pretrained teacher parity or full Mamba CUDA kernel behavior.
"""
from __future__ import annotations

import unittest

import torch

from iq_model.glm_first_mamba import (
    GLMFirstMambaHybrid,
    GLMFirstMambaContract,
)
from test_glm_first_mamba_hybrid import ToyMambaBlock


class RealGlmDecoderOverlayTests(unittest.TestCase):
    def test_zero_gate_on_actual_hf_glm_decoder(self):
        from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

        torch.manual_seed(37)
        config = GlmMoeDsaConfig(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            first_k_dense_replace=3,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            index_topk=4,
            index_head_dim=8,
            index_n_heads=2,
            indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
            max_position_embeddings=64,
            use_cache=False,
            bos_token_id=1,
            eos_token_id=[15, 16],
            pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            attention_dropout=0.0,
        )
        reference = GlmMoeDsaForCausalLM(config).eval()
        contract = GLMFirstMambaContract(
            glm_vocab=17, glm_hidden=8, glm_layers=6,
            mamba_hidden=4, mamba_layers=3, mamba_mlp_hidden=8,
            mimo_rank=4, state_size=2,
        )
        hybrid = GLMFirstMambaHybrid(
            reference,
            [ToyMambaBlock() for _ in range(3)],
            contract=contract,
            placements=(0, 2, 5),
        ).eval()
        input_ids = torch.tensor([[1, 3, 4, 7]])
        with torch.inference_mode():
            teacher_logits = reference(input_ids, use_cache=False).logits
            student_logits = hybrid(input_ids, use_cache=False).logits
        torch.testing.assert_close(student_logits, teacher_logits, rtol=0, atol=0)
        self.assertEqual(
            sum(len(block._forward_hooks) for block in reference.model.layers), 0
        )
        # Controlled full-prefix hybrid decoding, still with zero gates,
        # must match recomputation using the native Transformers GLM model.
        hybrid.enable_branches()
        with torch.inference_mode():
            expected = input_ids
            for _ in range(2):
                next_logits = reference(expected, use_cache=False).logits
                next_token = next_logits[:, -1, :].argmax(-1, keepdim=True)
                expected = torch.cat((expected, next_token), dim=1)
            actual = hybrid.experimental_generate_uncached(
                input_ids, max_new_tokens=2, stop_token_ids=()
            )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            sum(len(block._forward_hooks) for block in reference.model.layers), 0
        )


if __name__ == "__main__":
    unittest.main()
