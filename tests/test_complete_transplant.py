from __future__ import annotations

import math
import unittest

import torch

from iq_model import (
    BlockAttnResConfig,
    CompressedContextConfig,
    CompressedSparseContextAttention,
    StableLatentMoE,
    StableLatentMoEConfig,
)
from iq_model.residual.attnres import AttentionResidualMixer
from iq_transfer.complete_transplant import (
    RESIDUAL_EMBED_SCALE,
    _embed_residual_output,
    _project_residual_input,
    canonical_complete_config,
    canonical_complete_schedule,
    donor_layer_positions,
    expected_complete_state_keys,
)
from iq_transfer.gpt_oss20b import GptOss20BConfig, GptOss20BError


class CompleteTransplantTests(unittest.TestCase):
    def test_canonical_schedule_maps_all_donor_layers(self):
        schedule = canonical_complete_schedule()
        self.assertEqual(len(schedule.layers), 80)
        self.assertEqual(schedule.to_tokens().count("mamba3"), 32)
        self.assertEqual(schedule.to_tokens().count("moe"), 24)
        self.assertEqual(schedule.to_tokens().count("csa"), 12)
        self.assertEqual(schedule.to_tokens().count("hca"), 12)

        config = canonical_complete_config()
        mapping = donor_layer_positions(config)
        self.assertEqual(len(mapping), 24)
        self.assertEqual(config.mamba3.num_layers, 32)
        self.assertEqual(config.model.hidden_size, 4096)
        self.assertEqual(config.model.vocab_size, 201088)
        self.assertEqual(config.stable_moe.latent_size, 2880)
        self.assertEqual(config.stable_moe.num_experts, 32)
        self.assertTrue(config.stable_moe.expert_bias)
        self.assertTrue(config.stable_moe.router_bias)

    def test_complete_state_schema_is_self_consistent(self):
        config = canonical_complete_config()
        keys = expected_complete_state_keys(config)
        self.assertIn("embed_tokens.weight", keys)
        self.assertIn("lm_head.weight", keys)
        self.assertIn("layers.0.mamba.core.in_proj.weight", keys)
        context_physical, moe_physical = donor_layer_positions(config)[0]
        self.assertIn(
            f"layers.{context_physical}.attention.q_a_proj.weight",
            keys,
        )
        self.assertIn(
            f"layers.{moe_physical}.moe.routed_experts.0.gate_proj.weight",
            keys,
        )
        self.assertIn("attnres.mixers.0.mix_logit", keys)
        self.assertIn("reasoning_injector.gate_logit", keys)

    def test_residual_embedding_scale_preserves_rms(self):
        torch.manual_seed(1)
        source = torch.randn(5, 2880)
        target = torch.zeros(5, 4096)
        target[:, :2880] = source * RESIDUAL_EMBED_SCALE
        source_rms = source.square().mean(dim=-1)
        target_rms = target.square().mean(dim=-1)
        self.assertTrue(torch.allclose(source_rms, target_rms, atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(
            RESIDUAL_EMBED_SCALE,
            math.sqrt(4096 / 2880),
            places=12,
        )

    def test_input_and_output_operator_transport_preserve_embedded_function(self):
        torch.manual_seed(2)
        x = torch.randn(3, 2880)
        input_op = torch.randn(7, 2880)
        target_x = torch.zeros(3, 4096)
        target_x[:, :2880] = x * RESIDUAL_EMBED_SCALE

        transported_input = _project_residual_input(input_op)
        self.assertTrue(
            torch.allclose(
                target_x @ transported_input.T,
                x @ input_op.T,
                atol=1e-4,
                rtol=1e-4,
            )
        )

        output_op = torch.randn(2880, 7)
        y_source = (x @ input_op.T) @ output_op.T
        transported_output = _embed_residual_output(output_op)
        y_target = (target_x @ transported_input.T) @ transported_output.T
        expected = torch.zeros(3, 4096)
        expected[:, :2880] = y_source * RESIDUAL_EMBED_SCALE
        self.assertTrue(torch.allclose(y_target, expected, atol=1e-3, rtol=1e-3))

    def test_stable_latent_moe_can_retain_donor_biases(self):
        config = StableLatentMoEConfig(
            hidden_size=16,
            latent_size=8,
            expert_intermediate_size=12,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            expert_bias=True,
            router_bias=True,
        )
        moe = StableLatentMoE(config)
        self.assertIsNotNone(moe.router.bias)
        self.assertIsNotNone(moe.routed_experts[0].gate_proj.bias)
        self.assertIsNotNone(moe.routed_experts[0].up_proj.bias)
        self.assertIsNotNone(moe.routed_experts[0].down_proj.bias)

    def test_compressed_context_can_retain_projection_biases(self):
        config = CompressedContextConfig(
            hidden_size=16,
            num_attention_heads=4,
            head_dim=4,
            q_lora_rank=8,
            partial_rotary_dim=2,
            max_position_embeddings=32,
            sliding_window=4,
            csa_compress_rate=2,
            hca_compress_rate=4,
            o_groups=2,
            o_lora_rank=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            compress_rope_theta=10000.0,
            projection_bias=True,
        )
        attention = CompressedSparseContextAttention(config)
        self.assertIsNotNone(attention.q_a_proj.bias)
        self.assertIsNotNone(attention.q_b_proj.bias)
        self.assertIsNotNone(attention.kv_proj.bias)
        self.assertIsNotNone(attention.output.out_proj.bias)

    def test_attnres_native_init_stays_on_residual_baseline(self):
        config = BlockAttnResConfig(hidden_size=8, num_layers=2, block_size=2)
        mixer = AttentionResidualMixer(config)
        blocks = torch.randn(2, 3, 2, 8)
        prefix = torch.randn(2, 3, 8)
        baseline = blocks.sum(dim=2) + prefix
        output = mixer(blocks, prefix)
        # sigmoid(-12) ~= 6e-6, so the native mixer is effectively transparent.
        self.assertTrue(torch.allclose(output, baseline, atol=2e-4, rtol=2e-4))

    def test_gpt_oss_config_is_fail_closed(self):
        good = {
            "num_hidden_layers": 24,
            "num_experts": 32,
            "experts_per_token": 4,
            "vocab_size": 201088,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "swiglu_limit": 7.0,
            "head_dim": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "sliding_window": 128,
            "initial_context_length": 4096,
            "rope_theta": 150000.0,
            "rope_scaling_factor": 32.0,
            "rope_ntk_alpha": 1.0,
            "rope_ntk_beta": 32.0,
        }
        parsed = GptOss20BConfig.from_mapping(good)
        self.assertEqual(parsed.hidden_size, 2880)
        with self.assertRaises(GptOss20BError):
            GptOss20BConfig.from_mapping({**good, "num_hidden_layers": 25})


if __name__ == "__main__":
    unittest.main()
