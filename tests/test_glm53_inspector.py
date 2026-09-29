from __future__ import annotations

import unittest

import torch

from iq_transfer import (
    DonorError,
    GLM53Inspector,
    MappingTensorSource,
)


class GLM53InspectorTests(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return {
            "_name_or_path": "zai-org/GLM-5.3-BF16",
            "model_type": "glm_moe_dsa",
            "dtype": "bfloat16",
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "q_lora_rank": 8,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 3,
            "qk_rope_head_dim": 1,
            "v_head_dim": 4,
            "moe_intermediate_size": 6,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "scoring_func": "sigmoid",
            "index_head_dim": 4,
            "index_n_heads": 2,
            "indexer_types": ["full", "sparse"],
        }

    def source(self) -> MappingTensorSource:
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.empty(32, 16),
            "model.norm.weight": torch.empty(16),
            "lm_head.weight": torch.empty(32, 16),
        }
        for layer in range(2):
            p = f"model.layers.{layer}"
            tensors[f"{p}.input_layernorm.weight"] = torch.empty(16)
            tensors[f"{p}.post_attention_layernorm.weight"] = torch.empty(16)
            tensors[f"{p}.self_attn.q_a_proj.weight"] = torch.empty(8, 16)
            tensors[f"{p}.self_attn.q_a_layernorm.weight"] = torch.empty(8)
            tensors[f"{p}.self_attn.q_b_proj.weight"] = torch.empty(8, 8)
            tensors[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = torch.empty(5, 16)
            tensors[f"{p}.self_attn.kv_a_layernorm.weight"] = torch.empty(4)
            tensors[f"{p}.self_attn.kv_b_proj.weight"] = torch.empty(14, 4)
            tensors[f"{p}.self_attn.o_proj.weight"] = torch.empty(16, 8)
            tensors[f"{p}.self_attn.indexer.wq_b.weight"] = torch.empty(8, 8)
            tensors[f"{p}.self_attn.indexer.wk.weight"] = torch.empty(4, 16)
            tensors[f"{p}.self_attn.indexer.weights_proj.weight"] = torch.empty(2, 16)
            tensors[f"{p}.self_attn.indexer.k_norm.weight"] = torch.empty(4)
            tensors[f"{p}.self_attn.indexer.k_norm.bias"] = torch.empty(4)
        p = "model.layers.0.mlp"
        tensors[f"{p}.gate_proj.weight"] = torch.empty(24, 16)
        tensors[f"{p}.up_proj.weight"] = torch.empty(24, 16)
        tensors[f"{p}.down_proj.weight"] = torch.empty(16, 24)

        p = "model.layers.1.mlp"
        tensors[f"{p}.gate.weight"] = torch.empty(2, 16)
        tensors[f"{p}.shared_experts.gate_proj.weight"] = torch.empty(6, 16)
        tensors[f"{p}.shared_experts.up_proj.weight"] = torch.empty(6, 16)
        tensors[f"{p}.shared_experts.down_proj.weight"] = torch.empty(16, 6)
        for expert in range(2):
            ep = f"{p}.experts.{expert}"
            tensors[f"{ep}.gate_proj.weight"] = torch.empty(6, 16)
            tensors[f"{ep}.up_proj.weight"] = torch.empty(6, 16)
            tensors[f"{ep}.down_proj.weight"] = torch.empty(16, 6)
        return MappingTensorSource(tensors)

    def test_validates_real_glm_attention_and_moe_layout(self):
        inspector = GLM53Inspector.from_config_mapping(self.config())
        report = inspector.validate_checkpoint(self.source())
        self.assertTrue(report.ok)
        self.assertEqual(report.warnings, ())

        roles = {ref.role for ref in inspector.operators(self.source())}
        self.assertIn("attn.q_a", roles)
        self.assertIn("attn.kv_b", roles)
        self.assertIn("dsa.indexer.q", roles)
        self.assertIn("mlp.gate", roles)
        self.assertIn("moe.router", roles)
        self.assertIn("moe.expert.1.down", roles)
        self.assertIn("embedding", roles)
        self.assertIn("lm_head", roles)

    def test_missing_expert_tensor_fails_closed(self):
        inspector = GLM53Inspector.from_config_mapping(self.config())
        source = self.source()
        tensors = dict(source._tensors)
        del tensors["model.layers.1.mlp.experts.1.down_proj.weight"]
        report = inspector.validate_checkpoint(MappingTensorSource(tensors))
        self.assertFalse(report.ok)
        self.assertIn("missing GLM-5.3 checkpoint tensor", report.errors[0])

    def test_non_sigmoid_router_is_rejected(self):
        config = self.config()
        config["scoring_func"] = "softmax"
        with self.assertRaisesRegex(DonorError, "sigmoid"):
            GLM53Inspector.from_config_mapping(config)


if __name__ == "__main__":
    unittest.main()
