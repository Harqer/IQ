"""Small CPU tests of the real saved-shard HCA test's strict loading and causal execution."""
from dataclasses import asdict
import unittest

import torch

from iq_model.architecture import HybridLayerType
from iq_model.attention.compressed import CompressedContextConfig
from iq_model.config import IQModelConfig
from iq_model.hybrid import CompressedContextResidualLayer
from scripts.test_glm53_real_context_layer import build_context_layer, exercise_context


class TestRealContextLayer(unittest.TestCase):
    def setUp(self):
        self.model = IQModelConfig(
            vocab_size=128, hidden_size=32,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        )
        self.context = CompressedContextConfig(
            hidden_size=32, num_attention_heads=4, head_dim=8,
            q_lora_rank=16, partial_rotary_dim=4,
            max_position_embeddings=64, hca_compress_rate=4,
            csa_compress_rate=2, sliding_window=4,
            o_groups=2, o_lora_rank=8,
            index_n_heads=2, index_head_dim=8, index_topk=4,
        )
        self.config = {
            "model": asdict(self.model), "compressed_context": asdict(self.context),
            "schedule": {"layers": ["mamba3"] * 10 + ["hca"] + ["moe"]},
        }
        torch.manual_seed(32)
        layer = CompressedContextResidualLayer(
            self.model, self.context, mode=HybridLayerType.HCA,
        )
        self.shard = {
            "layers.10." + name: value.detach().to(torch.bfloat16)
            for name, value in layer.state_dict().items()
        }

    def test_saved_hca_forward_and_causal_invariance(self):
        layer = build_context_layer(self.config, self.shard)
        result = exercise_context(
            layer, device=torch.device("cpu"), seed=31, sequence=12,
        )
        self.assertEqual(result["mode"], "hca")
        self.assertEqual(result["causal_prefix_max_difference"], 0.)
        self.assertGreater(result["output_projection_ablation_max_difference"], 0.)

    def test_missing_saved_tensor_rejected(self):
        shard = dict(self.shard)
        del shard["layers.10.norm.weight"]
        with self.assertRaisesRegex(RuntimeError, "Saved HCA tensor coverage mismatch"):
            build_context_layer(self.config, shard)

    def test_bad_tensor_shape_rejected(self):
        shard = dict(self.shard)
        shard["layers.10.norm.weight"] = torch.ones(2, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "Saved HCA tensor invalid"):
            build_context_layer(self.config, shard)


if __name__ == "__main__":
    unittest.main()
