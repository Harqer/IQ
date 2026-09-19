from __future__ import annotations

import unittest
import numpy as np

from iq_model import IQModelConfig
from iq_transfer import (
    GQATransportError,
    PhiLayerMapIds,
    PhiPipelineError,
    build_donor_manifest,
    build_phi_dense_plan_spec,
    build_phi_dense_transport_plan,
    fit_gqa_group_projection,
)
from iq_transfer.donor import MappingTensorSource
from iq_transfer.phi4 import Phi4Inspector


class PhiPipelineTests(unittest.TestCase):
    def manifest(self):
        cfg = {
            "_name_or_path": "phi-test",
            "model_type": "phi3",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
        }
        tensors = {}
        for layer in range(2):
            p = f"model.layers.{layer}"
            tensors[f"{p}.self_attn.qkv_proj.weight"] = np.zeros((16, 8))
            tensors[f"{p}.self_attn.o_proj.weight"] = np.zeros((8, 8))
            tensors[f"{p}.mlp.gate_up_proj.weight"] = np.zeros((24, 8))
            tensors[f"{p}.mlp.down_proj.weight"] = np.zeros((8, 12))
        tensors["model.embed_tokens.weight"] = np.zeros((32, 8))
        tensors["lm_head.weight"] = np.zeros((32, 8))
        inspector = Phi4Inspector.from_config_mapping(cfg)
        source = MappingTensorSource(tensors)
        return build_donor_manifest(
            inspector.config,
            source,
            donor_id="phi",
            checkpoint_revision="x",
            tokenizer_hash="t",
            license="l",
            operator_layout_version="v1",
            source_uri="memory://x",
            allow_metadata_only=True,
        )

    def test_gqa_projection_preserves_group_structure(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(100, 4))
        projection = fit_gqa_group_projection(
            x,
            source_q_heads=4,
            source_kv_heads=2,
            target_q_heads=2,
            target_kv_heads=1,
            head_dim=2,
        )
        self.assertEqual(projection.group_map.shape, (2, 1))
        self.assertEqual(projection.q_map.matrix.shape, (8, 4))
        self.assertEqual(projection.kv_map.matrix.shape, (4, 2))
        self.assertTrue(np.allclose(projection.group_map.T @ projection.group_map, np.eye(1)))
        with self.assertRaises(GQATransportError):
            fit_gqa_group_projection(
                x,
                source_q_heads=4,
                source_kv_heads=2,
                target_q_heads=3,
                target_kv_heads=1,
                head_dim=2,
            )

    def test_full_phi_plan_covers_dense_recipient(self):
        recipient = IQModelConfig(
            vocab_size=32,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=6,
            max_position_embeddings=16,
        )
        ids = PhiLayerMapIds(
            "attn_in", "q", "kv", "attn_out", "mlp_in", "mlp_hidden", "mlp_out"
        )
        spec = build_phi_dense_plan_spec(
            recipient_config=recipient,
            donor_manifest=self.manifest(),
            calibration_manifest_hash="cal",
            layer_mapping={0: 1},
            layer_maps={0: ids},
            embedding_map_id="embed",
            final_residual_map_id="final",
        )
        plan = build_phi_dense_transport_plan(spec)
        by_path = {a.target_module_path: a for a in plan.assignments}
        self.assertIn("blocks.0.attn.q_proj", by_path)
        self.assertIn("blocks.0.mlp.down_proj", by_path)
        self.assertEqual(by_path["blocks.0.attn.k_proj"].output_map_id, "kv")
        self.assertEqual(by_path["blocks.0.attn.v_proj"].output_map_id, "kv")
        self.assertEqual(by_path["blocks.0.attn.o_proj"].input_map_id, "q")
        self.assertEqual(by_path["blocks.0.mlp.gate_proj"].output_map_id, "mlp_hidden")
        self.assertEqual(len(plan.assignments), 12)

    def test_phi_plan_rejects_non_monotonic_layers(self):
        recipient = IQModelConfig(
            vocab_size=32,
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=6,
            max_position_embeddings=16,
        )
        ids = PhiLayerMapIds("a", "q", "kv", "ao", "mi", "mh", "mo")
        with self.assertRaises(PhiPipelineError):
            build_phi_dense_plan_spec(
                recipient_config=recipient,
                donor_manifest=self.manifest(),
                calibration_manifest_hash="cal",
                layer_mapping={0: 1, 1: 0},
                layer_maps={0: ids, 1: ids},
                embedding_map_id="e",
                final_residual_map_id="f",
            )


if __name__ == "__main__":
    unittest.main()
