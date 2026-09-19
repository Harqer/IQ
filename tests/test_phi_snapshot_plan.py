from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from iq_model import IQModelConfig
from iq_transfer import (
    CalibrationError,
    DenseLayerMapIds,
    DensePlanError,
    build_calibration_manifest,
    build_phi_dense_transport_plan,
    open_local_phi_snapshot,
)


class PhiSnapshotDensePlanTests(unittest.TestCase):
    def write_phi_snapshot(self, root: Path) -> None:
        config = {
            "_name_or_path": "test-phi4",
            "model_type": "phi3",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "torch_dtype": "float32",
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 128, "tokenizer_class": "PreTrainedTokenizerFast"}),
            encoding="utf-8",
        )
        tensors = {
            "model.embed_tokens.weight": np.zeros((32, 8), dtype=np.float32),
            "lm_head.weight": np.zeros((32, 8), dtype=np.float32),
        }
        for layer in range(4):
            prefix = f"model.layers.{layer}"
            tensors[f"{prefix}.self_attn.qkv_proj.weight"] = np.zeros((16, 8), dtype=np.float32)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = np.zeros((8, 8), dtype=np.float32)
            tensors[f"{prefix}.mlp.gate_up_proj.weight"] = np.zeros((24, 8), dtype=np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = np.zeros((8, 12), dtype=np.float32)
        save_file(tensors, str(root / "model.safetensors"))

    def test_calibration_manifest_is_content_addressed(self):
        samples = [
            ("a", "code", "def add(a, b):\n    return a + b\n"),
            ("b", "reasoning", "If x is 3, compute x * 2."),
        ]
        first = build_calibration_manifest("proof", samples)
        second = build_calibration_manifest("proof", samples)
        changed = build_calibration_manifest(
            "proof",
            [samples[0], ("b", "reasoning", "If x is 4, compute x * 2.")],
        )
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        with self.assertRaises(CalibrationError):
            build_calibration_manifest("proof", [("a", "code", "x"), ("a", "code", "y")])

    def test_local_phi_snapshot_hashes_real_weights_and_tokenizer_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_phi_snapshot(root)
            first = open_local_phi_snapshot(root, checkpoint_revision="revision-1", license="MIT")
            self.assertTrue(first.manifest.files)
            self.assertEqual(first.manifest.num_layers, 4)
            self.assertEqual(first.manifest.hidden_size, 8)
            first_fingerprint = first.manifest.fingerprint
            first_tokenizer = first.manifest.tokenizer_hash

            (root / "tokenizer_config.json").write_text(
                json.dumps({"model_max_length": 256, "tokenizer_class": "PreTrainedTokenizerFast"}),
                encoding="utf-8",
            )
            second = open_local_phi_snapshot(root, checkpoint_revision="revision-1", license="MIT")
            self.assertNotEqual(first_tokenizer, second.manifest.tokenizer_hash)
            self.assertNotEqual(first_fingerprint, second.manifest.fingerprint)

    def test_dense_plan_covers_every_projection_and_uses_layer_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_phi_snapshot(root)
            snapshot = open_local_phi_snapshot(root, checkpoint_revision="revision-1", license="MIT")

            target = IQModelConfig(
                vocab_size=32,
                hidden_size=4,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=6,
                max_position_embeddings=16,
                rope_theta=10000.0,
            )
            layer_mapping = {0: 0, 1: 3}
            maps = {
                layer: DenseLayerMapIds(
                    residual_in=f"resid-in-{layer}",
                    q=f"q-{layer}",
                    k=f"k-{layer}",
                    v=f"v-{layer}",
                    attn_in=f"attn-in-{layer}",
                    residual_out=f"resid-out-{layer}",
                    gate=f"gate-{layer}",
                    up=f"up-{layer}",
                    mlp_in=f"mlp-in-{layer}",
                )
                for layer in range(2)
            }
            calibration = build_calibration_manifest("proof", [("a", "code", "x = 1\n")])
            plan = build_phi_dense_transport_plan(
                target,
                snapshot.manifest,
                layer_mapping=layer_mapping,
                layer_maps=maps,
                calibration_manifest_hash=calibration.fingerprint,
                embedding_map_id="embedding",
                lm_head_map_id="lm-head",
            )

            # embedding + lm head + 2 * (7 transported operators + 2 native norms) + final norm
            self.assertEqual(len(plan.assignments), 21)
            layer1 = [
                assignment
                for assignment in plan.assignments
                if assignment.target_module_path.startswith("blocks.1.")
                and assignment.transfer_method.value == "operator_transport"
            ]
            self.assertEqual(len(layer1), 7)
            self.assertTrue(all(assignment.source_layer == 3 for assignment in layer1))

            q0 = next(
                assignment
                for assignment in plan.assignments
                if assignment.target_module_path == "blocks.0.attn.q_proj"
            )
            k0 = next(
                assignment
                for assignment in plan.assignments
                if assignment.target_module_path == "blocks.0.attn.k_proj"
            )
            self.assertEqual(q0.target_shape, (4, 4))
            self.assertEqual(k0.target_shape, (2, 4))

            with self.assertRaises(DensePlanError):
                build_phi_dense_transport_plan(
                    target,
                    snapshot.manifest,
                    layer_mapping={0: 0},
                    layer_maps=maps,
                    calibration_manifest_hash=calibration.fingerprint,
                    embedding_map_id="embedding",
                    lm_head_map_id="lm-head",
                )


if __name__ == "__main__":
    unittest.main()
