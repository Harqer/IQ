"""CPU algorithmic acceptance for real IQ CSA stagewise alignment.

The small numerical test verifies real torch autograd/AdamW and source/target
alignment contracts. Tiny synthetic data is NOT a claim to donor-weight parity.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from iq_model import CompressedContextConfig, CompressedSparseContextAttention
from iq_transfer.glm53_attention_alignment import (
    GLM53AttentionAlignmentError, PairedAttentionBatch,
    evaluate_alignment, fit_attention_from_paired_activations, validate_pairing,
)
from iq_transfer.transport import CoordinateMap, MapDiagnostics
from scripts.distill_glm53_attention_stage import checksum, load_paired


def small_config() -> CompressedContextConfig:
    return CompressedContextConfig(
        hidden_size=8, num_attention_heads=2, head_dim=4,
        q_lora_rank=4, partial_rotary_dim=2,
        max_position_embeddings=32, sliding_window=3,
        csa_compress_rate=2, hca_compress_rate=4,
        o_groups=2, o_lora_rank=2,
        index_n_heads=2, index_head_dim=4, index_topk=2,
        compress_rope_theta=10000.0,
        direct_token_indexer=True,
        normalize_query_output=False,
        normalize_candidate_nonrotary_only=True,
        projection_bias=False,
    )


def map_data() -> CoordinateMap:
    return CoordinateMap(
        matrix=np.eye(8, dtype=np.float64),
        ridge=0.001,
        source_space="test.teacher",
        target_space="test.student",
        diagnostics=MapDiagnostics(
            sample_count=64, source_features=8, target_features=8,
            effective_rank=8, condition_number=1.0,
            fit_rmse=0.0, validation_rmse=0.0,
        )
    )


class StagewiseAlignmentTests(unittest.TestCase):
    def make_real_small_module_pairs(self):
        torch.manual_seed(14)
        teacher = CompressedSparseContextAttention(small_config()).eval()
        student = CompressedSparseContextAttention(small_config())
        x_train = torch.randn(2, 5, 8)
        x_val = torch.randn(2, 5, 8)
        with torch.no_grad():
            # Algorithm test only: another real CSA module supplies a fixed
            # numerical training target. Production uses official GLM.
            output_train = teacher(x_train)
            output_val = teacher(x_val)
        train = PairedAttentionBatch(
            iq_attention_input=x_train,
            glm_attention_output=output_train,
            token_ids=torch.tensor([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]]),
            dsa_indices=torch.zeros(2, 5, 2, dtype=torch.int32),
        )
        val = PairedAttentionBatch(
            iq_attention_input=x_val,
            glm_attention_output=output_val,
            token_ids=torch.tensor([[6, 7, 8, 9, 10], [21, 22, 23, 24, 25]]),
            dsa_indices=torch.zeros(2, 5, 2, dtype=torch.int32),
        )
        return student, train, val

    def test_actual_csa_autograd_alignment_and_best_holdout_restore(self):
        student, train, val = self.make_real_small_module_pairs()
        record = fit_attention_from_paired_activations(
            student, train, val, map_data(),
            steps=5, learning_rate=0.004,
        )
        self.assertEqual(record["stage_training_steps"], 5)
        self.assertFalse(record["source_operator_equivalence"])
        self.assertFalse(record["full_model_pretrained"])
        self.assertGreaterEqual(record["heldout_dsa_topk_recall"], 0.0)
        self.assertLessEqual(record["heldout_dsa_topk_recall"], 1.0)
        self.assertTrue(record["heldout_student_causal"]["causal_pass"])
        self.assertLessEqual(
            record["best_holdout"]["mse"],
            record["initial_holdout"]["mse"],
        )
        scores = evaluate_alignment(
            student, val, basis=torch.eye(8),
        )
        self.assertAlmostEqual(scores["mse"], record["best_holdout"]["mse"], places=5)

    def test_rejects_token_leakage_and_missing_map_validation(self):
        module, train, val = self.make_real_small_module_pairs()
        repeated = replace(val, token_ids=train.token_ids.clone())
        with self.assertRaisesRegex(GLM53AttentionAlignmentError, "leakage"):
            validate_pairing(
                train, repeated,
                coordinate_map=map_data(), student=module,
            )
        unvalidated = replace(map_data(), diagnostics=None)
        with self.assertRaisesRegex(GLM53AttentionAlignmentError, "held-out"):
            validate_pairing(
                train, val, coordinate_map=unvalidated, student=module,
            )

    def test_rejects_bad_dsa_indices_and_tensor_shape(self):
        _, train, _ = self.make_real_small_module_pairs()
        with self.assertRaisesRegex(GLM53AttentionAlignmentError, "out of sequence"):
            replace(train, dsa_indices=torch.full((2, 5, 2), 10, dtype=torch.int32))
        with self.assertRaisesRegex(GLM53AttentionAlignmentError, "token axes"):
            replace(train, glm_attention_output=torch.zeros(2, 4, 8))

    def test_capture_file_hash_and_exact_token_pairing(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            teacher = folder / "teacher"
            teacher.mkdir()
            iq = folder / "iq.safetensors"
            token_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
            ref = teacher / "reference.safetensors"
            save_file({
                "source_attention_output": torch.randn(1, 3, 8),
                "dsa_indices": torch.zeros(1, 3, 2, dtype=torch.int32),
                "token_ids": token_ids,
            }, str(ref))
            save_file({
                "iq_attention_input": torch.randn(1, 3, 8),
                "token_ids": token_ids,
            }, str(iq))
            meta = {
                "format": "iq_source_native_glm53_attention_reference_v1",
                "layer": 6,
                "donor_operator": "transformers==5.15.0 GlmMoeDsaAttention",
                "output_sha256": checksum(ref),
                "donor_config_sha256": "abc",
                "index_source_block": 6,
                "indexer_kind": "full",
            }
            (teacher / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
            sample, evidence = load_paired(
                teacher_dir=teacher, iq_capture=iq, layer=6,
            )
            self.assertEqual(sample.token_ids.shape, (1, 3))
            self.assertEqual(evidence["index_source_block"], 6)
            save_file({
                "iq_attention_input": torch.randn(1, 3, 8),
                "token_ids": torch.tensor([[99, 2, 3]], dtype=torch.int64),
            }, str(iq))
            with self.assertRaisesRegex(ValueError, "identical token"):
                load_paired(teacher_dir=teacher, iq_capture=iq, layer=6)
            meta["output_sha256"] = "tampered"
            (teacher / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_paired(teacher_dir=teacher, iq_capture=iq, layer=6)


if __name__ == "__main__":
    unittest.main()
