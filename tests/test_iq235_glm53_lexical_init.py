"""CPU-only sparse GLM lexical initializer tests; synthetic 2D rows only."""
from __future__ import annotations

from copy import deepcopy
import unittest
import torch

from iq_training.glm53_lexical_init import iter_sparse_lexical_candidates
from iq_cloud.lexical_warmstart import (
    LLAMA_COPY_MAPPING_SHA256, LLAMA_COPY_REVISION,
    LexicalCandidateError,
)
from iq_cloud.glm53_lexical import GLM_PINNED_MAPPING_SHA256, GLM_VOCAB


def toy_manifest():
    return {
        "schema_version": 1,
        "stage": "lexical_candidate_mapping_only",
        "source_tokenizer_mapping_sha256": LLAMA_COPY_MAPPING_SHA256,
        "source_tokenizer_resolved_commit": LLAMA_COPY_REVISION,
        "source_tokenizer_official_equivalence_proven": False,
        "target_tokenizer_mapping_sha256": GLM_PINNED_MAPPING_SHA256,
        "target_model_vocab_capacity": GLM_VOCAB,
        "target_defined_token_count": 154856,
        "candidate_matched_defined_rows": 2,
        "unmapped_target_rows_including_reserved": GLM_VOCAB-2,
        "candidate_matches": [[10, 1], [11, 2]],
        "input_only_composition_candidates": [[20, [1, 2]]],
        "input_compositions_are_not_trained_output_weights": True,
        "full_pretrained_inference_ready": False,
        "source_weights_opened": False,
        "weights_initialized_or_modified": False,
        "candidate_rows_not_trained_alignment": True,
        "needs_training_and_held_out_validation": True,
    }


class IQ235LexicalSparseTests(unittest.TestCase):
    def setUp(self):
        self.embedding = torch.tensor([
            [1., 2., 3., 4.],
            [6., 8., 10., 12.],
            [10., 12., 14., 16.],
        ])
        # The existing IQ head has already been widened and scaled; preserve it.
        self.head = self.embedding / 2
        self.manifest = toy_manifest()

    def build(self, manifest=None):
        return list(iter_sparse_lexical_candidates(
            self.embedding, self.head,
            self.manifest if manifest is None else manifest,
            test_only_smaller_geometry=True,
        ))

    def test_single_row_preserves_both_IQ_space_tensors(self):
        saved_e, saved_h = self.embedding.clone(), self.head.clone()
        generated = self.build()
        self.assertEqual(len(generated), 3)
        self.assertTrue(torch.equal(generated[0].iq_input_candidate, self.embedding[1]))
        self.assertTrue(torch.equal(generated[0].iq_output_head_candidate, self.head[1]))
        self.assertEqual(generated[0].original_source_token_ids, (1,))
        self.assertTrue(torch.equal(self.embedding, saved_e))
        self.assertTrue(torch.equal(self.head, saved_h))
        generated[0].iq_input_candidate[0] = 999.0
        self.assertTrue(torch.equal(self.embedding, saved_e))

    def test_input_only_composition_never_copies_an_output_head(self):
        generated = self.build()
        composed = generated[-1]
        self.assertEqual(composed.target_token_id, 20)
        self.assertEqual(composed.method, "input_only_multitoken_mean_candidate")
        self.assertIsNone(composed.iq_output_head_candidate)
        self.assertTrue(torch.equal(
            composed.iq_input_candidate, (self.embedding[1] + self.embedding[2]) / 2
        ))

    def test_production_requires_original_IQ_geometry(self):
        with self.assertRaisesRegex(LexicalCandidateError, "128256x4096"):
            list(iter_sparse_lexical_candidates(
                self.embedding, self.head, self.manifest
            ))

    def test_wrong_mapping_digest_fails(self):
        invalid = deepcopy(self.manifest)
        invalid["source_tokenizer_mapping_sha256"] = "0" * 64
        with self.assertRaisesRegex(LexicalCandidateError, "provenance"):
            self.build(invalid)

    def test_wrong_source_shape_and_dtype_fail(self):
        with self.assertRaisesRegex(LexicalCandidateError, "invalid"):
            list(iter_sparse_lexical_candidates(
                self.embedding, self.head[:, :2], self.manifest,
                test_only_smaller_geometry=True,
            ))
        with self.assertRaisesRegex(LexicalCandidateError, "dtype"):
            list(iter_sparse_lexical_candidates(
                self.embedding, self.head.double(), self.manifest,
                test_only_smaller_geometry=True,
            ))

    def test_nonfinite_source_fails_closed(self):
        bad = self.embedding.clone()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(LexicalCandidateError, "nonfinite"):
            list(iter_sparse_lexical_candidates(
                bad, self.head, self.manifest,
                test_only_smaller_geometry=True,
            ))

    def test_wrong_source_row_fails(self):
        invalid = deepcopy(self.manifest)
        invalid["candidate_matches"] = [[10, 1], [11, 100]]
        with self.assertRaisesRegex(LexicalCandidateError, "row outside"):
            self.build(invalid)

    def test_full_checkpoint_must_remain_unverified(self):
        bad = deepcopy(self.manifest)
        bad["full_pretrained_inference_ready"] = True
        with self.assertRaises(LexicalCandidateError):
            self.build(bad)


if __name__ == "__main__":
    unittest.main()
