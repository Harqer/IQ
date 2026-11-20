"""CPU-only lexical candidate mapping tests: no tokenizers/GPU downloads."""
from __future__ import annotations
from copy import deepcopy
import unittest

from iq_cloud.lexical_warmstart import (
    LexicalCandidateError, candidate_rows, validate_candidate_manifest,
    validate_source_copy_provenance, LLAMA_COPY_REVISION, LLAMA_COPY_MAPPING_SHA256,
)
from iq_cloud.glm53_lexical import GLM_PINNED_MAPPING_SHA256, GLM_VOCAB


class TokenizerStub:
    def __init__(self, mapping, *, special=()):
        self._token_to_id = dict(mapping)
        self._id_to_token = {v:k for k,v in mapping.items()}
        self.all_special_ids = list(special)

    def get_vocab(self):
        return dict(self._token_to_id)

    def decode(self, ids, **kwargs):
        return "".join(self._id_to_token[i] for i in ids)

    def encode(self, text, **kwargs):
        return [self._token_to_id[text]] if text in self._token_to_id else []

    def __len__(self):
        return len(self._token_to_id)


class CandidateTests(unittest.TestCase):
    def test_exact_single_token_roundtrip_is_candidate_not_trained(self):
        source = TokenizerStub({"hello":1,"world":2, "other": 3, "special": 4}, special=[4])
        target = TokenizerStub({"hello":10,"world":11,"bad":12,"special":13}, special=[13])
        self.assertEqual(candidate_rows(source,target,target_limit=20,source_limit=10),
                         {10:1,11:2})

    def test_diagnostics_account_for_every_target_token(self):
        source = TokenizerStub({"hello":1, "world":2})
        target = TokenizerStub({"hello":10, "world":11, "absent":12, "<special>":13},
                               special=[13])
        report = {}
        matches = candidate_rows(source, target, diagnostics=report,
                                 target_limit=20, source_limit=10)
        self.assertEqual(matches, {10:1, 11:2})
        self.assertEqual(report["candidate_matched"], 2)
        self.assertEqual(report["target_special"], 1)
        self.assertEqual(report["source_not_single_defined_token"], 1)
        self.assertEqual(sum(report.values()), 4)
        self.assertEqual(report["input_only_composition_candidate"], 0)

    def test_control_and_replacement_char_filtered(self):
        source = TokenizerStub({"hello":1,"\ufffd":2,"\x00":3})
        target = TokenizerStub({"hello":10,"\ufffd":11,"\x00":12})
        self.assertEqual(candidate_rows(source,target,target_limit=20,source_limit=10),
                         {10:1})

    def test_duplicate_target_selection_rejected(self):
        source=TokenizerStub({"hi":1})
        target=TokenizerStub({"hi":2})
        with self.assertRaises(LexicalCandidateError):
            candidate_rows(source,target,target_ids=[2,2],target_limit=10)

    def test_multi_token_input_only_composition(self):
        class SourceWithPieces(TokenizerStub):
            def encode(self, text, **kwargs):
                if text == "hello-world":
                    return [1, 2]
                return super().encode(text, **kwargs)
            def decode(self, ids, **kwargs):
                if ids == [1, 2]:
                    return "hello-world"
                return super().decode(ids, **kwargs)
        source = SourceWithPieces({"hello":1, "world":2})
        target = TokenizerStub({"hello-world":10, "hello":11})
        compositions = {}
        diagnostics = {}
        matches = candidate_rows(
            source, target, compositions=compositions, diagnostics=diagnostics,
            source_limit=10, target_limit=20,
        )
        self.assertEqual(matches, {11:1})
        self.assertEqual(compositions, {10:[1,2]})
        self.assertEqual(diagnostics["input_only_composition_candidate"],1)
        self.assertEqual(sum(diagnostics.values()),2)

    def test_source_out_of_range_not_copied(self):
        source=TokenizerStub({"hi":11})
        target=TokenizerStub({"hi":2})
        self.assertEqual(candidate_rows(source,target,target_limit=10,source_limit=10),{})

    def test_source_copy_drift_rejected(self):
        validate_source_copy_provenance(LLAMA_COPY_MAPPING_SHA256, LLAMA_COPY_REVISION)
        with self.assertRaisesRegex(LexicalCandidateError, "drifted"):
            validate_source_copy_provenance("0"*64, LLAMA_COPY_REVISION)
        with self.assertRaisesRegex(LexicalCandidateError, "drifted"):
            validate_source_copy_provenance(LLAMA_COPY_MAPPING_SHA256, "0"*40)

    def test_manifest_rejects_false_ready(self):
        template=dict(schema_version=1,stage="lexical_candidate_mapping_only",
                      source_tokenizer_official_equivalence_proven=False,
                      target_tokenizer_mapping_sha256=GLM_PINNED_MAPPING_SHA256,
                      target_model_vocab_capacity=GLM_VOCAB,
                      full_pretrained_inference_ready=False,
                      weights_initialized_or_modified=False,
                      candidate_matches=[[2,1],[4,3]],candidate_matched_defined_rows=2,
                      unmapped_target_rows_including_reserved=GLM_VOCAB-2,
                      target_defined_token_count=154856)
        self.assertEqual(validate_candidate_manifest(template)["candidate_matched_rows"],2)
        for key,value in [("full_pretrained_inference_ready",True),
                          ("source_tokenizer_official_equivalence_proven",True),
                          ("candidate_matched_defined_rows",3),
                          ("target_model_vocab_capacity",128256)]:
            wrong=deepcopy(template)
            wrong[key]=value
            with self.subTest(key=key),self.assertRaises(LexicalCandidateError):
                validate_candidate_manifest(wrong)

    def test_manifest_unsorted_and_invalid_source_ids(self):
        template=dict(schema_version=1,stage="lexical_candidate_mapping_only",
                      source_tokenizer_official_equivalence_proven=False,
                      target_tokenizer_mapping_sha256=GLM_PINNED_MAPPING_SHA256,
                      target_model_vocab_capacity=GLM_VOCAB,
                      full_pretrained_inference_ready=False,
                      weights_initialized_or_modified=False,
                      candidate_matches=[[2,1],[3,2]],candidate_matched_defined_rows=2,
                      unmapped_target_rows_including_reserved=GLM_VOCAB-2,
                      target_defined_token_count=154856)
        for rows in ([[3,1],[2,2]],[[2,1],[3,128256]],[[2,1],[2,2]]):
            wrong=deepcopy(template)
            wrong["candidate_matches"]=rows
            with self.subTest(rows=rows),self.assertRaises(LexicalCandidateError):
                validate_candidate_manifest(wrong)


if __name__=="__main__":
    unittest.main()
