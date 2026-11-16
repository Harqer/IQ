"""CPU-only tokenizer-ID integrity fixtures (no source model downloads)."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from torch import nn

from iq_transfer.glm_first_tokenizer import (
    GLMTokenizerContractError,
    mapping_fingerprint,
    verify_glm_tokenizer,
)


IDS = (154820, 154827, 154829)
VOCAB = {
    "hello": 1, "<|endoftext|>": IDS[0],
    "<|assistant|>": IDS[1], "<eop>": IDS[2],
}


class TinyTokenizer:
    def __init__(self, vocab=None):
        self._vocab = dict(vocab or VOCAB)
        self.eos_token_id = IDS[0]
        self.pad_token_id = IDS[0]
        self._inverse = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, identifier):
        return self._inverse.get(identifier)

    def convert_tokens_to_ids(self, value):
        return self._vocab.get(value)


class TinyGLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=154880, hidden_size=4, eos_token_id=list(IDS),
            pad_token_id=IDS[0]
        )
        self.emb = nn.Embedding(154880, 4)
        self.head = nn.Linear(4, 154880, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head


class TokenizerIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyGLM()
        self.tokenizer = TinyTokenizer()
        self.reference_hash = mapping_fingerprint(VOCAB)

    def test_verified_mapping_and_special_tokens(self):
        result = verify_glm_tokenizer(
            self.tokenizer, self.model,
            expected_mapping_sha256=self.reference_hash,
        )
        self.assertEqual(result["status"], "verified_mapping_only")
        self.assertEqual(result["eos_token_ids"], list(IDS))

    def test_same_tokenizer_size_with_swapped_ids_fails(self):
        bad = TinyTokenizer({
            "hello": 1, "<|endoftext|>": IDS[0],
            "<|assistant|>": IDS[2], "<eop>": IDS[1],
        })
        with self.assertRaisesRegex(GLMTokenizerContractError, "fingerprint"):
            verify_glm_tokenizer(
                bad, self.model, expected_mapping_sha256=self.reference_hash
            )

    def test_glm_lexical_tensor_shape_mismatch_fails(self):
        self.model.head = nn.Linear(4, 154879, bias=False)
        with self.assertRaisesRegex(GLMTokenizerContractError, "lm_head"):
            verify_glm_tokenizer(
                self.tokenizer, self.model,
                expected_mapping_sha256=self.reference_hash,
            )

    def test_no_pinned_mapping_hash_fails(self):
        with self.assertRaisesRegex(GLMTokenizerContractError, "SHA-256"):
            verify_glm_tokenizer(self.tokenizer, self.model,
                                 expected_mapping_sha256="")

    def test_eos_mismatch_fails(self):
        self.model.config.eos_token_id = [154820]
        with self.assertRaisesRegex(GLMTokenizerContractError, "EOS"):
            verify_glm_tokenizer(self.tokenizer, self.model,
                                 expected_mapping_sha256=self.reference_hash)

    def test_out_of_range_token_id_fails(self):
        bad = TinyTokenizer({
            **VOCAB, "out_of_range": 154880
        })
        with self.assertRaisesRegex(GLMTokenizerContractError, "outside"):
            verify_glm_tokenizer(bad, self.model,
                                 expected_mapping_sha256=self.reference_hash)


if __name__ == "__main__":
    unittest.main()
