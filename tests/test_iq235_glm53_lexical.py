"""CPU-only, no-network GLM-5.3 lexical and full235 regression checks."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import unittest

from iq_cloud.glm53_lexical import (
    GLM53LexicalError, GLM_EOS_IDS, GLM_VOCAB, mapping_sha256,
    inspect_lexical_boundary, require_trained_glm53_lexical_weights,
    verify_glm53_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    pad_token_id = 154820
    eos_token_id = 154820

    def __init__(self):
        self.special = {i: f"<|glm_{i}|>" for i in GLM_EOS_IDS}
        self.vocab = {
            self.special.get(i, f"tok_{i}"): i for i in range(GLM_VOCAB)
        }

    def __len__(self):
        return GLM_VOCAB

    def get_vocab(self):
        return self.vocab

    def convert_ids_to_tokens(self, number):
        return self.special.get(number, f"tok_{number}")

    def convert_tokens_to_ids(self, token):
        for key, value in self.special.items():
            if token == value:
                return key
        if token.startswith("tok_"):
            return int(token[4:])
        return None


def official_config():
    return dict(
        model_type="glm_moe_dsa",
        vocab_size=154880,
        hidden_size=6144,
        pad_token_id=154820,
        eos_token_id=list(GLM_EOS_IDS),
        tie_word_embeddings=False,
    )


class Full235GLM53LexicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / "configs"
        cls.source = json.loads((base / "iq235_checkpoint_source.json").read_text())
        cls.target = json.loads((base / "iq235_glm53_tokenizer_phase_a.json").read_text())
        cls.tokenizer = FakeTokenizer()

    def test_existing_full235_source_and_profile_are_unchanged(self):
        outcome = inspect_lexical_boundary(self.source, self.target)
        self.assertEqual(outcome["physical_layers"], 235)
        self.assertEqual(outcome["target_embedding_shape"], [154880, 4096])
        self.assertFalse(outcome["full_pretrained_inference_ready"])

    def test_unrelated_architecture_edits_are_rejected(self):
        bad = deepcopy(self.target)
        bad["schedule"]["layers"][0] = "moe"
        with self.assertRaises(GLM53LexicalError):
            inspect_lexical_boundary(self.source, bad)

    def test_wrong_source_and_target_vocab_fail(self):
        for side in ("source", "target"):
            src, tgt = deepcopy(self.source), deepcopy(self.target)
            (src if side == "source" else tgt)["model"]["vocab_size"] = 1
            with self.subTest(side=side), self.assertRaises(GLM53LexicalError):
                inspect_lexical_boundary(src, tgt)

    def test_lexical_overlay_is_required_before_pretrained_generation(self):
        with self.assertRaisesRegex(GLM53LexicalError, "overlay"):
            require_trained_glm53_lexical_weights(self.source, self.target)

    def test_official_glm_tokenizer_ids_and_mapping(self):
        pinned = mapping_sha256(self.tokenizer.get_vocab())
        result = verify_glm53_tokenizer(
            self.tokenizer, official_config(), expected_mapping_sha256=pinned
        )
        self.assertTrue(result["mapping_independently_pinned"])
        self.assertEqual(result["eos_ids"], list(GLM_EOS_IDS))
        self.assertFalse(result["neural_weight_alignment_verified"])

    def test_corrupt_mapping_and_wrong_pin_fail(self):
        corrupted = dict(self.tokenizer.get_vocab())
        corrupted["tok_0"] = GLM_VOCAB
        with self.assertRaises(GLM53LexicalError):
            mapping_sha256(corrupted)
        with self.assertRaisesRegex(GLM53LexicalError, "digest"):
            verify_glm53_tokenizer(
                self.tokenizer, official_config(), expected_mapping_sha256="0" * 64
            )

    def test_special_id_and_official_config_guard(self):
        self.tokenizer.pad_token_id = 3
        try:
            with self.assertRaisesRegex(GLM53LexicalError, "padding"):
                verify_glm53_tokenizer(self.tokenizer, official_config())
        finally:
            self.tokenizer.pad_token_id = 154820
        incorrect = official_config()
        incorrect["eos_token_id"] = [154820]
        with self.assertRaisesRegex(GLM53LexicalError, "source config"):
            verify_glm53_tokenizer(self.tokenizer, incorrect)


if __name__ == "__main__":
    unittest.main()
