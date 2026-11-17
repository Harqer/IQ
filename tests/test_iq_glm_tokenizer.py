"""CPU-only GLM-5.3 tokenizer compatibility tests for IQ's actual hybrid schedule."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from torch import nn

from iq_model.architecture import HybridSchedule
from iq_model.config import IQModelConfig
from iq_model.glm_tokenizer import (
    GLM53_EOS_TOKEN_IDS,
    GLM53_VOCAB_SIZE,
    GLMTokenizerError,
    encode_glm_prompt,
    tokenizer_mapping_sha256,
    verify_glm_tokenizer_for_iq,
    with_glm_tokenizer_config,
)
from iq_model.hybrid import IQHybridConfig
from iq_model.mlp import RoutedMoEConfig
from iq_model.state import Mamba3MIMOConfig


VOCAB = {
    "Hello": 1,
    "world": 2,
    "<|endoftext|>": 154820,
    "<|assistant|>": 154827,
    "<eop>": 154829,
}


class SyntheticGLMTokenizer:
    pad_token_id = 154820
    eos_token_id = 154820

    def __init__(self, vocab=None):
        self._vocab = dict(VOCAB if vocab is None else vocab)
        self._reverse = {identifier: token for token, identifier in self._vocab.items()}

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, identifier):
        return self._reverse.get(identifier)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token)

    def encode(self, text, *, add_special_tokens):
        return [1, 2, 154829] if text else []


def official_config():
    return {
        "model_type": "glm_moe_dsa",
        "vocab_size": GLM53_VOCAB_SIZE,
        "pad_token_id": 154820,
        "eos_token_id": list(GLM53_EOS_TOKEN_IDS),
    }


def tiny_iq_config():
    return IQHybridConfig(
        model=IQModelConfig(vocab_size=128256, hidden_size=8,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, intermediate_size=16),
        schedule=HybridSchedule.from_tokens(["mamba3", "moe", "mamba3"]),
        mamba3=Mamba3MIMOConfig(d_model=8, num_layers=2, d_state=4,
                                headdim=4, mimo_rank=4),
        moe=RoutedMoEConfig(hidden_size=8, expert_intermediate_size=16,
                            num_experts=2, top_k=1),
    )


class GLMIQTokenizerContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SyntheticGLMTokenizer()
        self.config = with_glm_tokenizer_config(tiny_iq_config())

    def test_config_change_is_lexical_only(self):
        old = tiny_iq_config()
        new = with_glm_tokenizer_config(old)
        self.assertEqual(new.model.vocab_size, 154880)
        self.assertEqual(old.model.vocab_size, 128256)
        self.assertEqual(new.schedule, old.schedule)
        self.assertEqual(new.mamba3, old.mamba3)
        self.assertEqual(new.moe, old.moe)
        self.assertEqual(new.model.hidden_size, old.model.hidden_size)
        self.assertEqual(new.model.num_hidden_layers, old.model.num_hidden_layers)
        self.assertEqual(new.model.tie_word_embeddings, old.model.tie_word_embeddings)

    def test_roundtrip_config_does_not_change_architecture(self):
        original = tiny_iq_config()
        updated = with_glm_tokenizer_config(original)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "iq-glm.json"
            updated.write_json(str(path))
            restored = IQHybridConfig.from_json(str(path))
        self.assertEqual(restored.fingerprint, updated.fingerprint)
        self.assertEqual(restored.schedule, original.schedule)

    def test_valid_glm_tokenizer_with_iq_vocab(self):
        result = verify_glm_tokenizer_for_iq(
            self.tokenizer, official_config(), self.config.model,
            expected_mapping_sha256=tokenizer_mapping_sha256(VOCAB))
        self.assertEqual(result["vocab_size"], 154880)
        self.assertEqual(result["eos_token_ids"], list(GLM53_EOS_TOKEN_IDS))
        self.assertFalse(result["weight_alignment_verified"])
        self.assertEqual(
            encode_glm_prompt(self.tokenizer, "Hello world", self.config.model),
            [1, 2, 154829],
        )

    def test_original_mamba_vocab_is_not_compatible(self):
        with self.assertRaisesRegex(GLMTokenizerError, "IQ vocab_size"):
            verify_glm_tokenizer_for_iq(
                self.tokenizer, official_config(), tiny_iq_config().model)

    def test_same_size_swapped_token_mapping_fails_pinned_hash(self):
        modified = dict(VOCAB)
        modified["Hello"], modified["world"] = modified["world"], modified["Hello"]
        with self.assertRaisesRegex(GLMTokenizerError, "pinned hash"):
            verify_glm_tokenizer_for_iq(
                SyntheticGLMTokenizer(modified), official_config(), self.config.model,
                expected_mapping_sha256=tokenizer_mapping_sha256(VOCAB))

    def test_wrong_special_tokens_and_source_config(self):
        tokenizer = SyntheticGLMTokenizer()
        tokenizer.eos_token_id = 17
        with self.assertRaisesRegex(GLMTokenizerError, "EOS/PAD"):
            verify_glm_tokenizer_for_iq(tokenizer, official_config(), self.config.model)
        corrupted = official_config()
        corrupted["eos_token_id"] = [154820]
        with self.assertRaisesRegex(GLMTokenizerError, "source config"):
            verify_glm_tokenizer_for_iq(self.tokenizer, corrupted, self.config.model)

    def test_duplicate_out_of_range_and_missing_control_ids(self):
        for bad_vocab, message in (
            ({**VOCAB, "duplicated": 1}, "duplicate"),
            ({**VOCAB, "outside": GLM53_VOCAB_SIZE}, "exceeds"),
            ({"Hello": 1, "<|endoftext|>": 154820}, "missing GLM"),
        ):
            with self.subTest(reason=message):
                with self.assertRaisesRegex(GLMTokenizerError, message):
                    verify_glm_tokenizer_for_iq(
                        SyntheticGLMTokenizer(bad_vocab), official_config(),
                        self.config.model,
                    )

    def test_optional_actual_iq_lexical_tensor_shape_contract(self):
        class GeometryOnlyIQ(nn.Module):
            def __init__(self, vocab):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab, 8, device="meta")
                self.lm_head = nn.Linear(8, vocab, bias=False, device="meta")

        verify_glm_tokenizer_for_iq(
            self.tokenizer, official_config(), self.config.model,
            model=GeometryOnlyIQ(154880),
        )
        with self.assertRaisesRegex(GLMTokenizerError, "LM head"):
            model = GeometryOnlyIQ(154880)
            model.lm_head = nn.Linear(8, 128256, bias=False, device="meta")
            verify_glm_tokenizer_for_iq(
                self.tokenizer, official_config(), self.config.model, model=model)

    def test_encode_rejects_empty_out_of_range_and_oversize(self):
        with self.assertRaisesRegex(GLMTokenizerError, "nonempty"):
            encode_glm_prompt(self.tokenizer, "", self.config.model)
        context = IQModelConfig(vocab_size=154880, max_position_embeddings=2)
        with self.assertRaisesRegex(GLMTokenizerError, "context"):
            encode_glm_prompt(self.tokenizer, "hello", context)
        self.tokenizer.encode = lambda text, add_special_tokens: [154880]
        with self.assertRaisesRegex(GLMTokenizerError, "outside"):
            encode_glm_prompt(self.tokenizer, "hello", self.config.model)

    def test_cli_usage_needs_no_model_and_no_inference(self):
        output = subprocess.run(
            [sys.executable, "-m", "scripts.check_iq_glm_tokenizer", "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--write-config", output.stdout)
        self.assertIn("--iq-config", output.stdout)


if __name__ == "__main__":
    unittest.main()
