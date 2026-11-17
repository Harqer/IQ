"""CPU-only IQ checkpoint and tokenizer provenance gates.

These use fake shard bytes: preflight verifies metadata/layout, not
safetensors tensor content or pretrained model quality.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iq_transfer.checkpoint_preflight import (
    CheckpointPreflightError,
    preflight_iq_checkpoint,
    verify_iq_tokenizer,
)


class FakeTokenizer:
    def __init__(self, size, special_tokens):
        self.size = size
        self.mapping = {
            f"token_{i}": i for i in range(size)
        }
        for token, ident in special_tokens.items():
            self.mapping.pop(f"token_{ident}", None)
            self.mapping[token] = ident

    def __len__(self):
        return self.size

    def get_vocab(self):
        return self.mapping


MAMBA_SPECIALS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009,
}


class CheckpointPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {
            "model": {"vocab_size": 128256, "hidden_size": 4096},
            "schedule": {"layers": ["mamba3", "moe"]},
        }
        self.manifest = {
            "schema_version": 2,
            "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
            "foundation": {"repo": "state-spaces/mamba3-mimo-1.5b"},
        }
        self.index = {
            "metadata": {"total_size": 7},
            "weight_map": {"layers.0.mamba.core.D": "model-000.safetensors"},
        }
        (self.root / "model-000.safetensors").write_bytes(b"1234567")
        (self.root / "tokenizer.json").write_text("{}")
        self.write()

    def write(self):
        (self.root / "iq_config.json").write_text(json.dumps(self.config))
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(self.index)
        )
        (self.root / "transplant_manifest.json").write_text(
            json.dumps(self.manifest)
        )

    def test_valid_checkpoint_has_no_gpu_or_tensor_dependency(self):
        proof = preflight_iq_checkpoint(
            self.root, tokenizer_dir=self.root
        )
        self.assertEqual(proof.shard_count, 1)
        self.assertEqual(proof.shard_bytes, 7)
        self.assertEqual(proof.weight_count, 1)
        self.assertEqual(proof.manifest["method"], self.manifest["method"])

    def test_reject_parent_traversal_and_absolute_shards(self):
        for name in ("../secrets.safetensors", "/etc/secret.safetensors",
                     "./model-000.safetensors", "nested/model-000.safetensors",
                     "..\\secret.safetensors"):
            with self.subTest(name=name):
                self.index["weight_map"]["layers.0.mamba.core.D"] = name
                self.write()
                with self.assertRaisesRegex(CheckpointPreflightError, "unsafe"):
                    preflight_iq_checkpoint(self.root)

    def test_reject_symlink_to_external_shard(self):
        (self.root / "other.safetensors").symlink_to(
            self.root / "model-000.safetensors"
        )
        self.index["weight_map"]["layers.0.mamba.core.D"] = "other.safetensors"
        self.write()
        with self.assertRaisesRegex(CheckpointPreflightError, "symlink"):
            preflight_iq_checkpoint(self.root)

    def test_reject_corrupt_or_partial_index(self):
        self.index["metadata"]["total_size"] = 10
        self.write()
        with self.assertRaisesRegex(CheckpointPreflightError, "total_size"):
            preflight_iq_checkpoint(self.root)
        self.index["metadata"]["total_size"] = 7
        self.index["weight_map"]["layers.0.mamba.core.D"] = ["bad"]
        self.write()
        with self.assertRaisesRegex(CheckpointPreflightError, "invalid shard"):
            preflight_iq_checkpoint(self.root)

    def test_reject_missing_required_metadata(self):
        (self.root / "iq_config.json").unlink()
        with self.assertRaisesRegex(CheckpointPreflightError, "invalid JSON"):
            preflight_iq_checkpoint(self.root)

    def test_mamba_tokenizer_matches_own_native_lexical_source(self):
        tokenizer = FakeTokenizer(128256, MAMBA_SPECIALS)
        origin = verify_iq_tokenizer(tokenizer, self.config, self.manifest)
        self.assertIn("mamba3", origin)

    def test_reject_same_size_but_wrong_mamba_token_ids(self):
        bad = dict(MAMBA_SPECIALS)
        bad["<|eot_id|>"] = 128008
        tokenizer = FakeTokenizer(128256, bad)
        with self.assertRaisesRegex(CheckpointPreflightError, "special ID"):
            verify_iq_tokenizer(tokenizer, self.config, self.manifest)

    def test_reject_missing_or_unknown_lexical_foundation(self):
        tokenizer = FakeTokenizer(128256, MAMBA_SPECIALS)
        with self.assertRaisesRegex(CheckpointPreflightError, "missing transplant"):
            verify_iq_tokenizer(tokenizer, self.config, None)
        unknown = {"method": "arbitrary_untrained_shell"}
        with self.assertRaisesRegex(CheckpointPreflightError, "unsupported lexical"):
            verify_iq_tokenizer(tokenizer, self.config, unknown)

    def test_gpt_oss_foundation_does_not_require_llama_special_tokens(self):
        config = {
            "model": {"vocab_size": 201088, "hidden_size": 4096}
        }
        manifest = {
            "method": "direct_cross_architecture_weight_transplant",
            "donors": {"language": {"repo": "openai/gpt-oss-20b"}},
        }
        tokenizer = FakeTokenizer(
            201088, {"<|im_start|>": 200000, "<|im_end|>": 200001}
        )
        self.assertIn("gpt-oss", verify_iq_tokenizer(tokenizer, config, manifest))


if __name__ == "__main__":
    unittest.main()
