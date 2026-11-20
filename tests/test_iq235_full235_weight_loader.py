"""Only synthetic tiny checkpoints; production 539 GB is never fetched."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file

from iq_cloud.full235_weight_loader import (
    LoadedSourceBody, VerifiedBodyLoaderError, install_verified_shard,
    load_verified_full235_body, validate_model_inventory,
)
from iq_cloud.remote_shard_loader import verify_safetensors_header


class ToyIQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(3, 2)
        self.lm_head = nn.Linear(2, 3, bias=False)
        self.norm = nn.LayerNorm(2)


class FullLoaderTests(unittest.TestCase):
    def setUp(self):
        self.m = ToyIQ()
        self.index = {"weight_map": {
            "embed_tokens.weight": "model-a.safetensors",
            "lm_head.weight": "model-a.safetensors",
            "norm.weight": "model-a.safetensors",
            "norm.bias": "model-a.safetensors",
        }}

    def test_target_model_inventory_is_exact_and_untied(self):
        state = validate_model_inventory(
            self.m, self.index, expected_lexical_shape=(3, 2),
            require_production=False,
        )
        self.assertEqual(len(state), 4)
        bad = deepcopy(self.index)
        bad["weight_map"].pop("norm.bias")
        with self.assertRaisesRegex(VerifiedBodyLoaderError, "keysets"):
            validate_model_inventory(
                self.m, bad, expected_lexical_shape=(3, 2),
                require_production=False,
            )

    def test_shard_injection_reuses_norm_but_never_original_lexical(self):
        before_embed = self.m.embed_tokens.weight.detach().clone()
        before_head = self.m.lm_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.safetensors"
            save_file({
                "embed_tokens.weight": torch.ones(3, 2),
                "lm_head.weight": torch.ones(3, 2),
                "norm.weight": torch.tensor([4., 5.]),
                "norm.bias": torch.tensor([6., 7.]),
            }, path)
            with path.open("rb") as f:
                header = verify_safetensors_header(
                    f, file_size=path.stat().st_size,
                    expected_keys=set(self.index["weight_map"]),
                )
            loaded = install_verified_shard(
                path, header, set(header), self.m.state_dict(),
                original_lexical_shape=(3, 2), require_original_bf16=False,
            )
        self.assertEqual(loaded, {"norm.weight", "norm.bias"})
        torch.testing.assert_close(self.m.norm.weight, torch.tensor([4., 5.]))
        torch.testing.assert_close(self.m.norm.bias, torch.tensor([6., 7.]))
        self.assertTrue(torch.equal(before_embed, self.m.embed_tokens.weight))
        self.assertTrue(torch.equal(before_head, self.m.lm_head.weight))

    def test_mismatched_shape_fails_before_copy(self):
        orig = self.m.norm.bias.detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.safetensors"
            save_file({
                "embed_tokens.weight": torch.zeros(3, 2),
                "lm_head.weight": torch.zeros(3, 2),
                "norm.weight": torch.zeros(3),
                "norm.bias": torch.full((2,), 9.0),
            }, path)
            with path.open("rb") as f:
                header = verify_safetensors_header(
                    f, file_size=path.stat().st_size,
                )
            with self.assertRaisesRegex(VerifiedBodyLoaderError, "shape or dtype"):
                install_verified_shard(
                    path, header, set(header), self.m.state_dict(),
                    original_lexical_shape=(3, 2), require_original_bf16=False,
                )
        self.assertTrue(torch.equal(orig, self.m.norm.bias))

    def test_missing_external_manifest_is_rejected_before_network_or_gpu(self):
        with self.assertRaisesRegex(VerifiedBodyLoaderError, "manifest; production loading blocked"):
            load_verified_full235_body(
                self.m, approved_manifest_bytes=b"{}",
                approved_manifest_sha256="0"*64,
            )

    def test_self_attested_complete_digest_manifest_does_not_bypass_pin(self):
        import json
        from hashlib import sha256
        manifest = {
            "schema_version": 1,
            "source_index_sha256": "a"*64,
            "sha256_algorithm": "sha256",
            "model_source_status": "compiled_not_inference_validated",
            "shards_sha256": {"model-a.safetensors": "a"*64},
        }
        contents = json.dumps(manifest, sort_keys=True).encode()
        with self.assertRaisesRegex(VerifiedBodyLoaderError, "production loading blocked"):
            load_verified_full235_body(
                self.m, approved_manifest_bytes=contents,
                approved_manifest_sha256=sha256(contents).hexdigest(),
            )

    def test_return_type_cannot_claim_inference_ready(self):
        record = LoadedSourceBody(
            model=self.m, loaded_nonlexical_tensor_count=2,
            verified_source_shard_count=1,
            source_index_sha256="a"*64, digest_manifest_sha256="b"*64,
        )
        self.assertFalse(record.inference_ready)
        self.assertFalse(record.lexical_rows_initialized)


if __name__ == "__main__":
    unittest.main()
