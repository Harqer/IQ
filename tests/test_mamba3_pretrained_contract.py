"""CPU contract tests; actual pretrained CUDA inference is a separate required gate."""
from __future__ import annotations

from types import SimpleNamespace
import json
import subprocess
import sys
import unittest

import torch
from torch import nn

from iq_model.mamba3_pretrained import (
    FoundationInferenceError,
    IQMamba3Pretrained,
    MAX_VALIDATED_CONTEXT,
    validate_ids,
    validate_native_contract,
)


class Mamba3(nn.Module):
    def __init__(self, *, mimo=True, rank=4):
        super().__init__()
        self.is_mimo = mimo
        self.mimo_rank = rank


class GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = Mamba3()
        self.mlp = GatedMLP()


class FakeTokenizer:
    bos_token_id = 128000
    eos_token_id = 128001

    def __len__(self):
        return 128256


class GeometryOnlyDonor(nn.Module):
    """Correct source geometry without allocating a gigabyte of test weights."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            d_model=2048, d_intermediate=3824, n_layer=24,
            vocab_size=128256, tie_embeddings=True, rms_norm=True,
            residual_in_fp32=True, fused_add_norm=False, attn_layer_idx=[],
            ssm_cfg={"layer": "Mamba3", "is_mimo": True, "mimo_rank": 4,
                     "d_state": 128, "headdim": 64, "chunk_size": 16},
        )
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([Block() for _ in range(24)])
        self.backbone.embedding = nn.Module()
        embedding = nn.Parameter(torch.empty((128256, 2048), device="meta"))
        self.backbone.embedding.register_parameter("weight", embedding)
        self.backbone.norm_f = nn.Module()
        self.backbone.norm_f.register_parameter(
            "weight", nn.Parameter(torch.empty((2048,), device="meta"))
        )
        self.lm_head = nn.Module()
        self.lm_head.register_parameter("weight", embedding)


class NativeFoundationContractTests(unittest.TestCase):
    def setUp(self):
        self.donor = GeometryOnlyDonor()
        self.tokenizer = FakeTokenizer()

    def test_contract_accepts_complete_24_layer_donor(self):
        validate_native_contract(self.donor, self.tokenizer)
        wrapper = IQMamba3Pretrained(self.donor, self.tokenizer)
        self.assertIs(wrapper.core, self.donor)
        self.assertEqual(len(wrapper.core.backbone.layers), 24)

    def test_rejects_mimo_to_siso_changes(self):
        self.donor.backbone.layers[7].mixer.is_mimo = False
        with self.assertRaisesRegex(FoundationInferenceError, "MIMO"):
            validate_native_contract(self.donor, self.tokenizer)

    def test_rejects_untied_source_head(self):
        self.donor.lm_head.weight = nn.Parameter(torch.empty((128256, 2048), device="meta"))
        with self.assertRaisesRegex(FoundationInferenceError, "tying"):
            validate_native_contract(self.donor, self.tokenizer)

    def test_rejects_bad_mimo_rank(self):
        self.donor.backbone.layers[10].mixer.mimo_rank = 1
        with self.assertRaisesRegex(FoundationInferenceError, "MIMO"):
            validate_native_contract(self.donor, self.tokenizer)

    def test_rejects_missing_block(self):
        self.donor.backbone.layers = nn.ModuleList([Block() for _ in range(23)])
        with self.assertRaisesRegex(FoundationInferenceError, "24-block"):
            validate_native_contract(self.donor, self.tokenizer)

    def test_rejects_geometry_mutation(self):
        self.donor.config.d_model = 4096
        with self.assertRaisesRegex(FoundationInferenceError, "configuration mismatch"):
            validate_native_contract(self.donor, self.tokenizer)

    def test_rejects_incompatible_tokenizer(self):
        self.tokenizer.eos_token_id = 128256
        with self.assertRaisesRegex(FoundationInferenceError, "special token"):
            validate_native_contract(self.donor, self.tokenizer)

    def test_token_ids_and_context_bounds(self):
        validate_ids(torch.tensor([[128000, 2, 128001]]), 2)
        for ids, max_new in (
            (torch.tensor([[128256]]), 1),
            (torch.tensor([[-1]]), 1),
            (torch.tensor([[1, 2], [1, 2]]), 1),
            (torch.tensor([[1.0, 2.0]]), 1),
            (torch.tensor([[1]]), 0),
            (torch.ones(1, MAX_VALIDATED_CONTEXT, dtype=torch.long), 1),
        ):
            with self.subTest(ids_shape=tuple(ids.shape), max_new=max_new):
                with self.assertRaises(FoundationInferenceError):
                    validate_ids(ids, max_new)

    def test_cli_preflight_does_not_claim_inference(self):
        result = subprocess.run([sys.executable, "-m", "scripts.iq_mamba3_native_inference",
                                 "--preflight"], capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        self.assertIs(payload["native_pretrained_inference_executed"], False)
        self.assertIn("cuda_available", payload["hardware"])

    def test_bf16_device_guard_runs_before_hf_download(self):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.skipTest("Only the no-native-CUDA guard is exercised here")
        with self.assertRaisesRegex(FoundationInferenceError, "BF16"):
            IQMamba3Pretrained.load_official()


if __name__ == "__main__":
    unittest.main()
