from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from iq_model import IQForCausalLM, IQModelConfig
from iq_transfer import (
    CaptureRunnerError,
    build_phi_layer_calibration_from_bundles,
    capture_iq_activations,
    capture_phi_activations,
)


class FakePhiAttention(nn.Module):
    def __init__(self, hidden=8, q_heads=4, kv_heads=2, head_dim=2):
        super().__init__()
        self.q_width = q_heads * head_dim
        self.kv_width = kv_heads * head_dim
        self.qkv_proj = nn.Linear(hidden, self.q_width + 2 * self.kv_width, bias=False)
        self.o_proj = nn.Linear(self.q_width, hidden, bias=False)

    def forward(self, x, **kwargs):
        qkv = self.qkv_proj(x)
        return self.o_proj(qkv[..., : self.q_width]), None


class FakePhiMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class FakePhiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = FakePhiAttention()
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = FakePhiMLP()

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x))[0]
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class FakePhiBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([FakePhiLayer(), FakePhiLayer()])
        self.norm = nn.LayerNorm(8)


class FakePhi(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakePhiBase()
        self.config = types.SimpleNamespace(
            model_type="phi3",
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.norm(x)


class CaptureRunnerTests(unittest.TestCase):
    def test_phi_capture_filters_padding_and_derives_fused_spaces(self):
        torch.manual_seed(1)
        model = FakePhi()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
        bundle = capture_phi_activations(model, [batch])
        self.assertEqual(bundle.sample_count, 5)
        self.assertEqual(bundle.require("layer.0.q").shape, (5, 8))
        self.assertEqual(bundle.require("layer.0.k").shape, (5, 4))
        self.assertEqual(bundle.require("layer.0.mlp_hidden").shape, (5, 12))

    def test_iq_capture_and_calibration_pair(self):
        torch.manual_seed(2)
        config = IQModelConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=12,
            max_position_embeddings=16,
        )
        iq = IQForCausalLM(config)
        phi = FakePhi()
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

        source_fit = capture_phi_activations(phi, [batch])
        target_fit = capture_iq_activations(iq, [batch])
        source_validation = capture_phi_activations(phi, [batch])
        target_validation = capture_iq_activations(iq, [batch])

        calibration = build_phi_layer_calibration_from_bundles(
            source_fit,
            target_fit,
            source_validation,
            target_validation,
            source_layer=0,
            target_layer=0,
        )
        self.assertEqual(calibration.source_k_fit.shape, (4, 4))
        self.assertEqual(calibration.attn_in.source_fit.shape, (4, 8))
        self.assertEqual(calibration.attn_in.target_fit.shape, (4, 8))

        with self.assertRaises(CaptureRunnerError):
            capture_iq_activations(
                iq,
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 0]]),
                        "attention_mask": torch.tensor([[1, 1, 0]]),
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
