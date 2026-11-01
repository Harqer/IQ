"""Reference-parity tests of the FULL incremental hybrid inference orchestration.

CPU toy model uses a deterministic recurrent mixer in place of CUDA Mamba-3,
while exercising the actual IQ CSA/HCA, dense RoPE KV, AttnRes and MoE schedule.
The native H200 MIMO kernel needs a separate GPU equivalence test; this file
does not claim to validate H200 CUDA hardware.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from iq_model.architecture import HybridLayerType
from iq_model.attention import CompressedContextConfig, DenseContextAttention
from iq_model.attention import (
    CompressedSparseContextAttention, HeavilyCompressedContextAttention,
)
from iq_model.config import IQModelConfig
from iq_model.hybrid_inference import IQHybridInferenceEngine
from iq_model.residual import BlockAttentionResidual, BlockAttnResConfig


class DeterministicMIMOCore(nn.Module):
    mimo_rank = 4

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def step(self, x, angle, ssm, k, v):
        # Toy state-space recurrence, not an emulated Mamba-3 CUDA kernel.
        ssm.add_(x * 0.25)
        return (x + ssm, angle, ssm, k, v)


class ReferenceMambaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Identity()
        self.mamba = nn.Module()
        self.mamba.core = DeterministicMIMOCore(0)
        self.residual_dropout = nn.Identity()

    def forward(self, x, *, attention_mask, document_ids,
                inference_params=None):
        normed = self.norm(x)
        mixed = normed + torch.cumsum(normed, dim=1) * 0.25
        if inference_params is not None:
            final = normed.sum(dim=1) * 0.25
            zero = torch.zeros_like(final)
            inference_params.key_value_memory_dict[0] = (
                zero.clone(), final, zero.clone(), zero.clone(),
            )
        return x + mixed


class ReferenceMoELayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.routed = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return SimpleNamespace(hidden_states=x + self.routed(x))


class ReferenceCompressedLayer(nn.Module):
    def __init__(self, att):
        super().__init__()
        self.norm = nn.Identity()
        self.attention = att
        self.residual_dropout = nn.Identity()

    def forward(self, x, *, position_ids, attention_mask, document_ids):
        return x + self.attention(self.norm(x))


class ReferenceDenseLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.Identity()
        self.attention = DenseContextAttention(cfg)
        self.residual_dropout = nn.Identity()

    def forward(self, x, *, position_ids, attention_mask, document_ids):
        return x + self.attention(
            self.norm(x),
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )


class ReferenceReasoning(nn.Module):
    def forward(self, x, **kwargs):
        return SimpleNamespace(state=x.mean(dim=1))


class ReferenceInjector(nn.Module):
    def forward(self, x, state, *, token_mask=None):
        return x + 0.1 * state.unsqueeze(1)


class ReferenceHybridModel(nn.Module):
    def __init__(self, *, reasoning=True):
        super().__init__()
        self.model_config = IQModelConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=5,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        self.config = SimpleNamespace(
            mamba3=SimpleNamespace(mimo_rank=4),
            schedule=SimpleNamespace(layers=(
                HybridLayerType.MAMBA3,
                HybridLayerType.MOE,
                HybridLayerType.CSA,
                HybridLayerType.HCA,
                HybridLayerType.DENSE_ATTENTION,
            )),
            require_verifier_stability=False,
        )
        compressed = CompressedContextConfig(
            hidden_size=16, num_attention_heads=4, head_dim=8,
            q_lora_rank=8, partial_rotary_dim=4,
            max_position_embeddings=64, sliding_window=3,
            csa_compress_rate=2, hca_compress_rate=4,
            o_groups=2, o_lora_rank=4, index_n_heads=2,
            index_head_dim=4, index_topk=8,
            direct_token_indexer=True,
            compress_rope_theta=10000.,
        )
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([
            ReferenceMambaLayer(),
            ReferenceMoELayer(16),
            ReferenceCompressedLayer(CompressedSparseContextAttention(compressed)),
            ReferenceCompressedLayer(HeavilyCompressedContextAttention(compressed)),
            ReferenceDenseLayer(self.model_config),
        ])
        self.attnres = BlockAttentionResidual(BlockAttnResConfig(
            hidden_size=16, num_layers=5, block_size=2,
        ))
        self.norm = nn.Identity()
        self.reasoning = ReferenceReasoning() if reasoning else None
        self.reasoning_verifier = None
        self.reasoning_injector = ReferenceInjector() if reasoning else None
        self.multimodal = None
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, input_ids, *, mamba_inference_params=None,
                return_hidden_states=False):
        x = self.embed_tokens(input_ids)
        attnres = self.attnres.init_state(x)
        for kind, layer in zip(self.config.schedule.layers, self.layers):
            inp = self.attnres.read(attnres)
            if kind is HybridLayerType.MAMBA3:
                out = layer(inp, attention_mask=None, document_ids=None,
                            inference_params=mamba_inference_params)
            elif kind is HybridLayerType.MOE:
                out = layer(inp).hidden_states
            else:
                out = layer(inp, position_ids=None, attention_mask=None,
                            document_ids=None)
            attnres = self.attnres.advance(attnres, out - inp)
            x = out
        hidden = self.norm(self.attnres.finalize(attnres))
        if self.reasoning is not None:
            result = self.reasoning(hidden)
            hidden = self.reasoning_injector(hidden, result.state)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=hidden)


class StatefulHybridTests(unittest.TestCase):
    def _compare_prefix_and_decode(self, reasoning: bool):
        torch.manual_seed(2026)
        model = ReferenceHybridModel(reasoning=reasoning).eval()
        engine = IQHybridInferenceEngine(model, max_context_tokens=32)
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.inference_mode():
            expected = model(prefix).logits[:, -1:, :]
            actual = engine.prefill(prefix)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            for n in [5, 6, 7]:
                token = torch.tensor([[n]], dtype=torch.long)
                prefix = torch.cat((prefix, token), dim=1)
                expected = model(prefix).logits[:, -1:, :]
                actual = engine.decode_token(token)
                torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
        self.assertEqual(engine.state.length, 7)
        self.assertEqual(engine.state.decode_count, 3)
        self.assertEqual(engine.state.mamba.seqlen_offset, 7)

    def test_full_hybrid_recurrent_logits_parity_with_reasoning(self):
        self._compare_prefix_and_decode(reasoning=True)

    def test_full_hybrid_recurrent_logits_parity_without_reasoning(self):
        self._compare_prefix_and_decode(reasoning=False)

    def test_prefix_reuse_and_breaking_retokenization_reset(self):
        torch.manual_seed(2027)
        model = ReferenceHybridModel().eval()
        engine = IQHybridInferenceEngine(model, max_context_tokens=32)
        prefix = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            engine.prefill(prefix)
            longer = torch.tensor([[1, 2, 3, 4, 5]])
            received = engine.sync_prompt(longer)
            expected = model(longer).logits[:, -1:, :]
            torch.testing.assert_close(received, expected, atol=3e-4, rtol=3e-4)
            self.assertEqual(engine.state.decode_count, 2)
            changed = torch.tensor([[1, 9, 3, 4, 5]])
            received = engine.sync_prompt(changed)
            expected = model(changed).logits[:, -1:, :]
            torch.testing.assert_close(received, expected, atol=3e-4, rtol=3e-4)
            self.assertEqual(engine.state.prefill_count, 1)
            self.assertEqual(engine.state.decode_count, 0)

    def test_context_limits_and_no_inference_without_prefill(self):
        model = ReferenceHybridModel().eval()
        engine = IQHybridInferenceEngine(model, max_context_tokens=4)
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            engine.decode_token(torch.tensor([[2]]))
        with self.assertRaises(ValueError):
            engine.prefill(torch.tensor([[1, 2, 3, 4, 5]]))
        with torch.inference_mode():
            engine.prefill(torch.tensor([[1, 2, 3, 4]]))
        with self.assertRaisesRegex(ValueError, "context budget"):
            engine.decode_token(torch.tensor([[1]]))


if __name__ == "__main__":
    unittest.main()
