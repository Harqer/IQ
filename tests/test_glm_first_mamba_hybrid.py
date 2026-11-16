"""CPU-only executable tests for the first GLM-native/full-MIMO hybrid path.

Small synthetic reference models have the same decoder return and donor
mixer+gated-MLP interface as the official upstream models. These tests verify
integration, baseline identity and fail-closed behavior, not 753B-weight parity.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from iq_model.glm_first_mamba import (
    GLMFirstIntegrationError,
    GLMFirstMambaContract,
    GLMFirstMambaHybrid,
    evenly_spaced_insertions,
    validate_complete_mamba_block,
    wrap_preloaded_official_models,
)


CONTRACT = GLMFirstMambaContract(
    glm_vocab=17, glm_hidden=8, glm_layers=6,
    mamba_hidden=4, mamba_layers=3, mamba_mlp_hidden=8,
    mimo_rank=4, state_size=2,
)


class ToyGLMDecoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x, *, prev_topk_indices=None, **_kwargs):
        # GLM decoder returns both hidden state and DSA routing indices.
        topk = torch.full(x.shape[:2], 2, dtype=torch.long)
        return x + self.proj(x), topk


class ToyGLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=CONTRACT.glm_vocab,
            hidden_size=CONTRACT.glm_hidden,
            num_hidden_layers=CONTRACT.glm_layers,
            tie_word_embeddings=False,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(CONTRACT.glm_vocab, CONTRACT.glm_hidden)
        self.model.layers = nn.ModuleList(
            [ToyGLMDecoder(CONTRACT.glm_hidden) for _ in range(CONTRACT.glm_layers)]
        )
        self.lm_head = nn.Linear(CONTRACT.glm_hidden, CONTRACT.glm_vocab, bias=False)

    def forward(self, input_ids, *, use_cache=False, attention_mask=None, **_kwargs):
        hidden = self.model.embed_tokens(input_ids)
        topk = None
        for layer in self.model.layers:
            hidden, topk = layer(hidden, prev_topk_indices=topk)
        return SimpleNamespace(logits=self.lm_head(hidden), router_ids=topk)

    def generate(self, input_ids, **_kwargs):
        # Identity-path generation is delegated, not separately implemented.
        return torch.cat((input_ids, input_ids[:, -1:]), dim=1)


class ToyMimoMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_mimo = True
        self.mimo_rank = CONTRACT.mimo_rank
        self.d_state = CONTRACT.state_size
        self.proj = nn.Linear(CONTRACT.mamba_hidden, CONTRACT.mamba_hidden)

    def forward(self, x, inference_params=None):
        return self.proj(x)


class ToyGatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CONTRACT.mamba_hidden, CONTRACT.mamba_mlp_hidden * 2)
        self.fc2 = nn.Linear(CONTRACT.mamba_mlp_hidden, CONTRACT.mamba_hidden)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class ToyMambaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(CONTRACT.mamba_hidden)
        self.mixer = ToyMimoMixer()
        self.norm2 = nn.LayerNorm(CONTRACT.mamba_hidden)
        self.mlp = ToyGatedMLP()

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = hidden_states if residual is None else hidden_states + residual
        mixed = self.mixer(self.norm(residual), inference_params=inference_params)
        residual = mixed + residual
        result = self.mlp(self.norm2(residual))
        return result, residual


class ToyMambaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList(
            [ToyMambaBlock() for _ in range(CONTRACT.mamba_layers)]
        )


class GLMFirstHybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.glm = ToyGLM().eval()
        self.donor = ToyMambaModel().eval()
        self.hybrid = wrap_preloaded_official_models(
            self.glm, self.donor, contract=CONTRACT, placements=(0, 2, 5)
        ).eval()
        self.ids = torch.tensor([[1, 2, 3], [2, 5, 8]])

    def test_baseline_keeps_lexical_tensors_and_dsa_output_unchanged(self):
        native = self.glm(self.ids)
        donor_emb = self.glm.model.embed_tokens.weight.detach().clone()
        donor_head = self.glm.lm_head.weight.detach().clone()
        result = self.hybrid(self.ids, use_cache=False)
        torch.testing.assert_close(result.logits, native.logits, rtol=0, atol=0)
        self.assertTrue(torch.equal(result.router_ids, native.router_ids))
        self.assertTrue(torch.equal(self.glm.model.embed_tokens.weight, donor_emb))
        self.assertTrue(torch.equal(self.glm.lm_head.weight, donor_head))
        self.assertFalse(any(p.requires_grad for p in self.glm.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.hybrid.branches.parameters()))
        self.assertEqual(
            sum(len(block._forward_hooks) for block in self.glm.model.layers), 0
        )

    def test_full_donor_block_structure_has_two_normalizations_and_gated_mlp(self):
        for donor in self.donor.backbone.layers:
            validate_complete_mamba_block(donor, CONTRACT)
        self.assertEqual(len(self.hybrid.branches), CONTRACT.mamba_layers)
        self.assertIs(self.hybrid.branches[1].donor_block, self.donor.backbone.layers[1])

    def test_zero_gate_with_enabled_branch_is_exact_baseline_and_trainable(self):
        native = self.glm(self.ids).logits.detach()
        self.hybrid.enable_branches()
        logits = self.hybrid(self.ids, use_cache=False).logits
        torch.testing.assert_close(logits, native, rtol=0, atol=0)
        logits.square().mean().backward()
        gradients = [branch.gate.grad for branch in self.hybrid.branches]
        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(all(bool(torch.isfinite(g)) for g in gradients))

    def test_nonzero_gate_changes_only_hybrid_output(self):
        native = self.glm(self.ids).logits.detach().clone()
        self.hybrid.enable_branches()
        with torch.no_grad():
            self.hybrid.branches[0].gate.fill_(0.25)
        hybrid = self.hybrid(self.ids, use_cache=False).logits
        self.assertFalse(torch.equal(native, hybrid))
        torch.testing.assert_close(self.glm(self.ids).logits, native, rtol=0, atol=0)

    def test_active_cached_forward_is_blocked(self):
        self.hybrid.enable_branches()
        with self.assertRaisesRegex(GLMFirstIntegrationError, "use_cache=False"):
            self.hybrid(self.ids, use_cache=True)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "unpadded"):
            self.hybrid(
                self.ids, use_cache=False,
                attention_mask=torch.tensor([[1, 0, 1], [1, 1, 1]])
            )
        with self.assertRaisesRegex(GLMFirstIntegrationError, "contiguous"):
            self.hybrid(
                self.ids, use_cache=False,
                position_ids=torch.tensor([[2, 3, 4], [0, 1, 2]])
            )
        with self.assertRaisesRegex(GLMFirstIntegrationError, "hybrid generation is blocked"):
            self.hybrid.generate(self.ids)

    def test_original_glm_generate_stays_available_before_enabling(self):
        self.assertTrue(torch.equal(self.glm.generate(self.ids), self.hybrid.generate(self.ids)))
        self.assertEqual(
            sum(len(block._forward_hooks) for block in self.glm.model.layers), 0
        )

    def test_mixer_only_block_rejected(self):
        donor = ToyMambaBlock()
        del donor.mlp
        with self.assertRaisesRegex(GLMFirstIntegrationError, "mlp.fc1"):
            validate_complete_mamba_block(donor, CONTRACT)

    def test_bad_lexical_shape_and_donor_count_fail_closed(self):
        fake = ToyGLM()
        fake.lm_head = nn.Linear(CONTRACT.glm_hidden, CONTRACT.glm_vocab + 1)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "LM head"):
            GLMFirstMambaHybrid(fake, list(self.donor.backbone.layers), contract=CONTRACT)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "ALL complete"):
            GLMFirstMambaHybrid(self.glm, list(self.donor.backbone.layers)[:2], contract=CONTRACT)

    def test_schedule_validated_and_hooks_cleaned_up_on_failure(self):
        self.assertEqual(evenly_spaced_insertions(6, 3), (0, 2, 5))
        with self.assertRaisesRegex(GLMFirstIntegrationError, "placement"):
            GLMFirstMambaHybrid(self.glm, list(self.donor.backbone.layers),
                                contract=CONTRACT, placements=(0, 0, 4))
        self.hybrid.enable_branches()
        self.hybrid.branches[0].donor_block.forward = lambda *_args, **_kw: (
            torch.full((2, 3, 4), float("nan")), torch.zeros(2, 3, 4)
        )
        with self.assertRaisesRegex(GLMFirstIntegrationError, "nonfinite"):
            self.hybrid(self.ids, use_cache=False)
        self.assertEqual(
            sum(len(block._forward_hooks) for block in self.glm.model.layers), 0
        )


if __name__ == "__main__":
    unittest.main()
