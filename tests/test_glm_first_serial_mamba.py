"""CPU-only functional checks for the serial *complete* Mamba donor sidecar.

Uses deterministic tiny synthetic GLM and Mamba modules, not downloaded
pretrained weights; verifies donor interblock residual behavior and gate
isolation, not language-model capability preservation.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from iq_model.glm_first_mamba import GLMFirstIntegrationError
from iq_model.glm_first_serial_mamba import (
    GLMFirstSerialMambaHybrid, SerialMamba3DonorPath, validate_serial_donor,
    wrap_preloaded_official_serial_models,
)
from test_glm_first_mamba_hybrid import CONTRACT, ToyGLM, ToyMambaBlock


class ToyCompleteDonor(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            n_layer=CONTRACT.mamba_layers,
            d_model=CONTRACT.mamba_hidden,
            fused_add_norm=False,
            residual_in_fp32=False,
        )
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList(
            [ToyMambaBlock() for _ in range(CONTRACT.mamba_layers)]
        )
        self.backbone.norm_f = nn.LayerNorm(CONTRACT.mamba_hidden)


class SerialDonorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(709)
        self.glm = ToyGLM().eval()
        self.donor = ToyCompleteDonor().eval()
        self.hybrid = GLMFirstSerialMambaHybrid(
            self.glm, self.donor, contract=CONTRACT
        ).eval()
        self.ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    def test_complete_donor_uses_original_mixer_mlp_and_terminal_norm(self):
        assert self.hybrid.path.backbone is self.donor.backbone
        assert len(self.hybrid.path.backbone.layers) == CONTRACT.mamba_layers
        assert self.hybrid.path.backbone.norm_f is self.donor.backbone.norm_f
        for block in self.hybrid.path.backbone.layers:
            self.assertFalse(any(param.requires_grad for param in block.parameters()))

    def test_original_glm_logits_are_identical_when_gate_disabled(self):
        baseline = self.glm(self.ids)
        original_emb = self.glm.model.embed_tokens.weight.detach().clone()
        original_head = self.glm.lm_head.weight.detach().clone()
        result = self.hybrid(self.ids, use_cache=False)
        torch.testing.assert_close(result.logits, baseline.logits, rtol=0, atol=0)
        self.assertTrue(torch.equal(result.router_ids, baseline.router_ids))
        torch.testing.assert_close(self.glm.model.embed_tokens.weight, original_emb, rtol=0, atol=0)
        torch.testing.assert_close(self.glm.lm_head.weight, original_head, rtol=0, atol=0)
        self.assertEqual(sum(len(x._forward_hooks) for x in self.glm.model.layers), 0)

    def test_serial_donor_math_matches_manual_upstream_residual_chain(self):
        path = self.hybrid.path
        projected = torch.randn(2, 3, CONTRACT.mamba_hidden)
        hidden, residual = projected, None
        for block in self.donor.backbone.layers:
            hidden, residual = block(hidden, residual=residual, inference_params=None)
        expected = self.donor.backbone.norm_f(hidden + residual)
        actual = path._serial_donor_forward(projected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_serial_donor_is_distinct_from_parallel_independent_layer_calls(self):
        path = self.hybrid.path
        projected = torch.randn(2, 3, CONTRACT.mamba_hidden)
        serial = path._serial_donor_forward(projected)
        # Each independent branch sees the SAME GLM-projected input rather
        # than its predecessor's Mamba hidden state.
        parallel_results = []
        for block in self.donor.backbone.layers:
            out, residue = block(projected, residual=None)
            parallel_results.append(out + residue)
        parallel = self.donor.backbone.norm_f(
            torch.stack(parallel_results).mean(dim=0)
        )
        self.assertFalse(torch.equal(serial, parallel))

    def test_enabled_zero_gate_trains_scalar_and_preserves_logits(self):
        self.hybrid.enable_branch(True)
        baseline = self.glm(self.ids).logits.detach()
        output = self.hybrid(self.ids, use_cache=False).logits
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        output.square().mean().backward()
        self.assertIsNotNone(self.hybrid.path.gate.grad)
        self.assertTrue(torch.isfinite(self.hybrid.path.gate.grad).item())

    def test_nonzero_gate_changes_output_but_not_glm_weights(self):
        expected = self.glm(self.ids).logits.detach().clone()
        self.hybrid.enable_branch(True)
        with torch.no_grad():
            self.hybrid.path.gate.fill_(0.4)
        result = self.hybrid(self.ids, use_cache=False).logits
        self.assertFalse(torch.equal(result, expected))
        torch.testing.assert_close(self.glm(self.ids).logits, expected, rtol=0, atol=0)

    def test_cached_or_opaque_positional_forward_is_rejected(self):
        self.hybrid.enable_branch(True)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "use_cache=False"):
            self.hybrid(self.ids, use_cache=True)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "positional"):
            self.hybrid(self.ids, torch.ones_like(self.ids), use_cache=False)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "input_ids"):
            self.hybrid(inputs_embeds=torch.randn(2, 3, CONTRACT.glm_hidden), use_cache=False)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "generation"):
            self.hybrid.generate(self.ids)

    def test_missing_mlp_or_fused_mode_fails_closed(self):
        donor = ToyCompleteDonor()
        donor.backbone.layers[1].mlp = nn.Identity()
        with self.assertRaisesRegex(GLMFirstIntegrationError, "fc1"):
            validate_serial_donor(donor, CONTRACT)
        donor = ToyCompleteDonor()
        donor.config.fused_add_norm = True
        with self.assertRaisesRegex(GLMFirstIntegrationError, "fused"):
            validate_serial_donor(donor, CONTRACT)

    def test_hook_is_removed_after_failure(self):
        self.hybrid.enable_branch(True)
        self.donor.backbone.layers[0].forward = lambda *args, **kwargs: (
            torch.full((2, 3, CONTRACT.mamba_hidden), float("nan")),
            torch.zeros(2, 3, CONTRACT.mamba_hidden),
        )
        with self.assertRaisesRegex(GLMFirstIntegrationError, "NaN/Inf"):
            self.hybrid(self.ids, use_cache=False)
        self.assertEqual(sum(len(x._forward_hooks) for x in self.glm.model.layers), 0)

    def test_disabled_serial_donor_avoids_shared_decoder_hooks(self):
        # When disabled the original GLM forward is called directly, even
        # if a gate contains a trained nonzero value.
        with torch.no_grad():
            self.hybrid.path.gate.fill_(0.6)
        expected = self.glm(self.ids).logits
        output = self.hybrid(self.ids, use_cache=False).logits
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(
            sum(len(layer._forward_hooks) for layer in self.glm.model.layers), 0
        )

    def test_active_serial_rejects_invalid_glm_ids_and_context(self):
        self.hybrid.enable_branch(True)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "GLM token IDs"):
            self.hybrid(
                torch.tensor([[CONTRACT.glm_vocab]], dtype=torch.long),
                use_cache=False,
            )
        with self.assertRaisesRegex(GLMFirstIntegrationError, "context"):
            self.hybrid(
                torch.ones((1, 2049), dtype=torch.long),
                use_cache=False,
            )

    def test_production_factory_rejects_synthetic_configuration(self):
        with self.assertRaisesRegex(GLMFirstIntegrationError, "official full-scale"):
            wrap_preloaded_official_serial_models(
                self.glm, self.donor,
                contract=CONTRACT,
                tokenizer=object(),
                expected_glm_tokenizer_sha256="a" * 64,
            )

    def test_eval_mode_and_frozen_pretrained_weights(self):
        self.hybrid.train()
        self.assertFalse(self.hybrid.glm_model.training)
        self.assertFalse(self.hybrid.path.backbone.training)
        self.assertTrue(self.hybrid.path.in_adapter.training)
        self.assertFalse(any(p.requires_grad for p in self.hybrid.glm_model.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.hybrid.path.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
