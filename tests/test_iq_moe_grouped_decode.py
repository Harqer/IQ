"""Real top-k SiTU reference parity for grouped single-token expert decoding."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from iq_model.mlp.latent_moe import StableLatentMoE, StableLatentMoEConfig


class GroupedExpertDecodeTests(unittest.TestCase):
    def _make_model(self, dtype):
        torch.manual_seed(112)
        cfg = StableLatentMoEConfig(
            hidden_size=16, latent_size=8, expert_intermediate_size=16,
            num_experts=32, top_k=8, num_shared_experts=2,
        )
        return StableLatentMoE(cfg).to(dtype=dtype)

    def test_top8_single_token_grouped_matches_original_reference(self):
        for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, 3e-2)):
            with self.subTest(dtype=str(dtype)):
                moe = self._make_model(dtype)
                hidden = torch.randn(1, 16).to(dtype)
                moe.eval()
                with torch.inference_mode(), patch("torch.bmm", wraps=torch.bmm) as bmm:
                    batched = moe(hidden)
                    self.assertGreaterEqual(bmm.call_count, 3)
                moe.train()  # exact existing per-expert training/reference path
                with torch.inference_mode():
                    reference = moe(hidden)
                torch.testing.assert_close(
                    batched.hidden_states, reference.hidden_states,
                    atol=tolerance, rtol=tolerance,
                )
                torch.testing.assert_close(
                    batched.selected_weights, reference.selected_weights,
                    atol=tolerance, rtol=tolerance,
                )
                self.assertTrue(torch.equal(
                    batched.selected_experts, reference.selected_experts,
                ))

    def test_multi_token_prefill_keeps_unmodified_reference_path(self):
        model = self._make_model(torch.float32).eval()
        with torch.no_grad(), patch("torch.bmm", wraps=torch.bmm) as bmm:
            result = model(torch.randn(1, 4, 16))
            self.assertEqual(bmm.call_count, 0)
        self.assertEqual(result.hidden_states.shape, (1, 4, 16))

    def test_training_gradients_are_preserved(self):
        model = self._make_model(torch.float32).train()
        x = torch.randn(1, 16, requires_grad=True)
        output = model(x)
        output.hidden_states.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(model.router.weight.grad).all()))


if __name__ == "__main__":
    unittest.main()
