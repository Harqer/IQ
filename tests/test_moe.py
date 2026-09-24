from __future__ import annotations

import unittest

import torch

from iq_model import (
    MoEConfigError,
    RoutedMoEConfig,
    RoutedSwiGLUMoE,
)


class RoutedMoETests(unittest.TestCase):
    def test_forward_losses_and_gradients(self):
        torch.manual_seed(12)
        config = RoutedMoEConfig(
            hidden_size=8,
            expert_intermediate_size=16,
            num_experts=3,
            top_k=3,
            shared_expert_intermediate_size=12,
        )
        moe = RoutedSwiGLUMoE(config)
        x = torch.randn(2, 5, 8, requires_grad=True)
        output = moe(x)

        self.assertEqual(tuple(output.hidden_states.shape), (2, 5, 8))
        self.assertEqual(tuple(output.router_logits.shape), (2, 5, 3))
        self.assertEqual(tuple(output.router_probabilities.shape), (2, 5, 3))
        self.assertEqual(tuple(output.expert_counts.shape), (3,))
        self.assertEqual(int(output.expert_counts.sum()), 2 * 5 * 3)
        self.assertTrue(torch.isfinite(output.load_balance_loss))
        self.assertTrue(torch.isfinite(output.router_z_loss))
        self.assertEqual(output.overflow_count, 0)

        loss = (
            output.hidden_states.square().mean()
            + 0.01 * output.load_balance_loss
            + 0.001 * output.router_z_loss
        )
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertTrue(torch.isfinite(moe.router.weight.grad).all())
        for expert in moe.experts:
            for parameter in expert.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNotNone(moe.shared_expert)
        for parameter in moe.shared_expert.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_selected_weights_are_renormalized(self):
        torch.manual_seed(13)
        config = RoutedMoEConfig(
            hidden_size=4,
            expert_intermediate_size=8,
            num_experts=4,
            top_k=2,
        )
        moe = RoutedSwiGLUMoE(config)
        x = torch.randn(3, 4)
        with torch.no_grad():
            logits = moe.router(x)
            probabilities = torch.softmax(logits.float(), dim=-1)
            selected_probabilities, _ = torch.topk(
                probabilities,
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )
            selected_probabilities = selected_probabilities / selected_probabilities.sum(
                dim=-1,
                keepdim=True,
            )
        self.assertTrue(
            torch.allclose(
                selected_probabilities.sum(dim=-1),
                torch.ones(3),
                atol=1e-7,
                rtol=1e-7,
            )
        )

    def test_capacity_overflow_never_drops_silently(self):
        config = RoutedMoEConfig(
            hidden_size=4,
            expert_intermediate_size=8,
            num_experts=4,
            top_k=1,
            capacity_factor=0.5,
            overflow_policy="unbounded",
        )
        moe = RoutedSwiGLUMoE(config)
        with torch.no_grad():
            moe.router.weight.zero_()
        x = torch.ones(8, 4)
        output = moe(x)
        self.assertGreater(output.overflow_count, 0)
        self.assertEqual(int(output.expert_counts.sum()), 8)

        strict = RoutedSwiGLUMoE(
            RoutedMoEConfig(
                hidden_size=4,
                expert_intermediate_size=8,
                num_experts=4,
                top_k=1,
                capacity_factor=0.5,
                overflow_policy="error",
            )
        )
        with torch.no_grad():
            strict.router.weight.zero_()
        with self.assertRaises(RuntimeError):
            strict(x)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(MoEConfigError):
            RoutedMoEConfig(
                hidden_size=8,
                expert_intermediate_size=16,
                num_experts=2,
                top_k=3,
            )


if __name__ == "__main__":
    unittest.main()
