from __future__ import annotations

import unittest

import torch

from iq_model import (
    ReasoningEnergyCritic,
    ReasoningEnergyCriticConfig,
    energy_margin_ranking_loss,
)


class ReasoningEnergyCriticTests(unittest.TestCase):
    def config(self) -> ReasoningEnergyCriticConfig:
        return ReasoningEnergyCriticConfig(
            state_dim=16,
            context_dim=12,
            hidden_dim=24,
            dropout=0.0,
        )

    def test_scores_single_states_and_backpropagates(self):
        torch.manual_seed(71)
        critic = ReasoningEnergyCritic(self.config())
        state = torch.randn(4, 16, requires_grad=True)
        context = torch.randn(4, 12, requires_grad=True)

        energy = critic(state, context)

        self.assertEqual(tuple(energy.shape), (4,))
        self.assertTrue(torch.isfinite(energy).all())
        energy.mean().backward()
        self.assertIsNotNone(state.grad)
        self.assertIsNotNone(context.grad)
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_scores_and_selects_candidate_states(self):
        torch.manual_seed(72)
        critic = ReasoningEnergyCritic(self.config()).eval()
        candidates = torch.randn(3, 5, 16)
        context = torch.randn(3, 12)

        with torch.no_grad():
            energies = critic.score_candidates(candidates, context)
            selected = critic.best_candidate_index(candidates, context)

        self.assertEqual(tuple(energies.shape), (3, 5))
        self.assertEqual(tuple(selected.shape), (3,))
        self.assertTrue(torch.equal(selected, energies.argmin(dim=-1)))

    def test_margin_ranking_loss_prefers_lower_positive_energy(self):
        good_positive = torch.tensor([-2.0, -1.5])
        negatives = torch.tensor([1.0, 0.5])
        bad_positive = torch.tensor([2.0, 1.5])

        good_loss = energy_margin_ranking_loss(
            good_positive,
            negatives,
            margin=0.5,
        )
        bad_loss = energy_margin_ranking_loss(
            bad_positive,
            negatives,
            margin=0.5,
        )

        self.assertLess(float(good_loss), float(bad_loss))

    def test_critic_rejects_incompatible_context_shape(self):
        critic = ReasoningEnergyCritic(self.config())
        candidates = torch.randn(2, 3, 16)
        context = torch.randn(4, 12)

        with self.assertRaisesRegex(ValueError, "batch dimensions must match"):
            critic.score_candidates(candidates, context)


if __name__ == "__main__":
    unittest.main()
