from __future__ import annotations

import unittest

import torch

from iq_model import (
    ReasoningEnergyCritic,
    ReasoningEnergyCriticConfig,
    ReasoningRecurrence,
    ReasoningRecurrenceConfig,
    ReasoningStateInjector,
)


class ReasoningRecurrenceTests(unittest.TestCase):
    def config(
        self,
        *,
        max_steps: int = 4,
        min_steps: int = 1,
        halt_threshold: float = 0.9,
        state_delta_epsilon: float = 1e-3,
    ) -> ReasoningRecurrenceConfig:
        return ReasoningRecurrenceConfig(
            hidden_size=16,
            state_dim=8,
            transition_hidden_dim=24,
            max_steps=max_steps,
            min_steps=min_steps,
            halt_threshold=halt_threshold,
            state_delta_epsilon=state_delta_epsilon,
            energy_delta_epsilon=1e-4,
            dropout=0.0,
        )

    def critic(self) -> ReasoningEnergyCritic:
        return ReasoningEnergyCritic(
            ReasoningEnergyCriticConfig(
                state_dim=8,
                context_dim=16,
                hidden_dim=12,
                dropout=0.0,
            )
        )

    def test_training_uses_differentiable_soft_halting_weights(self):
        torch.manual_seed(101)
        recurrence = ReasoningRecurrence(self.config()).train()
        hidden = torch.randn(2, 6, 16, requires_grad=True)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )

        output = recurrence(hidden, attention_mask=mask)

        self.assertEqual(tuple(output.state.shape), (2, 8))
        self.assertEqual(tuple(output.state_trace.shape), (2, 4, 8))
        self.assertEqual(tuple(output.halt_probabilities.shape), (2, 4))
        self.assertEqual(tuple(output.halt_weights.shape), (2, 4))
        self.assertEqual(tuple(output.relative_state_deltas.shape), (2, 4))
        self.assertTrue(
            torch.allclose(
                output.halt_weights.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertGreaterEqual(float(output.expected_steps), 1.0)
        self.assertLessEqual(float(output.expected_steps), 4.0)
        self.assertEqual(output.steps_executed, 4)
        self.assertIsNone(output.energy_trace)

        loss = output.state.square().mean() + 0.01 * output.expected_steps
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertIsNotNone(recurrence.halting.proj.weight.grad)
        self.assertTrue(
            torch.isfinite(recurrence.halting.proj.weight.grad).all()
        )

    def test_ebm_does_not_change_generated_state_trajectory(self):
        torch.manual_seed(102)
        recurrence = ReasoningRecurrence(self.config()).train()
        critic = self.critic().train()
        hidden = torch.randn(3, 5, 16)

        without_critic = recurrence(hidden)
        with_critic = recurrence(
            hidden,
            energy_critic=critic,
        )

        self.assertTrue(
            torch.allclose(
                without_critic.state_trace,
                with_critic.state_trace,
                atol=0.0,
                rtol=0.0,
            )
        )
        self.assertIsNotNone(with_critic.energy_trace)
        assert with_critic.energy_trace is not None
        self.assertEqual(tuple(with_critic.energy_trace.shape), (3, 4))
        self.assertEqual(tuple(with_critic.energy_deltas.shape), (3, 4))

    def test_halting_gradients_do_not_train_energy_critic(self):
        torch.manual_seed(108)
        recurrence = ReasoningRecurrence(self.config()).train()
        critic = self.critic().train()
        hidden = torch.randn(2, 5, 16, requires_grad=True)

        output = recurrence(
            hidden,
            energy_critic=critic,
        )
        loss = output.state.square().mean() + 0.1 * output.expected_steps
        loss.backward()

        self.assertIsNotNone(recurrence.halting.proj.weight.grad)
        self.assertIsNotNone(recurrence.transition.update[0].weight.grad)
        critic_grads = [
            parameter.grad
            for parameter in critic.parameters()
        ]
        self.assertTrue(all(grad is None for grad in critic_grads))

    def test_inference_hard_exit_respects_minimum_steps(self):
        torch.manual_seed(103)
        recurrence = ReasoningRecurrence(
            self.config(
                max_steps=5,
                min_steps=2,
                halt_threshold=0.9,
                state_delta_epsilon=1e9,
            )
        ).eval()
        with torch.no_grad():
            recurrence.halting.proj.weight.zero_()
            recurrence.halting.proj.bias.fill_(10.0)

        output = recurrence(torch.randn(2, 5, 16))

        self.assertEqual(output.steps_executed, 2)
        self.assertEqual(tuple(output.state_trace.shape), (2, 2, 8))
        self.assertEqual(tuple(output.halt_weights.shape), (2, 2))
        self.assertTrue(
            torch.equal(
                output.halt_weights[:, -1],
                torch.ones(2),
            )
        )

    def test_energy_stability_needs_two_observations(self):
        torch.manual_seed(104)
        recurrence = ReasoningRecurrence(
            self.config(
                max_steps=5,
                min_steps=1,
                halt_threshold=0.9,
                state_delta_epsilon=1e9,
            )
        ).eval()
        critic = self.critic().eval()
        with torch.no_grad():
            recurrence.halting.proj.weight.zero_()
            recurrence.halting.proj.bias.fill_(10.0)
            critic.energy_head.weight.zero_()

        output = recurrence(
            torch.randn(2, 5, 16),
            energy_critic=critic,
            require_energy_stability=True,
        )

        self.assertEqual(output.steps_executed, 2)
        self.assertIsNotNone(output.energy_trace)
        self.assertIsNotNone(output.energy_deltas)
        assert output.energy_deltas is not None
        self.assertTrue(
            torch.equal(
                output.energy_deltas[:, 0],
                torch.zeros(2),
            )
        )

    def test_multi_document_packed_rows_fail_closed(self):
        recurrence = ReasoningRecurrence(self.config())
        hidden = torch.randn(1, 5, 16)
        document_ids = torch.tensor([[0, 0, 1, 1, 1]])

        with self.assertRaisesRegex(
            ValueError,
            "one document per batch row",
        ):
            recurrence(hidden, document_ids=document_ids)

    def test_reasoning_context_mask_blocks_future_token_leakage(self):
        torch.manual_seed(105)
        recurrence = ReasoningRecurrence(self.config()).train()
        hidden = torch.randn(2, 6, 16)
        changed = hidden.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])
        context_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        original = recurrence(
            hidden,
            reasoning_context_mask=context_mask,
        )
        modified = recurrence(
            changed,
            reasoning_context_mask=context_mask,
        )

        self.assertTrue(
            torch.allclose(
                original.state_trace,
                modified.state_trace,
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_reasoning_state_injector_masks_preboundary_tokens(self):
        torch.manual_seed(106)
        injector = ReasoningStateInjector(
            hidden_size=16,
            state_dim=8,
            gate_init=-2.0,
        )
        hidden = torch.randn(2, 5, 16)
        state = torch.randn(2, 8)
        token_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )

        output = injector(
            hidden,
            state,
            token_mask=token_mask,
        )

        self.assertTrue(
            torch.equal(
                output[~token_mask],
                hidden[~token_mask],
            )
        )
        self.assertFalse(
            torch.equal(
                output[token_mask],
                hidden[token_mask],
            )
        )

    def test_reasoning_state_injector_starts_near_identity(self):
        torch.manual_seed(107)
        injector = ReasoningStateInjector(
            hidden_size=16,
            state_dim=8,
            gate_init=-8.0,
        )
        hidden = torch.randn(2, 4, 16)
        state = torch.randn(2, 8)

        output = injector(hidden, state)

        self.assertEqual(output.shape, hidden.shape)
        relative_change = (
            (output - hidden).float().norm()
            / hidden.float().norm().clamp_min(1e-6)
        )
        self.assertLess(float(relative_change), 0.01)


if __name__ == "__main__":
    unittest.main()
