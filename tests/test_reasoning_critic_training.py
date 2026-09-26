from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from iq_model import (
    ReasoningEnergyCritic,
    ReasoningEnergyCriticConfig,
)
from iq_training import (
    OptimizerConfig,
    ReasoningCriticPairingConfig,
    ReasoningCriticTrainingError,
    ReasoningTrajectoryBatch,
    build_optimizer,
    build_same_task_energy_pairs,
    reasoning_critic_pair_loss,
    train_reasoning_critic_step,
)


class ReasoningCriticTrainingTests(unittest.TestCase):
    def critic(self) -> ReasoningEnergyCritic:
        return ReasoningEnergyCritic(
            ReasoningEnergyCriticConfig(
                state_dim=8,
                context_dim=6,
                hidden_dim=12,
                dropout=0.0,
            )
        )

    def trajectory_batch(self) -> ReasoningTrajectoryBatch:
        torch.manual_seed(201)
        context_a = torch.randn(6)
        context_b = torch.randn(6)
        context = torch.stack(
            (
                context_a,
                context_a.clone(),
                context_b,
                context_b.clone(),
            )
        )
        states = torch.randn(4, 3, 8)
        success = torch.tensor(
            [True, False, True, False],
            dtype=torch.bool,
        )
        return ReasoningTrajectoryBatch(
            task_ids=("task-a", "task-a", "task-b", "task-b"),
            context=context,
            state_trace=states,
            verified_success=success,
        )

    def test_model_output_capture_detaches_trajectory_graph(self):
        torch.manual_seed(202)
        context = torch.randn(2, 6, requires_grad=True)
        states = torch.randn(2, 3, 8, requires_grad=True)
        output = SimpleNamespace(
            reasoning_context=context,
            reasoning_state_trace=states,
        )
        success = torch.tensor([True, False], dtype=torch.bool)

        captured = ReasoningTrajectoryBatch.from_model_output(
            output,
            task_ids=("same-task", "same-task"),
            verified_success=success,
        )

        self.assertFalse(captured.context.requires_grad)
        self.assertFalse(captured.state_trace.requires_grad)
        self.assertIsNone(captured.context.grad_fn)
        self.assertIsNone(captured.state_trace.grad_fn)
        self.assertTrue(torch.equal(captured.verified_success, success))

        with self.assertRaisesRegex(
            ReasoningCriticTrainingError,
            "must be boolean",
        ):
            ReasoningTrajectoryBatch.from_model_output(
                output,
                task_ids=("same-task", "same-task"),
                verified_success=torch.tensor([1, 0]),
            )

    def test_same_task_pairing_uses_verified_success_and_aligned_steps(self):
        batch = self.trajectory_batch()
        pairs = build_same_task_energy_pairs(batch)

        self.assertEqual(tuple(pairs.context.shape), (6, 6))
        self.assertEqual(tuple(pairs.positive_states.shape), (6, 8))
        self.assertEqual(tuple(pairs.negative_states.shape), (6, 8))
        self.assertEqual(
            pairs.task_ids,
            (
                "task-a",
                "task-a",
                "task-a",
                "task-b",
                "task-b",
                "task-b",
            ),
        )
        self.assertEqual(pairs.step_indices.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertFalse(pairs.context.requires_grad)
        self.assertFalse(pairs.positive_states.requires_grad)
        self.assertFalse(pairs.negative_states.requires_grad)

    def test_final_step_only_pairing_is_explicit(self):
        batch = self.trajectory_batch()
        pairs = build_same_task_energy_pairs(
            batch,
            ReasoningCriticPairingConfig(
                include_all_steps=False,
            ),
        )

        self.assertEqual(tuple(pairs.positive_states.shape), (2, 8))
        self.assertEqual(pairs.step_indices.tolist(), [2, 2])

    def test_pairing_rejects_reused_task_id_with_different_context(self):
        batch = self.trajectory_batch()
        context = batch.context.clone()
        context[1] = context[1] + 0.5
        mismatched = ReasoningTrajectoryBatch(
            task_ids=batch.task_ids,
            context=context,
            state_trace=batch.state_trace,
            verified_success=batch.verified_success,
        )

        with self.assertRaisesRegex(
            ReasoningCriticTrainingError,
            "mismatched reasoning contexts",
        ):
            build_same_task_energy_pairs(mismatched)

    def test_pairing_requires_both_verified_outcomes(self):
        batch = self.trajectory_batch()
        success_only = ReasoningTrajectoryBatch(
            task_ids=("task-a", "task-a"),
            context=batch.context[:2],
            state_trace=batch.state_trace[:2],
            verified_success=torch.tensor(
                [True, True],
                dtype=torch.bool,
            ),
        )

        with self.assertRaisesRegex(
            ReasoningCriticTrainingError,
            "no same-task success/failure",
        ):
            build_same_task_energy_pairs(success_only)

    def test_critic_pair_loss_backpropagates_only_into_critic(self):
        torch.manual_seed(203)
        critic = self.critic()
        pairs = build_same_task_energy_pairs(self.trajectory_batch())

        loss, positive_energy, negative_energy = reasoning_critic_pair_loss(
            critic,
            pairs,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(positive_energy.shape, negative_energy.shape)
        loss.backward()

        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in critic.parameters()
            )
        )
        self.assertFalse(pairs.context.requires_grad)
        self.assertFalse(pairs.positive_states.requires_grad)
        self.assertFalse(pairs.negative_states.requires_grad)

    def test_separate_critic_training_step_updates_critic(self):
        torch.manual_seed(204)
        critic = self.critic()
        optimizer = build_optimizer(
            critic,
            OptimizerConfig(
                lr=1e-3,
                weight_decay=0.0,
            ),
        )
        pairs = build_same_task_energy_pairs(self.trajectory_batch())
        before = {
            name: parameter.detach().clone()
            for name, parameter in critic.named_parameters()
        }

        metrics = train_reasoning_critic_step(
            critic,
            optimizer,
            pairs,
        )

        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
        self.assertTrue(torch.isfinite(torch.tensor(metrics.grad_norm)))
        self.assertEqual(metrics.pairs, 6)
        self.assertGreaterEqual(metrics.separation_rate, 0.0)
        self.assertLessEqual(metrics.separation_rate, 1.0)
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in critic.named_parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
