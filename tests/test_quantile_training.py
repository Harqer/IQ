from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from iq_model import StableLatentMoEConfig, StableLatentMoELayer
from iq_training import OptimizerConfig, TrainStepConfig, build_optimizer, train_step, TrainingError
from iq_training.quantile import QuantileBalancingError, QuantileBalancingWindow


class TinyLatentTrainingModel(nn.Module):
    """Trainable CPU integration target with real IQ Stable LatentMoE layers."""

    def __init__(self) -> None:
        super().__init__()
        config = StableLatentMoEConfig(
            hidden_size=16,
            latent_size=8,
            expert_intermediate_size=24,
            num_experts=6,
            top_k=2,
        )
        self.embed_tokens = nn.Embedding(31, 16)
        self.blocks = nn.ModuleList(
            [StableLatentMoELayer(config) for _ in range(2)]
        )
        self.lm_head = nn.Linear(16, 31, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ):
        x = self.embed_tokens(input_ids)
        for layer in self.blocks:
            x = layer(x).hidden_states
        logits = self.lm_head(x).float()
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(input_ids)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
            if mask.any():
                loss = losses[mask].mean()
            else:
                loss = losses.sum() * 0.0
        else:
            loss = losses.mean()
        return SimpleNamespace(loss=loss)


class QuantileBalancingTrainingTests(unittest.TestCase):
    @staticmethod
    def batches():
        return [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
            },
            {
                "input_ids": torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
            },
        ]

    def test_logical_batch_uses_old_bias_then_commits_exact_masked_quantile(self):
        torch.manual_seed(109)
        model = TinyLatentTrainingModel()
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0))
        observations: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        old_bias = [layer.moe.routing_bias.detach().clone() for layer in model.blocks]
        records: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        handles = []

        for idx, layer in enumerate(model.blocks):
            def record(module, inputs, output, idx=idx):
                observations[idx].append(output.raw_router_scores.detach().clone())
                records[idx].append(module.routing_bias.detach().clone())
            handles.append(layer.moe.register_forward_hook(record))

        batches = self.batches()
        try:
            metrics = train_step(
                model,
                optimizer,
                batches,
                TrainStepConfig(gradient_accumulation_steps=2),
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(metrics.tokens, 12)
        for idx, layer in enumerate(model.blocks):
            self.assertEqual(len(observations[idx]), 2)
            for old in records[idx]:
                torch.testing.assert_close(old, old_bias[idx], atol=0, rtol=0)
            masked = torch.cat(
                [
                    score[batch["attention_mask"].bool()]
                    for score, batch in zip(observations[idx], batches, strict=True)
                ],
                dim=0,
            )
            # The expected margin cutoff is derived using the pre-update bias.
            scores = masked.float()
            k = layer.moe.config.top_k
            n = layer.moe.config.num_experts
            cutoff = (scores + old_bias[idx]).topk(k + 1, dim=-1).values[:, -1]
            margins = scores - cutoff[:, None]
            expected = -torch.quantile(margins, q=1 - k / n, dim=0)
            expected -= expected.mean()
            torch.testing.assert_close(layer.moe.routing_bias, expected)
            self.assertNotEqual(float(layer.moe.routing_bias.abs().sum()), 0.0)

        for layer in model.blocks:
            self.assertFalse(any(isinstance(v, torch.Tensor) and v.grad_fn is not None for v in layer.moe.buffers()))

    def test_failure_discards_routing_observations_and_leaves_bias_unchanged(self):
        torch.manual_seed(110)
        model = TinyLatentTrainingModel()
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0))
        before = [layer.moe.routing_bias.detach().clone() for layer in model.blocks]
        valid = self.batches()[0]
        invalid = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        with self.assertRaises(TrainingError):
            train_step(
                model,
                optimizer,
                [valid, invalid],
                TrainStepConfig(gradient_accumulation_steps=2),
            )
        for layer, bias in zip(model.blocks, before, strict=True):
            torch.testing.assert_close(layer.moe.routing_bias, bias, atol=0, rtol=0)
            self.assertEqual(len(layer.moe._forward_hooks), 0)
        # A subsequent independent optimizer step cannot see prior observations.
        train_step(model, optimizer, [self.batches()[1]])
        self.assertTrue(any(
            not torch.equal(layer.moe.routing_bias, bias)
            for layer, bias in zip(model.blocks, before, strict=True)
        ))

    def test_evaluation_freezes_bias_and_checkpoint_roundtrip_preserves_it(self):
        torch.manual_seed(111)
        model = TinyLatentTrainingModel()
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0))
        train_step(model, optimizer, self.batches()[:1])
        saved = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        expected = [layer.moe.routing_bias.clone() for layer in model.blocks]
        model.eval()
        with torch.no_grad():
            for batch in self.batches():
                model(batch["input_ids"], labels=batch["input_ids"])
        for layer, bias in zip(model.blocks, expected, strict=True):
            torch.testing.assert_close(layer.moe.routing_bias, bias, atol=0, rtol=0)
        restored = TinyLatentTrainingModel()
        restored.load_state_dict(saved, strict=True)
        for layer, bias in zip(restored.blocks, expected, strict=True):
            torch.testing.assert_close(layer.moe.routing_bias, bias, atol=0, rtol=0)

    def test_zero_valid_tokens_does_not_commit_a_quantile(self):
        model = TinyLatentTrainingModel()
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0))
        old = [layer.moe.routing_bias.detach().clone() for layer in model.blocks]
        with self.assertRaisesRegex(QuantileBalancingError, "no valid routing"):
            train_step(
                model,
                optimizer,
                [{
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.zeros(1, 3, dtype=torch.long),
                }],
            )
        for layer, bias in zip(model.blocks, old, strict=True):
            torch.testing.assert_close(layer.moe.routing_bias, bias, atol=0, rtol=0)

    def test_exact_window_is_explicitly_single_process(self):
        model = TinyLatentTrainingModel()
        with QuantileBalancingWindow(model) as window:
            self.assertEqual(len(window.layers), 2)


if __name__ == "__main__":
    unittest.main()
