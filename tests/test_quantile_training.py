from __future__ import annotations

import unittest
from math import ceil
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

    def test_histogram_backend_matches_published_required_bias_bin_equation(self):
        # Kimi K3 Appendix D: H[j,b] counts r=alpha-s, and recovers
        # b_hat = lower + (bin + clamp((mk/n - prior)/count,0,1)) * width.
        torch.manual_seed(112)
        model = TinyLatentTrainingModel()
        samples = self.batches()
        observations: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        handles = []
        for idx, layer in enumerate(model.blocks):
            def capture(module, inputs, output, idx=idx):
                observations[idx].append(output.raw_router_scores.detach().float().clone())
            handles.append(layer.moe.register_forward_hook(capture))
        try:
            with QuantileBalancingWindow(
                model, backend="histogram", histogram_bins=64
            ) as window:
                for batch in samples:
                    window.begin_microbatch(batch["attention_mask"])
                    try:
                        model(
                            batch["input_ids"],
                            labels=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    finally:
                        window.end_microbatch()
                proposals = window.propose()
                self.assertEqual(window.tokens, 12)
                for idx, layer in enumerate(model.blocks):
                    scores = torch.cat(
                        [
                            record[batch["attention_mask"].bool()]
                            for record, batch in zip(
                                observations[idx], samples, strict=True
                            )
                        ]
                    )
                    bias = layer.moe.routing_bias.detach()
                    n = scores.shape[-1]
                    k = layer.moe.config.top_k
                    lower, upper = float(bias.min()) - 1, float(bias.max()) + 1
                    width = (upper - lower) / 64
                    cutoff = (scores + bias).topk(k + 1, dim=-1).values[:, -1]
                    required = cutoff[:, None] - scores
                    expected_hist = torch.zeros(n, 64, dtype=torch.int64)
                    for expert in range(n):
                        for value in required[:, expert]:
                            index = max(0, min(63, int((float(value) - lower) / width)))
                            expected_hist[expert, index] += 1
                    name = f"blocks.{idx}.moe"
                    self.assertTrue(torch.equal(window._histograms[name], expected_hist))
                    target = len(scores) * k / n
                    estimates = []
                    for expert in range(n):
                        cumulative = expected_hist[expert].cumsum(0)
                        selected = int(torch.nonzero(cumulative >= ceil(target))[0])
                        prior = int(cumulative[selected - 1]) if selected else 0
                        count = int(expected_hist[expert, selected])
                        fraction = max(0.0, min(1.0, (target - prior) / count))
                        estimates.append(lower + (selected + fraction) * width)
                    expected = torch.tensor(estimates)
                    expected -= expected.mean()
                    torch.testing.assert_close(
                        proposals[name], expected, atol=1e-6, rtol=1e-6
                    )
        finally:
            for handle in handles:
                handle.remove()

    def test_histogram_microbatch_partition_invariance_and_commit(self):
        torch.manual_seed(113)
        model = TinyLatentTrainingModel()
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1]])
        proposals = []
        for sizes in ((8,), (3, 2, 3), (1,) * 8):
            with QuantileBalancingWindow(
                model, backend="histogram", histogram_bins=128
            ) as window:
                start = 0
                for size in sizes:
                    window.begin_microbatch(mask[:, start : start + size])
                    try:
                        ids = inputs[:, start : start + size]
                        model(
                            ids,
                            labels=ids,
                            attention_mask=mask[:, start : start + size],
                        )
                    finally:
                        window.end_microbatch()
                    start += size
                proposals.append(window.propose())
                self.assertEqual(window.tokens, 7)
        for other in proposals[1:]:
            for name in proposals[0]:
                torch.testing.assert_close(
                    other[name], proposals[0][name], atol=0, rtol=0
                )
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0))
        train_step(
            model, optimizer,
            [{"input_ids": inputs, "attention_mask": mask}],
            TrainStepConfig(
                quantile_balancing_backend="histogram",
                quantile_histogram_bins=128,
            ),
        )
        for idx, layer in enumerate(model.blocks):
            torch.testing.assert_close(
                layer.moe.routing_bias,
                proposals[0][f"blocks.{idx}.moe"],
                atol=0,
                rtol=0,
            )

    def test_exact_window_is_explicitly_single_process(self):
        model = TinyLatentTrainingModel()
        with QuantileBalancingWindow(model) as window:
            self.assertEqual(len(window.layers), 2)


if __name__ == "__main__":
    unittest.main()
