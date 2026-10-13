from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from iq_model import HybridLayerType
from iq_training import IQGLM53MohawkStage2, MohawkTrainingError, TrainStepConfig, train_step
from iq_transfer.mohawk import (
    MohawkConfig,
    MohawkStage,
    hidden_alignment_loss,
    logits_distillation_loss,
    matrix_orientation_loss,
    mohawk_loss,
)


class _ResidualExpert(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor):
        return SimpleNamespace(hidden_states=x + self.linear(x))


class _SmallAutogradBackbone(nn.Module):
    """CPU autograd fixture only; not a replacement for production Mamba-3."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(schedule=SimpleNamespace(layers=(
            HybridLayerType.MAMBA3, HybridLayerType.CSA, HybridLayerType.MOE,
        )))
        self.embed_tokens = nn.Embedding(16, width)
        self.layers = nn.ModuleList([
            nn.Linear(width, width),
            nn.Linear(width, width),
            _ResidualExpert(width),
        ])

    def forward(self, input_ids, **kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            output = layer(hidden)
            hidden = getattr(output, "hidden_states", output)
        return SimpleNamespace(logits=hidden)


class MohawkObjectivesTest(unittest.TestCase):
    def test_stage1_frobenius_reduces_when_mixers_match(self):
        x = torch.eye(4).repeat(1, 2, 1, 1)
        self.assertAlmostEqual(float(matrix_orientation_loss(x, x)), 0.0, places=7)
        self.assertGreater(float(matrix_orientation_loss(x, torch.zeros_like(x))), 0.0)

    def test_stage1_rejects_zero_or_invalid_mask(self):
        x = torch.eye(4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            matrix_orientation_loss(x, x, mask=torch.zeros(4, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "boolean"):
            matrix_orientation_loss(x, x, mask=torch.ones(4, 4))

    def test_stage2_ignores_padding_and_backpropagates(self):
        teacher = torch.zeros(1, 3, 4)
        student = torch.zeros_like(teacher, requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss = hidden_alignment_loss(teacher, student, token_mask=mask)
        self.assertAlmostEqual(float(loss), 0, places=6)
        student = torch.ones_like(teacher, requires_grad=True)
        loss = hidden_alignment_loss(teacher, student, token_mask=mask)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.equal(student.grad[:, 2], torch.zeros_like(student.grad[:, 2])))

    def test_stage3_rejects_cross_tokenizer_logits_even_if_shapes_match(self):
        logits = torch.randn(1, 3, 11)
        with self.assertRaisesRegex(ValueError, "identical verified tokenizer"):
            logits_distillation_loss(
                logits, logits,
                teacher_tokenizer_sha256="a" * 64,
                student_tokenizer_sha256="b" * 64,
            )

    def test_stage3_same_tokenizer_has_zero_self_kl(self):
        logits = torch.randn(1, 3, 11)
        loss = logits_distillation_loss(
            logits, logits,
            teacher_tokenizer_sha256="a" * 64,
            student_tokenizer_sha256="a" * 64,
        )
        self.assertLess(abs(float(loss)), 1e-6)

    def test_dispatch_requires_real_stage_inputs(self):
        with self.assertRaisesRegex(ValueError, "Stage 1"):
            mohawk_loss(MohawkConfig(stage=MohawkStage.MATRIX_ORIENTATION))
        with self.assertRaisesRegex(ValueError, "Stage 2"):
            mohawk_loss(MohawkConfig(stage=MohawkStage.HIDDEN_ALIGNMENT))


class MohawkIQStage2TrainingTest(unittest.TestCase):
    def _wrapper(self, *, train_mamba: bool = False):
        return IQGLM53MohawkStage2(
            _SmallAutogradBackbone(),
            source_glm_layer=20,
            target_physical_layer=2,
            alignment_manifest_sha256="a" * 64,
            train_preceding_mamba=train_mamba,
        )

    def test_default_freezes_mamba_and_lexical_boundary(self):
        wrapper = self._wrapper()
        self.assertEqual(wrapper.trainable_layers, (1, 2))
        self.assertFalse(wrapper.student.embed_tokens.weight.requires_grad)
        self.assertFalse(wrapper.student.layers[0].weight.requires_grad)
        self.assertTrue(wrapper.student.layers[1].weight.requires_grad)
        self.assertTrue(wrapper.student.layers[2].linear.weight.requires_grad)

    def test_explicit_mamba_unfreeze(self):
        wrapper = self._wrapper(train_mamba=True)
        self.assertEqual(wrapper.trainable_layers, (0, 1, 2))
        self.assertTrue(wrapper.student.layers[0].weight.requires_grad)

    def test_real_gradient_updates_in_canonical_train_step(self):
        torch.manual_seed(7)
        wrapper = self._wrapper()
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        teacher = torch.zeros(1, 4, 8)
        optimizer = torch.optim.AdamW(
            [p for p in wrapper.parameters() if p.requires_grad],
            lr=1e-3,
        )
        old = wrapper.student.layers[1].weight.detach().clone()
        metrics = train_step(
            wrapper, optimizer,
            [{"input_ids": ids, "teacher_aligned_hidden": teacher, "teacher_alignment_mask": torch.ones_like(ids, dtype=torch.bool)}],
            TrainStepConfig(precision="fp32"),
        )
        self.assertGreater(metrics.loss, 0)
        self.assertFalse(torch.equal(wrapper.student.layers[1].weight.detach(), old))
        self.assertEqual(sum(len(x._forward_hooks) for x in wrapper.student.layers), 0)

    def test_stage2_rejects_bad_target_shape(self):
        wrapper = self._wrapper()
        with self.assertRaisesRegex(MohawkTrainingError, "not aligned"):
            wrapper(
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                teacher_aligned_hidden=torch.zeros(1, 2, 8),
                teacher_alignment_mask=torch.ones(1, 3, dtype=torch.bool),
            )

    def test_cannot_select_non_moe_anchor(self):
        with self.assertRaisesRegex(MohawkTrainingError, "MoE pair"):
            IQGLM53MohawkStage2(
                _SmallAutogradBackbone(),
                source_glm_layer=20,
                target_physical_layer=1,
                alignment_manifest_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
