from __future__ import annotations

import unittest

import torch

from iq_transfer.mohawk import (
    MohawkConfig,
    MohawkStage,
    hidden_alignment_loss,
    logits_distillation_loss,
    matrix_orientation_loss,
    mohawk_loss,
)


class MohawkObjectiveTests(unittest.TestCase):
    def test_stage1_zero_for_identical_mixers(self):
        x = torch.randn(2, 3, 4, 4)
        loss = matrix_orientation_loss(x, x.clone())
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_stage2_masks_padding(self):
        teacher = torch.zeros(1, 3, 2)
        student = teacher.clone()
        student[:, 2] = 100.0
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        self.assertAlmostEqual(
            float(hidden_alignment_loss(teacher, student, token_mask=mask)),
            0.0,
            places=7,
        )

    def test_stage3_zero_for_identical_logits(self):
        logits = torch.randn(2, 3, 7)
        loss = logits_distillation_loss(logits, logits.clone(), temperature=2.0)
        self.assertLess(abs(float(loss)), 1e-6)

    def test_dispatch_fails_closed_when_stage_inputs_are_missing(self):
        with self.assertRaisesRegex(ValueError, "Stage 1 requires"):
            mohawk_loss(
                MohawkConfig(stage=MohawkStage.MATRIX_ORIENTATION)
            )

    def test_dispatch_selects_hidden_alignment(self):
        teacher = torch.zeros(1, 2, 3)
        student = torch.ones(1, 2, 3)
        config = MohawkConfig(
            stage=MohawkStage.HIDDEN_ALIGNMENT,
            hidden_weight=0.5,
        )
        expected = 0.5 * hidden_alignment_loss(teacher, student)
        actual = mohawk_loss(
            config,
            teacher_hidden=teacher,
            student_hidden=student,
        )
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
