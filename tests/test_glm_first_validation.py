"""Strict GLM teacher/zero-gated hybrid differential forward tests (CPU)."""
from __future__ import annotations

import unittest

import torch

from iq_model.glm_first_validation import verify_zero_gate_reference_parity
from iq_model.glm_first_mamba import GLMFirstIntegrationError
from test_glm_first_mamba_hybrid import (
    CONTRACT, ToyGLM, ToyMambaModel, wrap_preloaded_official_models,
)


class GLMTeacherBaselineParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(49)
        self.glm = ToyGLM().eval()
        self.mamba = ToyMambaModel().eval()
        self.hybrid = wrap_preloaded_official_models(
            self.glm, self.mamba, contract=CONTRACT
        ).eval()
        self.ids = torch.tensor([[1, 3, 5, 8]])

    def test_exact_reference_logits_and_clean_hooks(self):
        report = verify_zero_gate_reference_parity(
            self.glm, self.hybrid, self.ids, logits_to_keep=1
        )
        self.assertTrue(report.exact)
        self.assertEqual(report.max_abs_error, 0.0)
        self.assertEqual(report.mean_abs_error, 0.0)
        self.assertEqual(
            sum(len(block._forward_hooks) for block in self.glm.model.layers), 0
        )

    def test_activated_branch_cannot_claim_baseline_parity(self):
        self.hybrid.enable_branches()
        with self.assertRaisesRegex(GLMFirstIntegrationError, "disabled"):
            verify_zero_gate_reference_parity(self.glm, self.hybrid, self.ids)

    def test_bad_token_ids_are_rejected(self):
        bad = torch.tensor([[CONTRACT.glm_vocab]])
        with self.assertRaisesRegex(GLMFirstIntegrationError, "outside"):
            verify_zero_gate_reference_parity(self.glm, self.hybrid, bad)

    def test_wrong_teacher_is_rejected(self):
        with self.assertRaisesRegex(GLMFirstIntegrationError, "exact reference"):
            verify_zero_gate_reference_parity(ToyGLM(), self.hybrid, self.ids)


if __name__ == "__main__":
    unittest.main()
