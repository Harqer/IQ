"""Proof-carrying, no-training regression for GLM/Mamba weight-only claims.

Uses synthetic CPU weights; real source weights are NOT downloaded. The rank
theorem itself applies to 6144->2048 irrespective of checkpoint contents.
"""
from __future__ import annotations

import unittest
import torch

from iq_transfer.glm_first_no_learning import (
    prove_weight_only_limits,
    probe_untrained_hybrid,
)
from iq_model.glm_first_mamba import GLMFirstIntegrationError
from test_glm_first_mamba_hybrid import (
    CONTRACT,
    ToyGLM,
    ToyMambaModel,
    wrap_preloaded_official_models,
)


class NoLearningEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(142)
        self.glm = ToyGLM().eval()
        self.donor = ToyMambaModel().eval()
        self.hybrid = wrap_preloaded_official_models(
            self.glm, self.donor, contract=CONTRACT
        ).eval()
        self.ids = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=torch.long)

    def test_6144_to_2048_is_mathematically_noninvertible(self):
        result = prove_weight_only_limits()
        self.assertEqual(result.maximal_linear_rank, 2048)
        self.assertEqual(result.minimum_nullity, 4096)
        self.assertFalse(result.invertible_global_linear_adapter_possible)
        self.assertTrue(result.zero_gate_exact_baseline_by_construction)
        self.assertFalse(result.weight_only_replacement_certified)
        self.assertFalse(result.mamba_donor_native_interblock_connectivity_guaranteed)

    def test_explicit_nullspace_counterexample_without_learning(self):
        """Two distinct hidden vectors have the same 4-wide projected state."""
        torch.manual_seed(7)
        A = torch.randn(4, 8, dtype=torch.float64)
        _, _, vh = torch.linalg.svd(A, full_matrices=True)
        invisible_direction = vh[-1]
        x = torch.randn(8, dtype=torch.float64)
        self.assertGreater(float(invisible_direction.norm()), 0.99)
        torch.testing.assert_close(
            A @ x,
            A @ (x + invisible_direction),
            rtol=0,
            atol=1e-12,
        )
        self.assertFalse(torch.equal(x, x + invisible_direction))

    def test_zero_gates_preserve_teacher_logits_exactly(self):
        result = probe_untrained_hybrid(self.glm, self.hybrid, self.ids)
        self.assertTrue(result.exact_on_tested_inputs)
        self.assertEqual(result.max_abs_logit_change, 0.0)
        self.assertEqual(result.active_nonzero_branches, 0)
        # Probe equality is not proof for arbitrary future prompts/weights.
        self.assertFalse(result.global_equivalence_proven)
        self.assertFalse(result.weight_only_replacement_certified)

    def test_nonzero_weight_only_branch_changes_outputs(self):
        self.hybrid.enable_branches(True)
        with torch.no_grad():
            self.hybrid.branches[0].gate.fill_(0.5)
        result = probe_untrained_hybrid(self.glm, self.hybrid, self.ids)
        self.assertGreater(result.max_abs_logit_change, 0)
        self.assertEqual(result.active_nonzero_branches, 1)
        self.assertFalse(result.exact_on_tested_inputs)
        self.assertFalse(result.weight_only_replacement_certified)

    def test_reject_invalid_tokens_and_train_mode(self):
        self.hybrid.train(True)
        with self.assertRaisesRegex(GLMFirstIntegrationError, "eval"):
            probe_untrained_hybrid(self.glm, self.hybrid, self.ids)
        self.hybrid.eval()
        with self.assertRaisesRegex(GLMFirstIntegrationError, "outside"):
            probe_untrained_hybrid(
                self.glm, self.hybrid,
                torch.tensor([[CONTRACT.glm_vocab]], dtype=torch.long)
            )

    def test_allocation_shapes_never_suffice_for_donor_basis_parity(self):
        result = prove_weight_only_limits(8, 8)
        self.assertTrue(result.invertible_global_linear_adapter_possible)
        self.assertFalse(result.nonzero_branch_equivalence_proven)
        self.assertFalse(result.mamba_donor_native_input_distribution_guaranteed)


if __name__ == "__main__":
    unittest.main()
