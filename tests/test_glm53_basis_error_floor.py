"""Analytic fixed-subspace floor and oracle tests for GLM FFN compression."""
from __future__ import annotations

import unittest

import torch

from iq_transfer.glm53_basis_error_floor import (
    GLM53ErrorBudgetError,
    output_error_budget,
)
from iq_transfer.glm53_expert_fidelity import expert_function


class OutputErrorFloorTests(unittest.TestCase):
    def setUp(self):
        self.donor = {name: torch.eye(3, dtype=torch.float32)
                      for name in ("gate", "up", "down")}
        self.x = torch.tensor([[1., 2., 3.], [4., 2., 1.]])
        self.basis = torch.eye(3)[:, :2].contiguous()
        self.y = expert_function(self.x, **self.donor)

    def test_projection_is_exact_minimum_in_fixed_output_subspace(self):
        best_possible = (self.y @ self.basis) @ self.basis.T
        result = output_error_budget(
            donor=self.donor,
            input_activations=self.x,
            reference=self.y,
            estimate=best_possible,
            basis=self.basis,
        )
        self.assertAlmostEqual(
            result["output_subspace_floor_relative_rmse"],
            result["actual_relative_rmse"], places=6,
        )
        self.assertLess(result["within_subspace_relative_rmse"], 1e-6)
        self.assertAlmostEqual(
            result["output_floor_share_of_actual_squared_error"], 1.0, places=6,
        )
        self.assertAlmostEqual(
            result["full_precision_input_projected_oracle_relative_rmse"],
            result["output_subspace_floor_relative_rmse"], places=6,
        )
        self.assertLess(
            result["projected_weights_vs_float32_oracle_relative_rmse"], 1e-6
        )

    def test_additional_within_basis_error_uses_pythagorean_budget(self):
        prediction = (self.y @ self.basis) @ self.basis.T
        prediction[0, 0] += 3
        result = output_error_budget(
            donor=self.donor, input_activations=self.x,
            reference=self.y, estimate=prediction, basis=self.basis,
        )
        self.assertGreater(
            result["actual_relative_rmse"],
            result["output_subspace_floor_relative_rmse"],
        )
        self.assertGreater(result["within_subspace_relative_rmse"], 0)
        self.assertLess(result["squared_error_decomposition_abs_gap"], 1e-8)
        self.assertGreater(result["actual_minus_output_floor_relative_rmse"], 0)

    def test_rejects_nonorthogonal_projection_that_breaks_decomposition(self):
        bad = self.basis.clone()
        bad[0, 0] = 2
        predicted = (self.y @ bad) @ bad.T
        with self.assertRaisesRegex(
            GLM53ErrorBudgetError, "does not decompose"
        ):
            output_error_budget(
                donor=self.donor, input_activations=self.x,
                reference=self.y, estimate=predicted, basis=bad,
            )

    def test_rejects_nonfinite_reference(self):
        bad = self.y.clone()
        bad[0, 1] = float("nan")
        with self.assertRaisesRegex(GLM53ErrorBudgetError, "nonfinite"):
            output_error_budget(
                donor=self.donor, input_activations=self.x,
                reference=bad, estimate=self.y, basis=self.basis,
            )


if __name__ == "__main__":
    unittest.main()
