from __future__ import annotations

import unittest

import numpy as np

from iq_transfer import (
    WarmRemapError,
    WarmWeightOperator,
    fit_weight_orthogonal_remap,
    orthogonality_error,
)


class WarmRemapTests(unittest.TestCase):
    def test_builds_dense_semi_orthogonal_map(self):
        rng = np.random.default_rng(101)
        a = rng.normal(size=(7, 6))
        b = rng.normal(size=(6, 5))
        result = fit_weight_orthogonal_remap(
            (
                WarmWeightOperator(a, "input", "a"),
                WarmWeightOperator(b, "output", "b"),
            ),
            source_features=6,
            target_features=4,
            source_space="source",
            target_space="target",
        )
        q = result.coordinate_map.matrix
        self.assertEqual(q.shape, (6, 4))
        self.assertLess(orthogonality_error(q), 1e-10)
        self.assertGreater(result.diagnostics.retained_weight_energy, 0.0)
        self.assertLessEqual(result.diagnostics.retained_weight_energy, 1.0)
        self.assertEqual(result.diagnostics.operator_count, 2)
        # This must not collapse back to one-hot channel subcloning.
        self.assertTrue(np.any((np.abs(q) > 1e-6).sum(axis=0) > 1))

    def test_operator_scale_does_not_change_basis(self):
        rng = np.random.default_rng(102)
        a = rng.normal(size=(8, 6))
        b = rng.normal(size=(6, 4))
        first = fit_weight_orthogonal_remap(
            (
                WarmWeightOperator(a, "input", "a"),
                WarmWeightOperator(b, "output", "b"),
            ),
            source_features=6,
            target_features=3,
            source_space="s",
            target_space="t",
        ).coordinate_map.matrix
        second = fit_weight_orthogonal_remap(
            (
                WarmWeightOperator(1000.0 * a, "input", "a"),
                WarmWeightOperator(0.001 * b, "output", "b"),
            ),
            source_features=6,
            target_features=3,
            source_space="s",
            target_space="t",
        ).coordinate_map.matrix
        projector_a = first @ first.T
        projector_b = second @ second.T
        self.assertTrue(np.allclose(projector_a, projector_b, atol=1e-5))

    def test_rejects_wrong_feature_orientation(self):
        with self.assertRaisesRegex(WarmRemapError, "input width"):
            fit_weight_orthogonal_remap(
                (WarmWeightOperator(np.ones((3, 5)), "input", "bad"),),
                source_features=6,
                target_features=2,
                source_space="s",
                target_space="t",
            )


if __name__ == "__main__":
    unittest.main()
