"""CPU-only numerical tests for GLM expert fidelity diagnostics."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iq_transfer.glm53_expert_fidelity import (
    GLM53FidelityError,
    comparison,
    evaluate_expert,
    load_capture,
)


class GLM53FidelityTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator(device="cpu").manual_seed(31)
        # A genuinely non-axis-aligned orthonormal subspace.
        q, _ = torch.linalg.qr(torch.randn(12, 5, generator=gen))
        self.p = q
        self.donor = {
            "gate": torch.randn(8, 12, generator=gen),
            "up": torch.randn(8, 12, generator=gen),
            "down": torch.randn(12, 8, generator=gen),
        }
        self.recipient = {
            "gate": self.donor["gate"] @ q,
            "up": self.donor["up"] @ q,
            "down": q.T @ self.donor["down"],
        }
        self.generator = gen

    def test_subspace_algebra_nearly_exact(self):
        z = torch.randn(24, 5, generator=self.generator)
        report = evaluate_expert(
            self.donor, self.recipient, self.p, z @ self.p.T
        )
        self.assertLess(
            report["projector_numerical_error"]["relative_rmse"], 0.00001
        )
        self.assertLess(
            report["total_error_against_original"]["relative_rmse"], 0.00001
        )
        self.assertLess(report["input_discarded_energy_fraction"], 0.00001)

    def test_off_subspace_perturbation_detected(self):
        z = torch.randn(64, 5, generator=self.generator)
        noise = torch.randn(64, 12, generator=self.generator)
        orthogonal = noise - (noise @ self.p) @ self.p.T
        x = z @ self.p.T + orthogonal
        report = evaluate_expert(self.donor, self.recipient, self.p, x)
        self.assertGreater(report["input_discarded_energy_fraction"], 0.1)
        self.assertGreater(
            report["total_error_against_original"]["relative_rmse"], 0.01
        )
        self.assertLess(
            report["projector_numerical_error"]["relative_rmse"], 0.00001
        )
        self.assertGreater(
            report["discarded_input_effect"]["relative_rmse"], 0.01
        )

    def test_activation_capture_is_explicit_and_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "source_hidden_states.npy"
            x = np.arange(120, dtype=np.float32).reshape(10, 12)
            np.save(file, x)
            loaded = load_capture(file, width=12, max_samples=3)
            self.assertEqual(tuple(loaded.shape), (3, 12))
            self.assertEqual(float(loaded[2, 0]), float(x[2, 0]))
            with self.assertRaisesRegex(GLM53FidelityError, "shape"):
                load_capture(file, width=11, max_samples=3)
            np.save(file, np.full((2, 12), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(GLM53FidelityError, "nonfinite"):
                load_capture(file, width=12, max_samples=2)

    def test_bad_shapes_and_zero_baseline_rejected_or_bounded(self):
        x = torch.randn(2, 11, generator=self.generator)
        with self.assertRaisesRegex(GLM53FidelityError, "dimensions"):
            evaluate_expert(self.donor, self.recipient, self.p, x)
        with self.assertRaisesRegex(GLM53FidelityError, "geometry"):
            comparison(torch.ones(2, 3), torch.ones(2, 4))
        metrics = comparison(torch.zeros(2, 3), torch.ones(2, 3))
        self.assertEqual(metrics["rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
