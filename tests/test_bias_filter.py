from __future__ import annotations

import tempfile
import unittest

import torch

from iq_transfer.bias_filter import (
    BiasFilterArtifact,
    BiasFilterError,
    BiasFilterGate,
    BiasFilterMetrics,
    CapabilityMetricRule,
    CausalValidation,
    apply_approved_filter,
    approve_bias_filter,
    detect_magnitude_outlier_dimensions,
    fit_leace_bias_filter,
    paired_counterfactual_batch,
)


class BiasFilterTests(unittest.TestCase):
    def test_pair_centering_cancels_shared_nuisance(self):
        shared = torch.tensor([[100.0, -30.0], [50.0, 20.0]])
        direction = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        a = shared - direction
        b = shared + direction

        x, z = paired_counterfactual_batch(a, b)

        expected = torch.cat((-direction, direction), dim=0)
        torch.testing.assert_close(x, expected)
        self.assertEqual(tuple(z.shape), (4, 1))
        self.assertTrue(torch.equal(z[:2], -torch.ones(2, 1)))
        self.assertTrue(torch.equal(z[2:], torch.ones(2, 1)))

    def test_artifact_leaves_protected_dimensions_exactly_unchanged(self):
        artifact = BiasFilterArtifact(
            feature_dim=4,
            concept_dim=1,
            target_concept="test",
            space_name="donor.layer.0.residual",
            active_dimensions=torch.tensor([1, 2, 3], dtype=torch.long),
            proj_left=torch.tensor([[1.0], [0.0], [0.0]]),
            proj_right=torch.tensor([[1.0, 0.0, 0.0]]),
            bias=torch.zeros(3),
        )
        x = torch.tensor([[99.0, 5.0, 2.0, 3.0]])

        y = artifact.apply(x)

        self.assertEqual(float(y[0, 0]), 99.0)
        self.assertEqual(float(y[0, 1]), 0.0)
        torch.testing.assert_close(y[0, 2:], x[0, 2:])

    def test_detects_high_magnitude_dimension_robustly(self):
        torch.manual_seed(7)
        x = torch.randn(4096, 16)
        x[:, 3] *= 1000.0

        mask = detect_magnitude_outlier_dimensions(x, mad_threshold=6.0)

        self.assertTrue(bool(mask[3]))
        self.assertEqual(int(mask.sum()), 1)

    def test_leace_reduces_counterfactual_leakage_and_protects_outlier(self):
        torch.manual_seed(13)
        n = 4096
        labels = torch.where(torch.arange(n) % 2 == 0, -1.0, 1.0).reshape(-1, 1)
        x = torch.randn(n, 12)
        x[:, 2] += labels[:, 0] * 4.0
        x[:, 0] *= 500.0

        fit_x, val_x = x[:2048], x[2048:]
        fit_z, val_z = labels[:2048], labels[2048:]
        protected = torch.zeros(12, dtype=torch.bool)
        protected[0] = True

        artifact, metrics = fit_leace_bias_filter(
            fit_x,
            fit_z,
            validation_x=val_x,
            validation_z=val_z,
            target_concept="political_asymmetry_test",
            space_name="donor.layer.0.residual",
            protected_dimensions=protected,
        )
        clean = artifact.apply(val_x)

        torch.testing.assert_close(clean[:, 0], val_x[:, 0])
        self.assertGreater(metrics.leakage_reduction, 0.95)
        self.assertLess(metrics.relative_mse, 0.25)

    def test_approval_fails_closed_on_capability_drop(self):
        artifact = BiasFilterArtifact(
            feature_dim=2,
            concept_dim=1,
            target_concept="test",
            space_name="space",
            active_dimensions=torch.tensor([0, 1], dtype=torch.long),
            proj_left=torch.zeros(2, 1),
            proj_right=torch.zeros(1, 2),
            bias=torch.zeros(2),
        )
        metrics = BiasFilterMetrics(
            leakage_before=1.0,
            leakage_after=0.1,
            leakage_reduction=0.9,
            relative_mse=0.001,
            mean_cosine=0.999,
            protected_dimensions=0,
        )

        with self.assertRaisesRegex(BiasFilterError, "coding"):
            approve_bias_filter(
                artifact,
                metrics,
                activation_gate=BiasFilterGate(),
                causal_validation=CausalValidation(
                    asymmetry_before=1.0,
                    asymmetry_after=0.2,
                    control_delta=0.0,
                ),
                baseline_capabilities={"coding": 80.0, "nlp": 75.0},
                filtered_capabilities={"coding": 75.0, "nlp": 74.9},
                capability_rules=(
                    CapabilityMetricRule("coding", max_degradation=0.5),
                    CapabilityMetricRule("nlp", max_degradation=0.5),
                ),
            )

    def test_multimodal_checkpoint_can_require_vision_retention(self):
        artifact = BiasFilterArtifact(
            feature_dim=2,
            concept_dim=1,
            target_concept="test",
            space_name="space",
            active_dimensions=torch.tensor([0, 1], dtype=torch.long),
            proj_left=torch.zeros(2, 1),
            proj_right=torch.zeros(1, 2),
            bias=torch.zeros(2),
            protected_modalities=("vision",),
            required_capability_metrics=("coding", "nlp", "vision_vqa"),
        )
        metrics = BiasFilterMetrics(1.0, 0.1, 0.9, 0.0, 1.0, 0)

        with self.assertRaisesRegex(BiasFilterError, "vision_vqa"):
            approve_bias_filter(
                artifact,
                metrics,
                activation_gate=BiasFilterGate(),
                causal_validation=CausalValidation(1.0, 0.2, 0.0),
                baseline_capabilities={"coding": 80.0, "nlp": 75.0},
                filtered_capabilities={"coding": 80.0, "nlp": 75.0},
                capability_rules=(
                    CapabilityMetricRule("coding", 0.5),
                    CapabilityMetricRule("nlp", 0.5),
                    CapabilityMetricRule("vision_vqa", 0.5),
                ),
            )

    def test_artifact_round_trip(self):
        artifact = BiasFilterArtifact(
            feature_dim=4,
            concept_dim=1,
            target_concept="test",
            space_name="space",
            active_dimensions=torch.tensor([1, 2, 3], dtype=torch.long),
            proj_left=torch.randn(3, 1),
            proj_right=torch.randn(1, 3),
            bias=torch.randn(3),
            protected_modalities=("vision",),
            required_capability_metrics=("coding", "nlp", "vision_vqa"),
        )
        with tempfile.TemporaryDirectory() as temp:
            artifact.write(temp)
            loaded = BiasFilterArtifact.read(temp)

        self.assertEqual(loaded.fingerprint, artifact.fingerprint)
        self.assertEqual(loaded.protected_modalities, ("vision",))
        self.assertEqual(
            loaded.required_capability_metrics,
            ("coding", "nlp", "vision_vqa"),
        )
        torch.testing.assert_close(loaded.proj_left, artifact.proj_left)

    def test_approved_filter_requires_matching_artifact(self):
        a = BiasFilterArtifact(
            feature_dim=2,
            concept_dim=1,
            target_concept="a",
            space_name="space",
            active_dimensions=torch.tensor([0, 1], dtype=torch.long),
            proj_left=torch.zeros(2, 1),
            proj_right=torch.zeros(1, 2),
            bias=torch.zeros(2),
        )
        b = BiasFilterArtifact(
            feature_dim=2,
            concept_dim=1,
            target_concept="b",
            space_name="space",
            active_dimensions=torch.tensor([0, 1], dtype=torch.long),
            proj_left=torch.zeros(2, 1),
            proj_right=torch.zeros(1, 2),
            bias=torch.zeros(2),
        )
        approval = approve_bias_filter(
            a,
            BiasFilterMetrics(1.0, 0.1, 0.9, 0.0, 1.0, 0),
            activation_gate=BiasFilterGate(),
            causal_validation=CausalValidation(1.0, 0.2, 0.0),
            baseline_capabilities={"coding": 80.0},
            filtered_capabilities={"coding": 80.0},
            capability_rules=(CapabilityMetricRule("coding", 0.1),),
        )
        with self.assertRaisesRegex(BiasFilterError, "approval"):
            apply_approved_filter(b, approval, torch.randn(2, 2))


if __name__ == "__main__":
    unittest.main()
