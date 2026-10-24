"""Numerical tests for a matrix-free shared WARM tangent-basis probe."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from iq_transfer.glm53_expert_informed_basis import (
    ExpertBasisProbeError,
    gram_action,
    novel_orthogonal_directions,
    projection_components,
    rotate_energy,
    rotated_basis,
    run_expert_informed_probe,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class ExpertInformedBasisTests(unittest.TestCase):
    def test_operator_gram_action_and_rotation_formula(self):
        rng = np.random.default_rng(47)
        p, _ = np.linalg.qr(rng.normal(size=(12, 5)))
        candidate = np.array([0, 3])
        direction = rng.normal(size=(12, 2))
        direction -= p @ (p.T @ direction)
        r, _ = np.linalg.qr(direction)
        angle = 0.17
        result = rotated_basis(p, candidate, r, angle)
        self.assertLess(np.max(np.abs(result.T @ result - np.eye(5))), 1e-12)
        for kind in ("gate", "up", "down"):
            shape = (8, 12) if kind != "down" else (12, 8)
            w = rng.normal(size=shape).astype(np.float32)
            q0, q1 = p[:, candidate], result[:, candidate]
            want_gram = w.T @ (w @ q0) if kind != "down" else w @ (w.T @ q0)
            self.assertTrue(np.allclose(
                gram_action(w, q0, kind), want_gram,
            ))
            parts = projection_components(w, q0, r, kind)
            full_old = w @ p if kind != "down" else w.T @ p
            full_new = w @ result if kind != "down" else w.T @ result
            old_ratio = float(np.sum(full_old * full_old) / np.sum(w * w))
            actual = float(np.sum(full_new * full_new) / np.sum(w * w))
            predicted = rotate_energy(old_ratio, parts, angle)
            self.assertAlmostEqual(predicted, actual, delta=1e-8)

    def test_tangent_directions_do_not_modify_other_basis_columns(self):
        rng = np.random.default_rng(13)
        p, _ = np.linalg.qr(rng.normal(size=(16, 6)))
        subset = np.array([0, 2, 5])
        raw = rng.normal(size=(16, len(subset)))
        r = novel_orthogonal_directions(p, raw)
        self.assertLess(np.max(np.abs(p.T @ r)), 1e-12)
        self.assertLess(np.max(np.abs(r.T @ r - np.eye(len(subset)))), 1e-12)
        rotated = rotated_basis(p, subset, r, 0.1)
        for col in (1, 3, 4):
            self.assertTrue(np.array_equal(rotated[:, col], p[:, col]))

    def test_probe_reports_train_and_validation_without_replacing_map(self):
        rng = np.random.default_rng(4)
        p, _ = np.linalg.qr(rng.normal(size=(10, 4)))
        hidden, intermediate = 10, 2048
        operators = {}
        baseline = []
        for donor_id in (77, 87, 246, 131):
            values = {}
            ratios = {}
            for kind in ("gate", "up", "down"):
                shape = ((intermediate, hidden) if kind != "down"
                         else (hidden, intermediate))
                w = rng.normal(size=shape).astype(np.float32)
                values[kind] = w
                proj = w @ p if kind != "down" else w.T @ p
                ratios[kind] = float(
                    np.sum(proj * proj) / np.sum(w * w)
                )
            operators[donor_id] = values
            baseline.append(ratios)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            projected = root / "baseline"
            calibration = root / "calibration"
            raw.mkdir()
            projected.mkdir()
            calibration.mkdir()
            output = root / "candidate"
            from iq_transfer.complete_transplant import canonical_glm53_config
            fp = canonical_glm53_config(routed_experts=64).fingerprint
            (raw / "raw_expert_manifest.json").write_text(json.dumps({
                "source_repo": REPO,
                "source_revision": REVISION,
                "not_an_iq_checkpoint": True,
                "target_experts": 64,
                "target_config_fingerprint": fp,
                "layers": [
                    {"stage": 1, "source_layer": 3,
                     "exported_donor_ids": [77, 87, 246, 131]}
                ],
            }))
            fake_config = SimpleNamespace(
                fingerprint=fp,
                stable_moe=SimpleNamespace(expert_intermediate_size=2048),
            )
            def mock_reader(_raw, _manifest, _layer, donor_id, kind,
                            _hidden, _intermediate):
                return operators[donor_id][kind]
            with (
                patch(
                    "iq_transfer.glm53_expert_informed_basis.canonical_glm53_config",
                    return_value=fake_config,
                ),
                patch(
                    "iq_transfer.glm53_expert_informed_basis._calibration_map",
                    return_value=(p, 5, {"stage-01-latent.safetensors": "verified"}),
                ),
                patch(
                    "iq_transfer.glm53_expert_informed_basis._read_baseline",
                    return_value=baseline,
                ),
                patch(
                    "iq_transfer.glm53_expert_informed_basis._donor_operator",
                    side_effect=mock_reader,
                ),
            ):
                report = run_expert_informed_probe(
                    raw_dir=raw, baseline_dir=projected,
                    calibration_dir=calibration, output_dir=output,
                    candidate_columns=2, seed=14,
                )
            self.assertTrue(report["not_canonical_calibration"])
            self.assertTrue(report["weight_energy_only"])
            self.assertEqual(report["train_slots"], [0, 1, 2])
            self.assertEqual(report["validation_slot"], 3)
            self.assertEqual(len(report["modified_basis_columns"]), 2)
            self.assertGreater(
                report["candidate_results"]["train_mean_energy_delta"], 0
            )
            self.assertTrue((output / "candidate-stage-latent.safetensors").is_file())
            self.assertTrue((output / "basis_probe_report.json").is_file())
            self.assertEqual(len(report["candidate_results"]["experts"]), 4)

    def test_guardrails_reject_bad_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (0, 33):
                with self.assertRaisesRegex(ExpertBasisProbeError,
                                            "candidate_columns"):
                    run_expert_informed_probe(
                        raw_dir=tmp, baseline_dir=tmp, calibration_dir=tmp,
                        output_dir=Path(tmp) / "out",
                        candidate_columns=value,
                    )
            with self.assertRaisesRegex(ExpertBasisProbeError,
                                        "inside input"):
                run_expert_informed_probe(
                    raw_dir=tmp, baseline_dir=tmp, calibration_dir=tmp,
                    output_dir=Path(tmp) / "out",
                )


if __name__ == "__main__":
    unittest.main()
