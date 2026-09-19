from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from iq_transfer import (
    ActivationPair,
    CalibrationError,
    CalibrationManifest,
    CalibrationRecord,
    CalibrationSplit,
    PhiLayerCalibration,
    merge_coordinate_maps,
    solve_activation_pair,
    solve_layer_correspondence,
    solve_phi_layer_maps,
)


class CalibrationTests(unittest.TestCase):
    def pair(self, source_dim, target_dim, seed):
        rng = np.random.default_rng(seed)
        xs = rng.normal(size=(80, source_dim))
        projection = rng.normal(size=(source_dim, target_dim))
        xt = xs @ projection
        return ActivationPair(
            xs[:60],
            xt[:60],
            xs[60:],
            xt[60:],
            f"source.{seed}",
            f"target.{seed}",
        )

    def test_manifest_round_trip_and_required_splits(self):
        records = (
            CalibrationRecord("a", CalibrationSplit.MAP_FIT, "code", "h1", "repo", 0, 10),
            CalibrationRecord("b", CalibrationSplit.MAP_VALIDATION, "code", "h2", "repo", 10, 20),
            CalibrationRecord(
                "c", CalibrationSplit.TRANSFER_VALIDATION, "reasoning", "h3", "repo", 20, 30
            ),
        )
        manifest = CalibrationManifest("tokhash", records)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            manifest.write_json(path)
            loaded = CalibrationManifest.from_json(path)
        self.assertEqual(manifest.fingerprint, loaded.fingerprint)
        with self.assertRaises(CalibrationError):
            CalibrationManifest("tokhash", records[:2])

    def test_layer_correspondence_and_map_solution(self):
        rng = np.random.default_rng(4)
        base = [rng.normal(size=(40, 5)) for _ in range(4)]
        source = {i: base[i] for i in range(4)}
        target = {
            0: base[0] @ rng.normal(size=(5, 3)),
            1: base[3] @ rng.normal(size=(5, 3)),
        }
        correspondence = solve_layer_correspondence(
            source, target, measurements=64, seed=7, depth_prior=0.01
        )
        self.assertEqual(correspondence.mapping, {0: 0, 1: 3})

        calibration = PhiLayerCalibration(
            attn_in=self.pair(8, 4, 10),
            source_k_fit=rng.normal(size=(100, 4)),
            attn_out=self.pair(8, 4, 11),
            mlp_in=self.pair(8, 4, 12),
            mlp_hidden=self.pair(12, 6, 13),
            mlp_out=self.pair(8, 4, 14),
        )
        solved = solve_phi_layer_maps(
            calibration,
            target_layer=0,
            source_q_heads=4,
            source_kv_heads=2,
            target_q_heads=2,
            target_kv_heads=1,
            head_dim=2,
            ridge=1e-6,
        )
        self.assertEqual(
            set(solved.maps),
            {
                "layer.0.attn_in",
                "layer.0.q",
                "layer.0.kv",
                "layer.0.attn_out",
                "layer.0.mlp_in",
                "layer.0.mlp_hidden",
                "layer.0.mlp_out",
            },
        )
        self.assertLess(solved.maps["layer.0.attn_in"].diagnostics.validation_rmse, 1e-4)
        merged = merge_coordinate_maps(
            solved.maps,
            {"embedding": solve_activation_pair(self.pair(8, 4, 15), ridge=1e-6)},
        )
        self.assertIn("embedding", merged)
        with self.assertRaises(CalibrationError):
            merge_coordinate_maps(
                {"x": solved.maps["layer.0.attn_in"]},
                {"x": solved.maps["layer.0.attn_out"]},
            )


if __name__ == "__main__":
    unittest.main()
