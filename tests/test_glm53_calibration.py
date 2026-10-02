from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from iq_transfer import (
    CoordinateMap,
    GLM53CalibrationError,
    GLM53CalibrationSolution,
    GLM53StageCalibration,
)


class GLM53CalibrationTests(unittest.TestCase):
    def stage(self, *, sparse: bool) -> GLM53StageCalibration:
        return GLM53StageCalibration(
            stage=0,
            source_layer=3,
            context_physical_layer=1,
            moe_physical_layer=2,
            residual_map=CoordinateMap(np.eye(4), 1e-6, "glm.resid", "iq.resid"),
            compressed_kv_map=CoordinateMap(np.eye(3), 1e-6, "glm.kv", "iq.kv"),
            latent_map=CoordinateMap(np.eye(4, 2), 1e-6, "glm.resid", "iq.latent"),
            expert_usage=np.array([0.1, 0.2, 0.3, 0.4]) if sparse else None,
            dense_intermediate_map=(
                None
                if sparse
                else CoordinateMap(np.eye(5, 2), 1e-6, "glm.mlp", "iq.mlp")
            ),
        )

    def test_stage_requires_exactly_one_dense_or_sparse_payload(self):
        with self.assertRaisesRegex(GLM53CalibrationError, "exactly one"):
            GLM53StageCalibration(
                stage=0,
                source_layer=0,
                context_physical_layer=1,
                moe_physical_layer=2,
                residual_map=CoordinateMap(np.eye(2), 1e-6),
                compressed_kv_map=CoordinateMap(np.eye(2), 1e-6),
                latent_map=CoordinateMap(np.eye(2), 1e-6),
            )

    def test_solution_round_trip_persists_maps_and_usage(self):
        sparse = self.stage(sparse=True)
        dense = GLM53StageCalibration(
            stage=1,
            source_layer=7,
            context_physical_layer=4,
            moe_physical_layer=5,
            residual_map=CoordinateMap(np.eye(4), 1e-6, "glm.r1", "iq.r1"),
            compressed_kv_map=CoordinateMap(np.eye(3), 1e-6, "glm.k1", "iq.k1"),
            latent_map=CoordinateMap(np.eye(4, 2), 1e-6, "glm.r1", "iq.l1"),
            dense_intermediate_map=CoordinateMap(
                np.eye(5, 2), 1e-6, "glm.m1", "iq.m1"
            ),
        )
        solution = GLM53CalibrationSolution(
            stages=(sparse, dense),
            source_layers=78,
            target_config_fingerprint="cfg",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = solution.write(tmp)
            self.assertTrue(manifest.is_file())
            loaded = GLM53CalibrationSolution.load(tmp)
        self.assertEqual(loaded.source_layer_map, {0: 3, 1: 7})
        self.assertTrue(np.array_equal(
            loaded.stages[0].expert_usage,
            sparse.expert_usage,
        ))
        self.assertTrue(np.array_equal(
            loaded.stages[1].dense_intermediate_map.matrix,
            dense.dense_intermediate_map.matrix,
        ))

    def test_solution_rejects_nonmonotonic_source_assignment(self):
        a = self.stage(sparse=True)
        b = GLM53StageCalibration(
            stage=1,
            source_layer=2,
            context_physical_layer=4,
            moe_physical_layer=5,
            residual_map=CoordinateMap(np.eye(4), 1e-6),
            compressed_kv_map=CoordinateMap(np.eye(3), 1e-6),
            latent_map=CoordinateMap(np.eye(4, 2), 1e-6),
            expert_usage=np.ones(4) / 4,
        )
        with self.assertRaisesRegex(GLM53CalibrationError, "monotonic"):
            GLM53CalibrationSolution(
                stages=(a, b),
                source_layers=78,
                target_config_fingerprint="cfg",
            )


if __name__ == "__main__":
    unittest.main()
