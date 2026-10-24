"""Stage-lazy calibration must reconstruct the exact original GLM WARM maps."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from iq_transfer.glm53_calibration import (
    GLM53CalibrationSolution, GLM53StageCalibration,
)
from iq_transfer.glm53_lazy_calibration import LazyGLM53Calibration
from iq_transfer.transport import CoordinateMap


class LazyCalibrationTests(unittest.TestCase):
    def test_stagewise_read_matches_eager_calibration(self):
        basis = CoordinateMap(np.eye(4), ridge=1e-8)
        latent = CoordinateMap(np.eye(4, 2), ridge=1e-8)
        stage = GLM53StageCalibration(
            stage=0, source_layer=3,
            context_physical_layer=1, moe_physical_layer=2,
            residual_map=basis, attention_input_map=basis,
            moe_input_map=basis, compressed_kv_map=latent, latent_map=latent,
            expert_usage=np.ones(256, dtype=np.float64) / 256,
        )
        source = GLM53CalibrationSolution(
            stages=(stage,), source_layers=78, target_config_fingerprint="unit",
            lexical_input_map=basis, final_output_map=basis,
            calibration_mode="weight_only",
        )
        with tempfile.TemporaryDirectory() as tmp:
            source.write(Path(tmp))
            proxy = LazyGLM53Calibration(tmp)
            self.assertEqual(len(proxy.stages), 1)
            self.assertEqual(proxy.source_layer_map, {0: 3})
            self.assertEqual(proxy.calibration_mode, "weight_only")
            loaded = proxy.stages[0]
            self.assertEqual(loaded.source_layer, 3)
            np.testing.assert_array_equal(loaded.residual_map.matrix, np.eye(4))
            np.testing.assert_array_equal(loaded.latent_map.matrix, np.eye(4, 2))
            np.testing.assert_array_equal(loaded.expert_usage, stage.expert_usage)
            self.assertIsNot(loaded, proxy.stages[0])


if __name__ == "__main__":
    unittest.main()
