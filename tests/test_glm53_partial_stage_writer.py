"""No-cloud-cost regression tests for bounded missing-only WARM stage repair."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from iq_transfer.glm53_partial_repair import (
    PartialCalibrationStageWriter, _files,
)
from iq_transfer.glm53_calibration import GLM53CalibrationError, GLM53StageCalibration
from iq_transfer.transport import CoordinateMap


class StageRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        for stage in range(78):
            n = f"stage-{stage:02d}"
            self.records.append({
                "stage": stage, "source_layer": stage,
                "context_physical_layer": stage * 3 + 1,
                "moe_physical_layer": stage * 3 + 2,
                "residual_map": n + "-residual",
                "attention_input_map": n + "-attention-input",
                "moe_input_map": n + "-moe-input",
                "compressed_kv_map": n + "-kv",
                "latent_map": n + "-latent",
                "dense_intermediate_map": n + "-dense-intermediate" if stage < 3 else None,
                "expert_usage": None if stage < 3 else n + "-expert-usage.npy",
            })
        self.manifest = {
            "schema_version": 2, "source_layers": 78,
            "calibration_mode": "weight_only",
            "target_config_fingerprint": "full78-checked",
            "stages": self.records,
        }
        (self.root / "glm53_calibration.json").write_text(json.dumps(self.manifest))
        self.target = SimpleNamespace(fingerprint="full78-checked")
        # Already-complete fixture files are deliberately tiny and are never
        # loaded by the stage writer (only their existence is inspected).
        for record in self.records[1:]:
            for name in _files(record):
                (self.root / name).write_bytes(b"persisted")

    def _writer(self):
        with patch("iq_transfer.glm53_partial_repair.donor_layer_positions",
                   return_value=tuple((3*i + 1, 3*i + 2) for i in range(78))):
            return PartialCalibrationStageWriter(self.root, target_config=self.target)

    def test_only_missing_stage_written_and_existing_bytes_unchanged(self):
        writer = self._writer()
        self.assertEqual(writer.missing_stages, [0])
        self.assertEqual(len(writer.skip_stages), 77)
        basis = CoordinateMap(np.eye(2), 1e-6)
        new_stage = GLM53StageCalibration(
            stage=0, source_layer=0,
            context_physical_layer=1, moe_physical_layer=2,
            residual_map=basis, attention_input_map=basis,
            moe_input_map=basis, compressed_kv_map=basis, latent_map=basis,
            dense_intermediate_map=basis,
        )
        writer.write_stage(new_stage)
        self.assertEqual(writer.missing_stages, [])
        for name in _files(self.records[0]):
            self.assertTrue((self.root / name).is_file())
            self.assertGreater((self.root / name).stat().st_size, 0)
        self.assertEqual((self.root / "stage-01-kv.json").read_bytes(), b"persisted")
        self.assertFalse(list(self.root.glob(".warm-stage-*.partial")))

    def test_partial_stage_files_refuse_unsafe_overwrite(self):
        (self.root / "stage-00-residual.json").write_bytes(b"partial")
        with self.assertRaisesRegex(GLM53CalibrationError, "partially published"):
            self._writer()

    def test_mismatched_fingerprint_refused(self):
        self.target.fingerprint = "changed-model"
        with self.assertRaisesRegex(GLM53CalibrationError, "mismatches"):
            self._writer()


if __name__ == "__main__":
    unittest.main()
