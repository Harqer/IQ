"""Fail-closed bucket-staged WARM recovery: receipts last, no overwrites."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from iq_transfer.glm53_calibration import GLM53CalibrationError, GLM53StageCalibration
from iq_transfer.glm53_partial_repair import PartialCalibrationStageWriter, _files
from iq_transfer.glm53_partial_recovery_receipts import (
    PINNED_DONOR_REVISION, rehydrate_committed_stages, sha256_digest,
)
from iq_transfer.transport import CoordinateMap


class BucketStageCommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.local = root / "computed"
        self.previous = root / "workspace"
        self.bucket = root / "bucket"
        for p in (self.local, self.previous, self.bucket):
            p.mkdir()
        stages = []
        for i in range(78):
            s = {
                "stage": i,
                "source_layer": i,
                "context_physical_layer": 3 * i + 1,
                "moe_physical_layer": 3 * i + 2,
                "residual_map": f"stage-{i:02d}-residual",
                "attention_input_map": f"stage-{i:02d}-attention-input",
                "moe_input_map": f"stage-{i:02d}-moe-input",
                "compressed_kv_map": f"stage-{i:02d}-kv",
                "latent_map": f"stage-{i:02d}-latent",
                "dense_intermediate_map": f"stage-{i:02d}-dense-intermediate" if i < 3 else None,
                "expert_usage": f"stage-{i:02d}-expert-usage.npy" if i >= 3 else None,
            }
            stages.append(s)
        self.manifest = {
            "schema_version": 2, "source_layers": 78,
            "calibration_mode": "weight_only",
            "target_config_fingerprint": "verified-target",
            "stages": stages,
        }
        self.config = SimpleNamespace(fingerprint="verified-target")
        for p in (self.local, self.previous):
            (p / "glm53_calibration.json").write_text(json.dumps(self.manifest))
        for stage in stages[1:]:
            for name in _files(stage):
                (self.local / name).write_bytes(b"already-complete")

    def _writer(self):
        self.bucket.mkdir(exist_ok=True)
        with patch(
            "iq_transfer.glm53_partial_repair.donor_layer_positions",
            return_value=tuple((3*i + 1, 3*i + 2) for i in range(78)),
        ):
            return PartialCalibrationStageWriter(
                self.local, target_config=self.config,
                publish_directory=self.bucket / "recovery-stage-uploads",
            )

    def _dense_stage_zero(self):
        basis = CoordinateMap(np.eye(2), 1e-6)
        return GLM53StageCalibration(
            stage=0, source_layer=0,
            context_physical_layer=1, moe_physical_layer=2,
            residual_map=basis, attention_input_map=basis,
            moe_input_map=basis, compressed_kv_map=basis,
            latent_map=basis, dense_intermediate_map=basis,
        )

    def test_stage_publishes_sha256_receipt_last_and_rehydrates(self):
        writer = self._writer()
        self.assertEqual(writer.missing_stages, [0])
        writer.write_stage(self._dense_stage_zero())
        source = self.bucket / "recovery-stage-uploads" / "stage-00"
        receipt = json.loads((source / "receipt.json").read_text())
        self.assertEqual(receipt["source_layer"], 0)
        self.assertEqual(receipt["donor_revision"], PINNED_DONOR_REVISION)
        self.assertEqual(set(receipt["sha256"]), _files(self.manifest["stages"][0]))
        for filename, digest in receipt["sha256"].items():
            self.assertEqual(digest, sha256_digest(source / filename))
        result = rehydrate_committed_stages(self.previous, self.bucket)
        self.assertEqual(result, [0])
        for name in receipt["sha256"]:
            self.assertTrue((self.previous / name).is_symlink())
        self.assertFalse((self.previous / "stage-01-residual.json").exists())

    def test_tampered_remote_group_blocks_restore(self):
        writer = self._writer()
        writer.write_stage(self._dense_stage_zero())
        dest = self.bucket / "recovery-stage-uploads" / "stage-00"
        (dest / "stage-00-residual.json").write_bytes(b"altered")
        with self.assertRaisesRegex(GLM53CalibrationError, "SHA mismatch"):
            rehydrate_committed_stages(self.previous, self.bucket)

    def test_uncommitted_partial_remote_group_blocks_restore_and_overwrite(self):
        source = self.bucket / "recovery-stage-uploads" / "stage-00"
        source.mkdir(parents=True)
        (source / "stage-00-residual.json").write_bytes(b"partial")
        with self.assertRaisesRegex(GLM53CalibrationError, "uncommitted"):
            rehydrate_committed_stages(self.previous, self.bucket)
        writer = self._writer()
        with self.assertRaisesRegex(GLM53CalibrationError, "already has remote"):
            writer.write_stage(self._dense_stage_zero())


if __name__ == "__main__":
    unittest.main()
