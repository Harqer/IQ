"""A resumed run must use exactly the original calibration/donor configuration."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from iq_transfer.glm53_provenance import (
    GLM53ProvenanceError, enforce_run_provenance,
)


class ProvenanceTests(unittest.TestCase):
    def test_new_run_can_resume_only_unchanged_donor_and_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            calibration = root / "calibration"
            output = root / "output"
            calibration.mkdir()
            receipt = {"schema_version": 1,
                       "files": {f"artifact-{i}": "f"*64 for i in range(866)}}
            (calibration / "calibration_sha256.json").write_text(json.dumps(receipt))
            kw = dict(glm_revision="teacher-v1", mamba_revision="mamba-v1",
                      target_fingerprint="iq-config-v1")
            saved = enforce_run_provenance(output, calibration, resume=False, **kw)
            self.assertEqual(saved["schema_version"], 1)
            self.assertEqual(enforce_run_provenance(
                output, calibration, resume=True, **kw), saved)
            with self.assertRaises(GLM53ProvenanceError):
                enforce_run_provenance(
                    output, calibration, resume=True, glm_revision="teacher-v2",
                    mamba_revision="mamba-v1", target_fingerprint="iq-config-v1")
            (calibration / "calibration_sha256.json").write_text(
                json.dumps(receipt, indent=2))
            with self.assertRaises(GLM53ProvenanceError):
                enforce_run_provenance(output, calibration, resume=True, **kw)

    def test_legacy_run_without_provenance_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "calibration").mkdir()
            (root / "output").mkdir()
            (root / "calibration" / "calibration_sha256.json").write_text(
                json.dumps({"schema_version": 1,
                            "files": {f"a{i}": "f"*64 for i in range(866)}}))
            with self.assertRaisesRegex(GLM53ProvenanceError, "unsafe legacy"):
                enforce_run_provenance(
                    root / "output", root / "calibration", resume=True,
                    glm_revision="t", mamba_revision="m",
                    target_fingerprint="cfg")


if __name__ == "__main__":
    unittest.main()
