"""Verify true missing-only CUDA work planning without 16-GiB redundant hash reads."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from iq_transfer.glm53_partial_repair import _files
from scripts.select_glm53_missing_gpu_stages import (
    ORIGINAL_FINGERPRINT, TARGET_FINGERPRINT, choose_missing_stages,
)

SOURCES = [0, 3, 6, 10, 14, 17, 22, 23, 26, 30, 34, 37,
           42, 44, 46, 50, 54, 57, 62, 64, 66, 70, 74, 77]


class GpuStageSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.original = root / "original"
        self.public = root / "public"
        self.checkpoint = root / "checkpoint"
        for p in (self.original, self.public, self.checkpoint):
            p.mkdir()
        original_json = json.dumps({
            "schema_version": 2, "source_layers": 78,
            "target_config_fingerprint": ORIGINAL_FINGERPRINT,
            "stages": [{"source_layer": x} for x in SOURCES],
        }).encode()
        (self.original / "glm53_calibration.json").write_bytes(original_json)
        (self.original / ".calibration_integrity.json").write_text(json.dumps({
            "donor_revision": "abf42e6c0e53bc29a875511301e7b7fcd220315b",
            "files": {
                "glm53_calibration.json": hashlib.sha256(original_json).hexdigest(),
            },
        }))
        stages = []
        for n in range(78):
            base = f"stage-{n:02d}"
            stages.append({
                "stage": n, "source_layer": n,
                "residual_map": base + "-residual",
                "attention_input_map": base + "-attention-input",
                "moe_input_map": base + "-moe-input",
                "compressed_kv_map": base + "-kv",
                "latent_map": base + "-latent",
                "dense_intermediate_map": base + "-dense-intermediate" if n < 3 else None,
                "expert_usage": base + "-expert-usage.npy" if n >= 3 else None,
            })
        self.stages = stages
        (self.public / "glm53_calibration.json").write_text(json.dumps({
            "schema_version": 2, "source_layers": 78,
            "calibration_mode": "weight_only",
            "target_config_fingerprint": TARGET_FINGERPRINT, "stages": stages,
        }))
        for stage in stages[50:]:
            for filename in _files(stage):
                (self.public / filename).write_bytes(b"saved-late-stage")

    def test_missing_four_stages_excludes_legacy_and_late(self):
        self.assertEqual(
            choose_missing_stages(
                self.original, self.public, self.checkpoint, 4),
            [1, 2, 4, 5],
        )

    def test_already_saved_sha256_receipt_excludes_completed_stage(self):
        p = self.public / "recovery-stage-uploads" / "stage-01"
        p.mkdir(parents=True)
        names = _files(self.stages[1])
        for name in names:
            (p / name).write_bytes(b"saved-new-stage")
        (p / "receipt.json").write_text(json.dumps({
            "schema_version": 1, "source_layer": 1,
            "donor_revision": "abf42e6c0e53bc29a875511301e7b7fcd220315b",
            "target_fingerprint": TARGET_FINGERPRINT,
            "sha256": {name: "a" * 64 for name in names},
        }))
        self.assertEqual(
            choose_missing_stages(
                self.original, self.public, self.checkpoint, 3),
            [2, 4, 5],
        )

    def test_original_provenance_tamper_fails_closed(self):
        (self.original / "glm53_calibration.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            choose_missing_stages(
                self.original, self.public, self.checkpoint, 4)

    def test_late_saved_stage_incomplete_blocks_selection(self):
        (self.public / "stage-50-latent.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            choose_missing_stages(
                self.original, self.public, self.checkpoint, 4)

    def test_uncommitted_partial_recovery_blocks_selection(self):
        (self.public / "recovery-stage-uploads" / "stage-01").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "incomplete/unsafe"):
            choose_missing_stages(
                self.original, self.public, self.checkpoint, 4)


if __name__ == "__main__":
    unittest.main()
