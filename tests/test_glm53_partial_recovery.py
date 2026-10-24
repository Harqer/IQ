"""Read-only inventory and hash-protected 24->78 calibration recovery tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.reconcile_glm53_partial import (
    DONOR_REVISION, RecoveryError, inspect, restore_original_maps, stage_artifacts, link_overlay,
)


class PartialRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root / "old24"
        self.remote = self.root / "existing78"
        self.stage = self.root / "staging"
        self.checkpoint = self.root / "checkpoint"
        for p in (self.old, self.remote, self.stage, self.checkpoint):
            p.mkdir()
        old_source = [0, 3, 6, 10, 14, 17, 22, 23, 26, 30, 34, 37,
                      42, 44, 46, 50, 54, 57, 62, 64, 66, 70, 74, 77]

        def record(stage, donor):
            n = f"stage-{stage:02d}"
            return {
                "stage": stage, "source_layer": donor,
                "context_physical_layer": 3 * stage + 1,
                "moe_physical_layer": 3 * stage + 2,
                "residual_map": n + "-residual",
                "attention_input_map": n + "-attention-input",
                "moe_input_map": n + "-moe-input",
                "compressed_kv_map": n + "-kv",
                "latent_map": n + "-latent",
                "dense_intermediate_map": n + "-dense-intermediate" if donor < 3 else None,
                "expert_usage": n + "-expert-usage.npy" if donor >= 3 else None,
            }

        old_records = [record(i, src) for i, src in enumerate(old_source)]
        full_records = [record(i, i) for i in range(78)]
        self.old_manifest = {
            "schema_version": 2, "source_layers": 78,
            "calibration_mode": "weight_only",
            "target_config_fingerprint": "original-fingerprint",
            "stages": old_records,
        }
        self.full_manifest = {
            "schema_version": 2, "source_layers": 78,
            "calibration_mode": "weight_only",
            "target_config_fingerprint": "full78-fingerprint",
            "stages": full_records,
        }
        old_mf = self.old / "glm53_calibration.json"
        old_mf.write_text(json.dumps(self.old_manifest))
        data_hashes = {
            "glm53_calibration.json": hashlib.sha256(old_mf.read_bytes()).hexdigest()
        }
        for record in old_records:
            for name, role in stage_artifacts(record).items():
                val = f"donor={record['source_layer']} role={role}\n".encode()
                (self.old / name).write_bytes(val)
                data_hashes[name] = hashlib.sha256(val).hexdigest()
        (self.old / ".calibration_integrity.json").write_text(json.dumps({
            "donor_revision": DONOR_REVISION,
            "files": data_hashes,
        }))
        (self.remote / "glm53_calibration.json").write_text(
            json.dumps(self.full_manifest)
        )
        (self.stage / "glm53_calibration.json").write_text(
            json.dumps(self.full_manifest)
        )
        for record in full_records[50:]:
            for name, role in stage_artifacts(record).items():
                (self.remote / name).write_bytes(
                    f"preexisting {record['stage']} {role}".encode()
                )
        for name in ("lexical-input.json", "lexical-input.safetensors",
                     "final-output.json", "final-output.safetensors"):
            (self.remote / name).write_bytes(b"existing read-only global map")

    def _inspect(self):
        return inspect(
            self.stage, self.old, self.checkpoint,
            expected_old_fingerprint="original-fingerprint",
            existing_calibration=self.remote,
        )

    def test_exact_15_old_layers_and_35_new_layers(self):
        report = self._inspect()
        r = report["calibration"]
        self.assertEqual(r["present_stages"], list(range(50, 78)))
        self.assertEqual(
            r["old24_sha256_reusable_stages"],
            [0, 3, 6, 10, 14, 17, 22, 23, 26, 30, 34, 37, 42, 44, 46],
        )
        self.assertEqual(len(r["must_compute_stages"]), 35)
        self.assertEqual(r["old24_reusable_files"], 166)
        self.assertEqual(r["missing_files_before"], 553)

    def test_restoring_verified_old_maps_preserves_all_existing_files(self):
        report = self._inspect()
        count = restore_original_maps(self.stage, self.old, report)
        self.assertEqual(count, 166)
        self.assertFalse((self.stage / "stage-50-kv.json").exists())
        self.assertTrue((self.remote / "stage-50-kv.json").exists())
        updated = self._inspect()["calibration"]
        self.assertEqual(len(updated["must_compute_stages"]), 35)
        self.assertEqual(updated["old24_sha256_reusable_stages"], [])

    def test_verified_zero_copy_overlay_does_not_modify_original_buckets(self):
        report = self._inspect()
        remote_hashes = {
            x.name: hashlib.sha256(x.read_bytes()).hexdigest()
            for x in self.remote.iterdir() if x.is_file()
        }
        linked = link_overlay(self.stage, self.old, self.remote, report)
        self.assertEqual(linked["hash_verified_original_links"], 166)
        self.assertEqual(linked["existing_unverified_candidate_links"], 312)
        self.assertTrue((self.stage / "stage-03-kv.safetensors").is_symlink())
        self.assertTrue((self.stage / "stage-50-kv.safetensors").is_symlink())
        self.assertFalse((self.stage / "stage-02-kv.safetensors").exists())
        self.assertEqual(len(self._inspect()["calibration"]["must_compute_stages"]), 35)
        self.assertEqual(remote_hashes, {
            x.name: hashlib.sha256(x.read_bytes()).hexdigest()
            for x in self.remote.iterdir() if x.is_file()
        })

    def test_tampered_original_hash_blocks_copy(self):
        report = self._inspect()
        (self.old / "stage-00-residual.safetensors").write_bytes(b"TAMPER")
        with self.assertRaisesRegex(RecoveryError, "integrity"):
            restore_original_maps(self.stage, self.old, report)

    def test_incomplete_existing_stage_fails_closed(self):
        (self.remote / "stage-52-latent.safetensors").unlink()
        with self.assertRaisesRegex(RecoveryError, "partially written"):
            self._inspect()

    def test_original_donor_revision_must_match(self):
        file = self.old / ".calibration_integrity.json"
        j = json.loads(file.read_text())
        j["donor_revision"] = "wrong"
        file.write_text(json.dumps(j))
        with self.assertRaisesRegex(RecoveryError, "donor revision"):
            self._inspect()


if __name__ == "__main__":
    unittest.main()
