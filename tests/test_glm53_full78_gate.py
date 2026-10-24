"""Offline zero-cloud-cost tests for full78 stage and durability guards."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.glm53_full78_gate import (
    GateFailure, check_calibration, check_capacity,
    check_checkpoint_stage, expected_calibration_files,
    expected_stage_shards,
)


def _manifest():
    stages = []
    for i in range(78):
        prefix = f"stage-{i:02d}-"
        stages.append({
            "stage": i, "source_layer": i,
            "residual_map": prefix + "residual",
            "attention_input_map": prefix + "attention-input",
            "moe_input_map": prefix + "moe-input",
            "compressed_kv_map": prefix + "kv",
            "latent_map": prefix + "latent",
            "dense_intermediate_map": (prefix + "dense-intermediate") if i < 3 else None,
            "expert_usage": None if i < 3 else prefix + "expert-usage.npy",
        })
    return {"schema_version": 2, "source_layers": 78,
            "stages": stages,
            "lexical_input_map": "lexical-input",
            "final_output_map": "final-output"}


class Full78GateTests(unittest.TestCase):
    def test_real_manifest_footprint_is_866_files(self):
        expected = expected_calibration_files(_manifest())
        self.assertEqual(len(expected), 866)
        self.assertIn("stage-03-expert-usage.npy", expected)
        self.assertIn("stage-77-residual.safetensors", expected)

    def test_missing_stage_zero_to_49_fails_even_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest()
            (root / "glm53_calibration.json").write_text(json.dumps(manifest))
            for name in expected_calibration_files(manifest):
                if name.startswith("stage-") and int(name.split("-")[1]) < 50:
                    continue
                if name.endswith(".safetensors"):
                    (root / name).write_bytes(b"x" * 32)
                elif name != "glm53_calibration.json":
                    (root / name).write_bytes(b"x")
            status = check_calibration(root)
            self.assertFalse(status["ready"])
            self.assertGreater(status["missing_count"], 500)
            self.assertIn("stage-03-expert-usage.npy", status["missing_examples"] + [
                k for k in expected_calibration_files(manifest) if not (root / k).exists()
            ])

    def test_single_stage_requires_every_expert_and_projection(self):
        files = expected_stage_shards(3)
        self.assertEqual(len(files), 260)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in files:
                if "expert-040" not in name:
                    (root / name).write_bytes(b"complete-data" * 3)
            status = check_checkpoint_stage(root, 3)
            self.assertEqual(status["missing_count"], 1)
            self.assertFalse(status["ready"])

    def test_manifest_rejects_wrong_layer_or_early_routing(self):
        manifest = _manifest()
        manifest["stages"][77]["source_layer"] = 76
        with self.assertRaises(GateFailure):
            expected_calibration_files(manifest)
        manifest = _manifest()
        manifest["stages"][1]["expert_usage"] = "stage-01-expert-usage.npy"
        with self.assertRaises(GateFailure):
            expected_calibration_files(manifest)

    def test_capacity_provenance_freshness_and_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "attestation.json"
            p.write_text(json.dumps({
                "bucket": "kofecheckmate/iq-glm53-all256",
                "source": "billing-console-verification",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "confirmed_free_bytes": 900_000_000_000,
            }))
            self.assertTrue(check_capacity(p, 700_000_000_000,
                                            "kofecheckmate/iq-glm53-all256")["ready"])
            with self.assertRaises(GateFailure):
                check_capacity(p, 950_000_000_000,
                               "kofecheckmate/iq-glm53-all256")
            with self.assertRaises(GateFailure):
                check_capacity(p, 700_000_000_000,
                               "different/private-bucket")


if __name__ == "__main__":
    unittest.main()
