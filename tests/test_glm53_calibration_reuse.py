"""CPU-only safety checks for reusing existing GLM WARM maps without recomputation."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from iq_transfer.complete_transplant import canonical_glm53_config, donor_layer_positions
from scripts.reuse_glm53_calibration import CalibrationReuseError, prepare_reuse


SOURCES = [
    0, 3, 6, 10, 14, 17, 22, 23, 26, 30, 34, 37, 42,
    44, 46, 50, 54, 57, 62, 64, 66, 70, 74, 77,
]


def fixture(path: Path) -> Path:
    path.mkdir()
    file_bytes: dict[str, bytes] = {}

    def add(name: str, data: bytes = b"test only, not a coordinate map") -> None:
        (path / name).write_bytes(data)
        file_bytes[name] = data

    stages = []
    config = canonical_glm53_config()
    for index, ((context, moe), source) in enumerate(
        zip(donor_layer_positions(config), SOURCES)
    ):
        prefix = f"stage-{index:02d}"
        record = {
            "stage": index,
            "source_layer": source,
            "context_physical_layer": context,
            "moe_physical_layer": moe,
            "expert_usage": f"{prefix}-expert-usage.npy" if source >= 3 else None,
            "dense_intermediate_map": f"{prefix}-dense-intermediate" if source < 3 else None,
        }
        for key, suffix in (
            ("residual_map", "residual"),
            ("attention_input_map", "attention-input"),
            ("moe_input_map", "moe-input"),
            ("compressed_kv_map", "kv"),
            ("latent_map", "latent"),
        ):
            record[key] = f"{prefix}-{suffix}"
        for key in (
            "residual_map", "attention_input_map", "moe_input_map",
            "compressed_kv_map", "latent_map", "dense_intermediate_map",
        ):
            if record[key] is not None:
                add(record[key] + ".json")
                add(record[key] + ".safetensors")
        if record["expert_usage"]:
            add(record["expert_usage"])
        stages.append(record)
    for name in ("lexical-input", "final-output"):
        add(name + ".json")
        add(name + ".safetensors")
    manifest = {
        "schema_version": 2,
        "source_layers": 78,
        "calibration_mode": "weight_only",
        "target_config_fingerprint": config.fingerprint,
        "lexical_input_map": "lexical-input",
        "final_output_map": "final-output",
        "stages": stages,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    add("glm53_calibration.json", manifest_bytes)
    integrity = {
        "donor_revision": "abf42e6c0e53bc29a875511301e7b7fcd220315b",
        "files": {k: sha256(v).hexdigest() for k, v in file_bytes.items()},
    }
    (path / ".calibration_integrity.json").write_text(json.dumps(integrity))
    return path


class TestReuseCalibration(unittest.TestCase):
    def test_maps_reused_for_new_fingerprint_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = fixture(root / "original")
            out = root / "candidate"
            source_bytes = (src / "glm53_calibration.json").read_bytes()
            result = prepare_reuse(src, out, routed_experts=64, verify_hashes=True)
            self.assertTrue(result["original_file_hashes_verified"])
            self.assertTrue((out / "stage-02-kv.safetensors").is_symlink())
            self.assertEqual((src / "glm53_calibration.json").read_bytes(), source_bytes)
            meta = json.loads((out / "glm53_calibration.json").read_text())
            self.assertEqual(meta["target_config_fingerprint"],
                             canonical_glm53_config(routed_experts=64).fingerprint)
            self.assertEqual(meta["calibration_mode"], "weight_only")
            self.assertFalse(result["activation_refinement_performed"])
            self.assertTrue(result["not_self_contained"])

    def test_missing_or_tampered_original_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = fixture(root / "original")
            (src / "stage-04-residual.safetensors").write_bytes(b"tamper")
            with self.assertRaisesRegex(CalibrationReuseError, "integrity mismatch"):
                prepare_reuse(src, root / "new", routed_experts=64, verify_hashes=True)
            self.assertFalse((root / "new").exists())

    def test_reject_incorrect_old_fingerprint_and_output_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = fixture(root / "original")
            (root / "new").mkdir()
            with self.assertRaisesRegex(CalibrationReuseError, "already exists"):
                prepare_reuse(src, root / "new", routed_experts=128)
            bad = json.loads((src / "glm53_calibration.json").read_text())
            bad["target_config_fingerprint"] = "not-original"
            (src / "glm53_calibration.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(CalibrationReuseError, "canonical 32-expert"):
                prepare_reuse(src, root / "another", routed_experts=128)


if __name__ == "__main__":
    unittest.main()
