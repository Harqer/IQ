from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/transfer_mamba_bucket_foundation.py"
spec = importlib.util.spec_from_file_location("iq_bucket_runner", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class IQMambaBucketFoundationTests(unittest.TestCase):
    def make_foundation(self, root: Path) -> Path:
        placements = [
            {
                "source_layer": i,
                "target_mamba_ordinal": i,
                "shard": f"mamba/layer-{i:03d}.safetensors",
            }
            for i in range(24)
        ]
        folder = root / "mamba"
        folder.mkdir()
        for p in placements:
            (root / p["shard"]).write_bytes(b"test-shard")
        (root / "model-global.safetensors").write_bytes(b"test-globals")
        manifest = {
            "artifact_type": "iq_mamba3_foundation",
            "donor": {
                "revision": runner.PINNED_REVISION,
                "checkpoint_sha256": runner.PINNED_SHA256,
            },
            "target": {
                "mamba3": {"d_model": 4096, "num_layers": 32, "mimo_rank": 4},
            },
            "placements": placements,
            "identity_mamba_ordinals": list(range(24, 32)),
        }
        (root / "mamba3_transplant.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root / "mamba3_transplant.json"

    def test_validated_foundation_creates_complete_hash_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_foundation(root)
            files = runner.validate_compilation(root)
            self.assertEqual(len(files), 26)  # 25 weight files + provenance JSON
            checksums = runner.write_checksums(root, files)
            lines = checksums.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 26)
            self.assertTrue(all(len(line.split("  ")[0]) == 64 for line in lines))

    def test_fails_closed_when_a_transplanted_layer_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_foundation(root)
            (root / "mamba/layer-000.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "shard files"):
                runner.validate_compilation(root)

    def test_fails_closed_on_donor_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.make_foundation(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["donor"]["checkpoint_sha256"] = "invalid"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.validate_compilation(root)

    def test_fails_closed_on_invalid_identity_depth_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.make_foundation(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["identity_mamba_ordinals"] = list(range(23, 31))
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage"):
                runner.validate_compilation(root)


if __name__ == "__main__":
    unittest.main()
