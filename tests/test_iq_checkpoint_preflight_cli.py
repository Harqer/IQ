"""No-GPU, no-network CLI checks before any paid IQ inference job."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import check_iq_checkpoint_files


class CheckpointGateCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"model": {"vocab_size": 128256}, "schedule": {"layers": ["mamba3"]}}
        self.index = {"metadata": {"total_size": 4}, "weight_map": {"x": "model-000.safetensors"}}
        self.manifest = {"schema_version": 2, "method": "mamba3_foundation_plus_glm53_capability_weight_transfer"}
        (self.root / "model-000.safetensors").write_bytes(b"1234")
        (self.root / "tokenizer.json").write_text("{}")
        self.write()

    def write(self):
        (self.root / "iq_config.json").write_text(json.dumps(self.config))
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))
        (self.root / "transplant_manifest.json").write_text(json.dumps(self.manifest))

    def run_gate(self):
        with patch.object(sys, "argv", ["check_iq_checkpoint_files.py", str(self.root)]):
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                return check_iq_checkpoint_files.main()

    def test_complete_metadata_passes_without_gpu(self):
        self.assertEqual(self.run_gate(), 0)

    def test_missing_manifest_blocks_before_job(self):
        (self.root / "transplant_manifest.json").unlink()
        with self.assertRaises(SystemExit):
            self.run_gate()

    def test_unknown_foundation_blocks_before_job(self):
        self.manifest["method"] = "unknown_new_foundation"
        self.write()
        with self.assertRaises(SystemExit):
            self.run_gate()

    def test_symlink_and_size_mismatch_block_before_job(self):
        (self.root / "indirect.safetensors").symlink_to(
            self.root / "model-000.safetensors"
        )
        self.index["weight_map"]["x"] = "indirect.safetensors"
        self.write()
        with self.assertRaises(SystemExit):
            self.run_gate()
        self.index["weight_map"]["x"] = "model-000.safetensors"
        self.index["metadata"]["total_size"] = 999
        self.write()
        with self.assertRaises(SystemExit):
            self.run_gate()


if __name__ == "__main__":
    unittest.main()
