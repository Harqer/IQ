"""Offline safeguards for pinned Mamba3/CUTLASS/TileLang runtime compatibility."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.patch_iq_mamba3_dependencies import (
    IQ_TILELANG, IQ_TVM, UPSTREAM_TILELANG, UPSTREAM_TVM,
    apply_patch, apply_kernel_tail_guard, record_installed,
    KERNEL_RELATIVE_PATH, UPSTREAM_V_READ, SAFE_V_READ, KERNEL_PATCH,
)
from iq_model.state import mamba3


class DummyDistribution:
    def __init__(self, root: Path):
        self.root = root

    def locate_file(self, path: str) -> Path:
        return self.root / path

    def read_text(self, name: str):
        return '{"url": "file:///tmp/verified-local-checkout", "dir_info": {}}'


class Mamba3CompatibilityTests(unittest.TestCase):
    def test_dependency_metadata_patch_only_expected_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = ("[project]\ndependencies = [\n    " +
                     UPSTREAM_TILELANG + ",\n    " + UPSTREAM_TVM + ",\n]\n")
            second = ("install_requires=[\n        " + UPSTREAM_TILELANG +
                      ",\n        " + UPSTREAM_TVM + ",\n]\n")
            (root / "pyproject.toml").write_text(first)
            (root / "setup.py").write_text(second)
            result = apply_patch(root)
            self.assertEqual(result, ["pyproject.toml", "setup.py"])
            self.assertEqual(
                (root / "pyproject.toml").read_text(),
                first.replace(UPSTREAM_TILELANG, IQ_TILELANG)
                     .replace(UPSTREAM_TVM, IQ_TVM),
            )
            self.assertEqual(
                (root / "setup.py").read_text(),
                second.replace(UPSTREAM_TILELANG, IQ_TILELANG)
                      .replace(UPSTREAM_TVM, IQ_TVM),
            )

    def test_source_drift_does_not_partially_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial = UPSTREAM_TILELANG + "\n" + UPSTREAM_TVM
            (root / "pyproject.toml").write_text(initial)
            (root / "setup.py").write_text('unknown tilelang requirement')
            with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
                apply_patch(root)
            self.assertEqual((root / "pyproject.toml").read_text(), initial)

    def test_native_mimo_short_tail_read_guard_exact_source_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kernel = root / KERNEL_RELATIVE_PATH
            kernel.parent.mkdir(parents=True)
            upstream = (
                "before\\n" + UPSTREAM_V_READ + "\\n"
                "after (all other MIMO logic preserved)\\n"
            )
            kernel.write_text(upstream, encoding="utf-8")
            result = apply_kernel_tail_guard(root)
            self.assertEqual(result, KERNEL_RELATIVE_PATH)
            patched = kernel.read_text(encoding="utf-8")
            self.assertEqual(
                patched, upstream.replace(UPSTREAM_V_READ, SAFE_V_READ)
            )
            self.assertIn("if chunk_start + cs < S:", patched)
            self.assertIn("v_shared[cs, p] = 0.0", patched)
            self.assertNotIn(UPSTREAM_V_READ, patched)
            with self.assertRaisesRegex(RuntimeError, "Unexpected pinned"):
                apply_kernel_tail_guard(root)

    def test_kernel_patch_refuses_unrecognized_upstream_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kernel = root / KERNEL_RELATIVE_PATH
            kernel.parent.mkdir(parents=True)
            kernel.write_text("unrecognized kernel", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unexpected pinned"):
                apply_kernel_tail_guard(root)
            self.assertEqual(kernel.read_text(), "unrecognized kernel")

    def test_installed_source_provenance_fails_on_mutation(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as install_dir:
            source = Path(source_dir)
            install = Path(install_dir)
            source_file = source / "mamba_ssm/modules/mamba3.py"
            installed_file = install / "mamba_ssm/modules/mamba3.py"
            source_file.parent.mkdir(parents=True)
            installed_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"exact original MIMO source")
            installed_file.write_bytes(source_file.read_bytes())
            kernel_source = source / KERNEL_RELATIVE_PATH
            kernel_installed = install / KERNEL_RELATIVE_PATH
            kernel_source.parent.mkdir(parents=True)
            kernel_installed.parent.mkdir(parents=True)
            kernel_source.write_text("start\\n" + SAFE_V_READ + "\\nend\\n", encoding="utf-8")
            kernel_installed.write_bytes(kernel_source.read_bytes())
            commit = "a" * 40
            with patch("scripts.patch_iq_mamba3_dependencies.metadata.distribution",
                       return_value=DummyDistribution(install)):
                marker = record_installed(source, commit)
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["source_commit"], commit)
            self.assertEqual(payload["kernel_patch"], KERNEL_PATCH)
            self.assertEqual(
                payload["kernel_sha256"],
                sha256(kernel_installed.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                payload["module_sha256"], sha256(installed_file.read_bytes()).hexdigest()
            )
            with patch("iq_model.state.mamba3.metadata.distribution",
                       return_value=DummyDistribution(install)):
                self.assertEqual(mamba3._installed_source_commit(), commit)
                kernel_installed.write_bytes(b"corrupted patched kernel")
                self.assertIsNone(mamba3._installed_source_commit())
                kernel_installed.write_bytes(kernel_source.read_bytes())
                self.assertEqual(mamba3._installed_source_commit(), commit)
                installed_file.write_bytes(b"unexpected change")
                self.assertIsNone(mamba3._installed_source_commit())

    def test_mismatched_installed_source_never_records_success(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as install_dir:
            source = Path(source_dir)
            install = Path(install_dir)
            a = source / "mamba_ssm/modules/mamba3.py"
            b = install / "mamba_ssm/modules/mamba3.py"
            a.parent.mkdir(parents=True)
            b.parent.mkdir(parents=True)
            a.write_text("original")
            b.write_text("changed")
            with patch("scripts.patch_iq_mamba3_dependencies.metadata.distribution",
                       return_value=DummyDistribution(install)):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    record_installed(source, "a" * 40)
            self.assertFalse((install / "mamba_ssm/iq_mamba3_provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
