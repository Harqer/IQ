"""Prevent costly H200 restarts caused by accidentally duplicated installer tails.

No CUDA or pip installation is performed; this is a fast CPU-only source test.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_model_runtime.sh"
WRAPPER = ROOT / "scripts" / "hf_container_venv.sh"
EXPECTED_LAST_LINE = (
    "# The HF launcher must proceed to full IQ checkpoint load and text generation."
)


class H200InstallerIntegrityTests(unittest.TestCase):
    def test_bash_syntax(self):
        if not shutil.which("bash"):
            self.skipTest("bash is unavailable")
        for script in (INSTALLER, WRAPPER):
            with self.subTest(script=script.name):
                subprocess.run(
                    ["bash", "-n", str(script)],
                    check=True, capture_output=True, text=True,
                )

    def test_installer_has_single_complete_dependency_pipeline(self):
        text = INSTALLER.read_text(encoding="utf-8")
        counts = (
            'python scripts/patch_iq_mamba3_dependencies.py "$source_dir"',
            'python scripts/verify_iq_mamba_patch_paths.py "$source_dir"',
            'MAMBA_FORCE_BUILD=TRUE python -m pip install',
            'python scripts/verify_iq_dependencies.py',
            'python scripts/hf_runtime_contract.py after --snapshot "$snapshot"',
        )
        for snippet in counts:
            with self.subTest(step=snippet):
                self.assertEqual(
                    text.count(snippet), 1,
                    f"Installer step must occur once: {snippet}",
                )
        self.assertEqual(text.strip().splitlines()[-1], EXPECTED_LAST_LINE)
        self.assertEqual(text.count(EXPECTED_LAST_LINE), 1)

    def test_no_orphaned_kernel_source_line_or_malformed_shell_fragment(self):
        text = INSTALLER.read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(
                r"^mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd[.]py",
                text, re.MULTILINE,
            )
        )
        self.assertNotIn("npyproject.tomlnsetup.py", text)
        self.assertNotIn('echo >&2 "Unexpected Mamba3 source edits: $changed_files"', text)
        self.assertIn('python scripts/verify_iq_mamba_patch_paths.py "$source_dir"', text)


if __name__ == "__main__":
    unittest.main()
