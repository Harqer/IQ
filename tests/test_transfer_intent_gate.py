"""Ensure Mamba-lexical transfer cannot start without explicit intent.

No credentials, downloads, GPU allocation, or model initialization are used.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_glm53_mamba_foundation_transfer.sh"


class LegacyMambaIntentGateTests(unittest.TestCase):
    def test_default_glm_first_request_fails_before_work(self) -> None:
        env = os.environ.copy()
        env.pop("HF_TOKEN", None)
        env.pop("IQ_LEXICAL_FOUNDATION", None)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing legacy Mamba-founded", result.stderr)

    def test_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], check=False, capture_output=True, text=True, timeout=5
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
