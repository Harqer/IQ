"""No GPU: prevent Baseten/Modal deployment provenance regressions."""
from __future__ import annotations

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


class DeploymentSourceContracts(unittest.TestCase):
    def test_baseten_uses_exact_vcs_checkout_provenance(self):
        text = (ROOT / "deploy/baseten_full_iq/config.yaml").read_text()
        self.assertIn(f"git+https://github.com/state-spaces/mamba.git@{COMMIT}",text)
        self.assertIn('d.read_text("direct_url.json")',text)
        self.assertIn('get("commit_id")==',text)
        self.assertIn("if chunk_start + cs < S:",text)
        self.assertNotIn("--no-deps /tmp/iq-mamba3-src",text)
        self.assertIn("IQ_PHASE: architecture_only",text)
        parsed = yaml.safe_load(text)
        self.assertEqual(parsed["model_class_name"], "Model")
        self.assertEqual(parsed["resources"]["accelerator"], "H200:8")
        self.assertEqual(parsed["environment_variables"]["IQ_PHASE"], "architecture_only")
        self.assertEqual(len(parsed["build_commands"]), 3)
        self.assertTrue(any(
            "git+https://github.com/state-spaces/mamba.git@" in command
            for command in parsed["build_commands"]
        ))

    def test_modal_requires_actual_stage_command_not_read_only_preflight(self):
        text = (ROOT / "deploy/modal_iq235_phase_a.py").read_text()
        self.assertIn("python3 -m scripts.stage_iq235_baseten",text)
        self.assertIn("from the repository first",text)
        self.assertIn("modal run deploy/modal_iq235_phase_a.py",text)
        self.assertIn("checkpoint_weights_injected",text)


if __name__ == "__main__":
    unittest.main()
