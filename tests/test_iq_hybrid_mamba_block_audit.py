"""No-GPU regression tests for full Mamba block structural audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
script = ROOT / "scripts" / "audit_iq_hybrid_mamba_block.py"
spec = importlib.util.spec_from_file_location("audit_iq_hybrid_mamba_block", script)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FullMambaBlockAuditTests(unittest.TestCase):
    def test_existing_mixer_only_runtime_is_not_marked_full_block(self):
        source = (ROOT / "iq_model" / "hybrid.py").read_text(encoding="utf-8")
        report = audit.audit_hybrid_mamba(source)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("block_component:mlp", report["findings"])
        self.assertIn("forward_component:mlp", report["findings"])
        self.assertIn("block_component:norm2", report["findings"])

    def test_complete_synthetic_structure_never_claims_tensor_parity(self):
        source = """
class Mamba3ResidualLayer:
    def __init__(self):
        self.norm = object()
        self.mamba = object()
        self.norm2 = object()
        self.mlp = object()

    def forward(self, x):
        x = self.mamba(self.norm(x))
        return self.mlp(self.norm2(x))
"""
        report = audit.audit_hybrid_mamba(source)
        self.assertEqual(report["status"], "structurally_complete_unverified")
        self.assertIn("NOT", report["note"])

    def test_missing_class_fails_closed(self):
        self.assertEqual(audit.audit_hybrid_mamba("class Other: pass")["status"],
                         "blocked")


if __name__ == "__main__":
    unittest.main()
