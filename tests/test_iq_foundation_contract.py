"""CPU-only, no-network checks for the IQ foundation provenance contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "audit_iq_foundation_contract.py"
spec = importlib.util.spec_from_file_location("audit_iq_foundation_contract", MODULE)
assert spec is not None and spec.loader is not None
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class IQFoundationContractTests(unittest.TestCase):
    def fixture(self, root: Path, *, glm: bool) -> None:
        foundation = audit_module.GLM if glm else audit_module.MAMBA
        files = {
            "iq_config.json": {"model": {"vocab_size": 154880 if glm else 128256, "hidden_size": 4096}},
            "transplant_manifest.json": {
                "method": "glm53_lexical_with_mamba_hybrid" if glm else "mamba3_foundation_plus_glm53_capability_weight_transfer",
                "foundation": {"repo": foundation},
                "lexical_source": foundation if glm else None,
                "calibration_mode": "activation_refined",
            },
            "model.safetensors.index.json": {
                "metadata": {"total_size": 10},
                "weight_map": {
                    "embed_tokens.weight": "model-global.safetensors",
                    "norm.weight": "model-global.safetensors",
                    "lm_head.weight": "model-global.safetensors",
                }
            },
            "COMPILE_COMPLETE.json": {
                "inference_ready": True,
                "end_to_end_knowledge_retention_proven": True,
            },
        }
        for name, obj in files.items():
            (root / name).write_text(json.dumps(obj), encoding="utf-8")

    def test_current_mamba_foundation_fails_glm_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, glm=False)
            result = audit_module.audit(root)
            self.assertEqual(result["status"], "blocked")
            codes = {f["code"] for f in result["findings"]}
            self.assertIn("vocab_mismatch", codes)
            self.assertIn("wrong_lexical_foundation", codes)
            self.assertIn("wrong_transfer_direction", codes)

    def test_metadata_consistency_is_not_quality_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, glm=True)
            result = audit_module.audit(root)
            self.assertEqual(result["status"], "metadata_consistent_not_quality_verified")
            self.assertIn("NOT prove", result["note"])

    def test_inference_not_ready_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, glm=True)
            p = root / "COMPILE_COMPLETE.json"
            status = json.loads(p.read_text(encoding="utf-8"))
            status["inference_ready"] = False
            p.write_text(json.dumps(status), encoding="utf-8")
            codes = {f["code"] for f in audit_module.audit(root)["findings"]}
            self.assertIn("inference_not_ready", codes)

    def test_missing_checkpoint_metadata_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, glm=True)
            (root / "transplant_manifest.json").unlink()
            self.assertEqual(audit_module.audit(root)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
