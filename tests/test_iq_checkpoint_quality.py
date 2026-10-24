"""CPU-only checks that experimental IQ metadata and gibberish cannot pass."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = load_script("audit_iq_checkpoint.py")
client = load_script("iq_hf_inference.py")


class TestIQQuality(unittest.TestCase):
    def test_corrupted_output_rejected(self):
        examples = (
            ", Hadateraterateraterateraterater,vaNat …king,b",
            "blblblblblbl",
            "",
            "3",
            "The answer is 8.",
            "4, vaNataterateraterater",
        )
        for answer in examples:
            with self.subTest(answer=answer):
                self.assertFalse(client.check_generation_smoke(answer)[0])

    def test_arithmetic_canary(self):
        for answer in ("4", "4.", "4.0", "four", "The answer is 4.", "Answer: 4"):
            with self.subTest(answer=answer):
                self.assertTrue(client.check_generation_smoke(answer)[0])

    def test_failed_generation_raises_not_ready(self):
        with patch.object(client, "chat_once", return_value="Hadateraterateraterater"):
            with self.assertRaisesRegex(RuntimeError, "MODEL_QUALITY_FAILURE"):
                client.verify_generation(object())

    def test_metadata_weight_only_is_blocker(self):
        cfg = {
            "model": {"vocab_size": 128256, "hidden_size": 4096},
            "schedule": {"layers": ["mamba3"] * 32 + ["csa"] * 12 + ["hca"] * 12 + ["moe"] * 24},
        }
        manifest = {
            "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
            "calibration_mode": "weight_only",
            "foundation": {"repo": "state-spaces/mamba3-mimo-1.5b"},
            "recipient_config_fingerprint": "abc",
        }
        report = audit.audit(cfg, manifest)
        blockers = [f for f in report["findings"] if f["level"] == "blocker"]
        self.assertTrue(any(f["key"] == "activation_calibration" for f in blockers))
        self.assertFalse(report["language_quality_confirmed"])

    def test_widened_lexical_head_preserves_donor_logits_on_replicated_states(self):
        # Mirrors mamba3_direct.expand_mamba3_foundation_globals, factor=2.
        source_hidden = [0.25, -1.5, 0.75]
        source_head_row = [1.25, 0.5, -0.25]
        donor_logit = sum(a * b for a, b in zip(source_hidden, source_head_row))
        recipient_hidden = source_hidden + source_hidden
        recipient_head = [v / 2 for v in source_head_row] * 2
        transplanted_logit = sum(
            a * b for a, b in zip(recipient_hidden, recipient_head)
        )
        self.assertAlmostEqual(donor_logit, transplanted_logit, places=7)
        # Shows why later nonreplicated residual transformations are dangerous:
        off_subspace = source_hidden + [x * 3 for x in source_hidden]
        off_subspace_logit = sum(
            a * b for a, b in zip(off_subspace, recipient_head)
        )
        self.assertNotAlmostEqual(donor_logit, off_subspace_logit, places=7)

    def test_wrong_vocab_fails(self):
        cfg = {
            "model": {"vocab_size": 201088, "hidden_size": 4096},
            "schedule": {"layers": ["mamba3"] * 32 + ["csa"] * 12 + ["hca"] * 12 + ["moe"] * 24},
        }
        manifest = {
            "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
            "calibration_mode": "activation_refined",
            "foundation": {"repo": "state-spaces/mamba3-mimo-1.5b"},
        }
        errors = [f for f in audit.audit(cfg, manifest)["findings"] if f["level"] == "error"]
        self.assertTrue(any(f["key"] == "vocabulary" for f in errors))


if __name__ == "__main__":
    unittest.main()
