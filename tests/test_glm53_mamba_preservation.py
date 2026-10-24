"""CPU fixture validates byte-identical Mamba foundation gate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_mamba_frozen.py"
spec = importlib.util.spec_from_file_location("verify_frozen_mamba", SCRIPT)
assert spec and spec.loader
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def build_fixture(parent: Path, folder: str):
    root = parent / folder
    root.mkdir()
    config = {
        "model": {"vocab_size": 128256, "hidden_size": 4096},
        "mamba3": {"num_layers": 32, "d_model": 4096},
        "schedule": {"layers": ["mamba3", "csa", "moe"]},
        "stable_moe": {"num_experts": 32 if folder == "original" else 64},
    }
    (root / "iq_config.json").write_text(json.dumps(config))
    shard_names = {"model-global.safetensors"} | {
        f"model-mamba-{i:03d}.safetensors" for i in range(32)
    }
    weight_map = {
        "embed_tokens.weight": "model-global.safetensors",
        "norm.weight": "model-global.safetensors",
        "lm_head.weight": "model-global.safetensors",
    }
    weight_map.update({f"layers.{i}.mamba.weight": f"model-mamba-{i:03d}.safetensors"
                       for i in range(32)})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    for name in shard_names:
        (root / name).write_bytes(("fixture:" + name).encode())
    return root


class TestMambaFrozen(unittest.TestCase):
    def test_identical_recurrent_and_lexical_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = build_fixture(Path(tmp), "original")
            b = build_fixture(Path(tmp), "candidate")
            result = check.verify_mamba_preservation(a, b)
            self.assertTrue(result["bitwise_mamba_and_lexical_weights_identical"])
            self.assertFalse(result["runtime_behavior_preserved"])
            self.assertEqual(result["foundation_shards"], 33)

    def test_mamba_weight_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = build_fixture(Path(tmp), "original")
            b = build_fixture(Path(tmp), "candidate")
            (b / "model-mamba-019.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(check.MambaPreservationError, "weights changed"):
                check.verify_mamba_preservation(a, b)

    def test_changed_vocab_and_schedule_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = build_fixture(Path(tmp), "original")
            b = build_fixture(Path(tmp), "candidate")
            path = b / "iq_config.json"
            cfg = json.loads(path.read_text())
            cfg["model"]["vocab_size"] = 154880
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(check.MambaPreservationError, "vocabulary"):
                check.verify_mamba_preservation(a, b)


if __name__ == "__main__":
    unittest.main()
