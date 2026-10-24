"""Unit tests for immutable Mamba/lexical shard reuse during GLM expert expansion."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from iq_model import HybridLayerType
from iq_transfer.complete_transplant import canonical_glm53_config
from iq_transfer.foundation_reuse import (
    FoundationReuseError,
    reuse_frozen_mamba_foundation,
)


MAMBA_REV = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
GLM_REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


def fixture(base: Path) -> Path:
    base.mkdir()
    original_config = canonical_glm53_config()
    original_config.write_json(str(base / "iq_config.json"))
    (base / "transplant_manifest.json").write_text(json.dumps({
        "foundation": {"repo": "state-spaces/mamba3-mimo-1.5b", "revision": MAMBA_REV},
        "capability_source": {"repo": "zai-org/GLM-5.3-BF16", "revision": GLM_REV},
    }))
    mapping = {
        "embed_tokens.weight": "model-global.safetensors",
        "norm.weight": "model-global.safetensors",
        "lm_head.weight": "model-global.safetensors",
    }
    files = ["model-global.safetensors"]
    positions = original_config.schedule.positions(HybridLayerType.MAMBA3)
    for ordinal, physical in enumerate(positions):
        name = f"model-mamba-{ordinal:03d}.safetensors"
        mapping[f"layers.{physical}.norm.weight"] = name
        mapping[f"layers.{physical}.mamba.core.out_proj.weight"] = name
        files.append(name)
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    for name in files:
        (base / name).write_bytes(("immutable " + name).encode())
    return base


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def call(original: Path, output: Path, *, mapping=None, glm_revision=GLM_REV):
    return reuse_frozen_mamba_foundation(
        original=original,
        output=output,
        expected_source_fingerprint=canonical_glm53_config().fingerprint,
        target_config=canonical_glm53_config(routed_experts=64),
        mamba_revision=MAMBA_REV,
        glm_revision=glm_revision,
        weight_map={} if mapping is None else mapping,
    )


class TestFoundationReuse(unittest.TestCase):
    def test_copies_33_frozen_shards_and_preserves_all_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = fixture(Path(tmp) / "original")
            output = Path(tmp) / "candidate"
            output.mkdir()
            keys = {}
            report = call(original, output, mapping=keys)
            self.assertEqual(report["foundation_shards_reused"], 33)
            self.assertTrue(report["mamba_pretrained_weights_unchanged"])
            self.assertFalse(report["full_hybrid_behavior_preserved"])
            self.assertEqual(len(report["file_sha256"]), 33)
            self.assertGreater(len(keys), 32)
            for name, digest in report["file_sha256"].items():
                self.assertEqual(digest, sha(original / name))
                self.assertEqual(digest, sha(output / name))
            self.assertEqual(original.stat().st_ino != output.stat().st_ino, True)

    def test_refuses_source_mismatch_and_output_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = fixture(Path(tmp) / "original")
            with self.assertRaisesRegex(FoundationReuseError, "overlap"):
                call(original, original)
            output = Path(tmp) / "candidate"
            output.mkdir()
            with self.assertRaisesRegex(FoundationReuseError, "revisions"):
                call(original, output, glm_revision="0" * 40)
            self.assertEqual(list(output.iterdir()), [])

    def test_invalid_mamba_mapping_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = fixture(Path(tmp) / "original")
            data = json.loads((original / "model.safetensors.index.json").read_text())
            data["weight_map"].pop("layers.0.norm.weight")
            (original / "model.safetensors.index.json").write_text(json.dumps(data))
            output = Path(tmp) / "candidate"
            output.mkdir()
            with self.assertRaisesRegex(FoundationReuseError, "mismatched"):
                call(original, output)

    def test_duplicate_tensor_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = fixture(Path(tmp) / "original")
            output = Path(tmp) / "candidate"
            output.mkdir()
            with self.assertRaisesRegex(FoundationReuseError, "duplicate"):
                call(original, output, mapping={"norm.weight": "other.safetensors"})


if __name__ == "__main__":
    unittest.main()
