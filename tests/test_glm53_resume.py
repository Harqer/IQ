"""Resume must recover only complete, structurally valid IQ shards."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file
from iq_transfer.glm53_resume import (
    GLM53ResumeError, expected_expert_keys, register_saved_shard,
    validate_legacy_partial,
)


class ShardResumeTests(unittest.TestCase):
    def test_register_expert_without_loading_all_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "model-moe-003-expert-040.safetensors"
            keys = expected_expert_keys(11, 40)
            save_file({key: torch.ones((2, 3), dtype=torch.bfloat16)
                       for key in keys}, str(root / name))
            validate_legacy_partial(root)
            index = {}
            self.assertTrue(register_saved_shard(
                root, name, index, layer=11, expected_keys=keys))
            self.assertEqual(index, {key: name for key in keys})
            self.assertFalse(register_saved_shard(
                root, "model-moe-003-expert-041.safetensors", index))
            with self.assertRaises(GLM53ResumeError):
                register_saved_shard(root, name, index, layer=11,
                                     expected_keys=keys)

    def test_corrupt_and_mismatched_shards_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "model-context-003.safetensors"
            (root / name).write_bytes(b"not a safetensors shard")
            with self.assertRaises((GLM53ResumeError, Exception)):
                register_saved_shard(root, name, {})
            save_file({"layers.8.norm.weight": torch.ones(8)}, str(root / name))
            with self.assertRaises(GLM53ResumeError):
                register_saved_shard(root, name, {}, layer=10)
            with self.assertRaises(GLM53ResumeError):
                register_saved_shard(root, "../outside.safetensors", {})

    def test_legacy_adoption_disallows_unknown_files_and_completed_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "model-moe-003-common.safetensors"
            save_file({"layers.11.norm.weight": torch.ones(4)}, str(root / name))
            validate_legacy_partial(root)
            (root / "COMPILE_COMPLETE.json").write_text("{}")
            with self.assertRaises(GLM53ResumeError):
                validate_legacy_partial(root)
            (root / "COMPILE_COMPLETE.json").unlink()
            (root / "other.bin").write_bytes(b"x")
            with self.assertRaises(GLM53ResumeError):
                validate_legacy_partial(root)


if __name__ == "__main__":
    unittest.main()
