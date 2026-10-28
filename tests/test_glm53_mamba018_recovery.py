"""CPU-only checks: recovery requires exact Mamba header and no overwrite."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from iq_model import Mamba3MIMOConfig
from scripts.recover_glm53_mamba018 import (
    expected_mamba_shapes,
    validate_local_shard,
)


class _Schedule:
    def positions(self, mode):
        return tuple(range(79))


class TestMamba018Recovery(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            mamba3=Mamba3MIMOConfig.production_4096x32(),
            schedule=_Schedule(),
        )

    def test_expected_12_keys_and_physical_layer(self):
        inventory = expected_mamba_shapes(self.config)
        self.assertEqual(len(inventory), 12)
        self.assertTrue(all(x.startswith("layers.18.") for x in inventory))
        self.assertEqual(inventory["layers.18.norm.weight"], ((4096,), "BF16"))

    def test_missing_and_wrong_tensor_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.safetensors"
            save_file({"layers.18.norm.weight": torch.ones(4096, dtype=torch.bfloat16)}, str(path))
            with self.assertRaisesRegex(RuntimeError, "Wrong Mamba shard tensor keys"):
                validate_local_shard(path, self.config)

    def test_invalid_mamba_depth_rejected(self):
        bad = SimpleNamespace(mamba3=self.config.mamba3,
                              schedule=SimpleNamespace(positions=lambda _: tuple(range(78))))
        with self.assertRaisesRegex(RuntimeError, "79-Mamba schedule"):
            expected_mamba_shapes(bad)


if __name__ == "__main__":
    unittest.main()
