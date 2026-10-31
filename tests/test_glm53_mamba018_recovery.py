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
        # Source layer 7 is PRETRAINED, not an identity-init Mamba slot.
        # Original donor bias/state values may be BF16 or FP32.
        self.assertEqual(
            inventory["layers.18.mamba.core.dt_bias"][1],
            frozenset({"BF16", "F32"}),
        )

    def test_missing_and_wrong_tensor_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.safetensors"
            save_file({"layers.18.norm.weight": torch.ones(4096, dtype=torch.bfloat16)}, str(path))
            with self.assertRaisesRegex(RuntimeError, "Wrong Mamba shard tensor keys"):
                validate_local_shard(path, self.config)

    def test_pretrained_dt_bias_accepts_bf16_and_f32_but_rejects_fp16(self):
        # Mock a safetensors header to test exact dtype dispatch without
        # allocating the actual 213 MB pretrained Mamba shard on CI.
        from contextlib import nullcontext
        from unittest.mock import patch

        inventory = expected_mamba_shapes(self.config)

        class Slice:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
            def get_shape(self):
                return self.shape
            def get_dtype(self):
                return self.dtype

        class Reader:
            def __init__(self, bias_dtype):
                self.bias_dtype = bias_dtype
            def keys(self):
                return inventory.keys()
            def get_slice(self, key):
                shape, dtype = inventory[key]
                actual = ("BF16" if isinstance(dtype, frozenset) else dtype)
                if key.endswith(".dt_bias"):
                    actual = self.bias_dtype
                return Slice(shape, actual)

        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "test.safetensors"
            file.write_bytes(b"test")
            for valid_dtype in ("BF16", "F32"):
                with self.subTest(valid_dtype=valid_dtype), patch(
                    "scripts.recover_glm53_mamba018.safe_open",
                    return_value=nullcontext(Reader(valid_dtype))
                ):
                    validate_local_shard(file, self.config)

            with patch(
                "scripts.recover_glm53_mamba018.safe_open",
                return_value=nullcontext(Reader("F16"))
            ):
                with self.assertRaisesRegex(RuntimeError, "got shape=.*dtype=F16"):
                    validate_local_shard(file, self.config)

    def test_invalid_mamba_depth_rejected(self):
        bad = SimpleNamespace(mamba3=self.config.mamba3,
                              schedule=SimpleNamespace(positions=lambda _: tuple(range(78))))
        with self.assertRaisesRegex(RuntimeError, "79-Mamba schedule"):
            expected_mamba_shapes(bad)


if __name__ == "__main__":
    unittest.main()
