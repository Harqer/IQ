from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file

from iq_transfer.glm53_fp8 import GLM53DequantizedSource
from iq_transfer.donor import DonorError


class GLM53FP8Tests(unittest.TestCase):
    def test_blocks_and_boundary_tiles(self):
        with TemporaryDirectory() as root:
            torch.manual_seed(23)
            fp8 = torch.randint(-24, 25, (130, 259)).float().to(torch.float8_e4m3fn)
            scale = torch.tensor(
                [[.001, .002, .003], [.004, .005, .006]], dtype=torch.float32
            )
            norm = torch.ones(130, dtype=torch.bfloat16)
            save_file(
                {"a.weight": fp8, "a.weight_scale_inv": scale, "norm.weight": norm},
                str(Path(root) / "model.safetensors"),
            )
            source = GLM53DequantizedSource(root)
            self.assertEqual(source.quantized_weight_count, 1)
            self.assertEqual(source.shape("a.weight"), (130, 259))
            self.assertNotIn("a.weight_scale_inv", source.keys())
            self.assertEqual(source.checkpoint_files()[0].name, "model.safetensors")
            expected = (
                fp8.float()
                * scale.repeat_interleave(128, 0)[:130]
                .repeat_interleave(128, 1)[:, :259]
            ).to(torch.bfloat16)
            torch.testing.assert_close(source.get("a.weight"), expected, atol=0, rtol=0)
            torch.testing.assert_close(source.get("norm.weight"), norm, atol=0, rtol=0)

    def test_orphan_scale_rejected(self):
        with TemporaryDirectory() as root:
            save_file(
                {"a.weight_scale_inv": torch.ones((1, 1))},
                str(Path(root) / "model.safetensors"),
            )
            with self.assertRaisesRegex(DonorError, "orphan"):
                GLM53DequantizedSource(root)

    def test_wrong_geometry_rejected(self):
        with TemporaryDirectory() as root:
            save_file(
                {
                    "a.weight": torch.ones((128, 129)),
                    "a.weight_scale_inv": torch.ones((1, 1)),
                },
                str(Path(root) / "model.safetensors"),
            )
            with self.assertRaisesRegex(DonorError, "geometry"):
                GLM53DequantizedSource(root)

    def test_fp8_without_scale_rejected(self):
        with TemporaryDirectory() as root:
            save_file(
                {"a.weight": torch.ones((128, 128)).to(torch.float8_e4m3fn)},
                str(Path(root) / "model.safetensors"),
            )
            with self.assertRaisesRegex(DonorError, "missing block scales"):
                GLM53DequantizedSource(root).get("a.weight")

    def test_nonpositive_scale_rejected(self):
        with TemporaryDirectory() as root:
            save_file(
                {
                    "a.weight": torch.ones((128, 128)).to(torch.float8_e4m3fn),
                    "a.weight_scale_inv": torch.zeros((1, 1)),
                },
                str(Path(root) / "model.safetensors"),
            )
            with self.assertRaisesRegex(DonorError, "positive"):
                GLM53DequantizedSource(root).get("a.weight")


if __name__ == "__main__":
    unittest.main()
