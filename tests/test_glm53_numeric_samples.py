"""CPU-only synthetic checks; no remote bucket or original donor downloads."""
import io
import json
import struct
import unittest
from unittest.mock import patch

import numpy as np

from scripts.verify_glm53_iq_numeric_samples import (
    RemoteShard, compute_sparse_row_reference, pass_compare,
    verify_contiguous_header, verify_dense_zero, verify_mamba_widening,
)


class TestIQNumericalSampleVerifier(unittest.TestCase):
    def test_sparse_reference_and_comparison(self):
        import torch
        source = np.array([1, 2, 3, 4], dtype=np.float64)
        basis = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)
        actual = compute_sparse_row_reference(source, basis)
        expected = torch.tensor([1., 2.], dtype=torch.bfloat16).float().numpy()
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(pass_compare(expected, actual)["bad"], 0)
        with self.assertRaisesRegex(RuntimeError, "Numerical mismatch"):
            pass_compare(expected, actual + 0.5)

    def test_invalid_header_gap_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Overlapping or gapped"):
            verify_contiguous_header(
                {"x": {"shape": [2], "dtype": "BF16", "data_offsets": [1, 5]}},
                8 + 20 + 5, 20
            )

    def test_dense_expert_full_zero_and_nonzero_rejection(self):
        p = "layers.2.moe.routed_experts.0."
        header = {
            p + suffix: {"shape": [2, 2], "dtype": "BF16", "data_offsets": [i * 8, (i+1)*8]}
            for i, suffix in enumerate(
                ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
            )
        }
        file = self._fake_shard(header, b"\x00" * 24)
        self.assertEqual(verify_dense_zero(file, physical=2)["nonzero_bytes"], 0)
        file._payload = file._payload[:-1] + b"\x01"
        with self.assertRaisesRegex(RuntimeError, "nonzero data"):
            verify_dense_zero(file, physical=2)

    def test_mamba_repeated_values_and_corruption(self):
        import torch
        p = "layers.54."
        objects = {
            p + "norm.weight": torch.tensor([1., 2., 1., 2.], dtype=torch.bfloat16),
            p + "mamba.core.dt_bias": torch.tensor([3., 4., 3., 4.], dtype=torch.float32),
            p + "mamba.core.D": torch.tensor([5., 6., 5., 6.], dtype=torch.float32),
            p + "mamba.core.B_bias": torch.tensor([[[7.]], [[8.]], [[7.]], [[8.]]], dtype=torch.float32),
        }
        entries = {}
        payload = bytearray()
        for k, x in objects.items():
            raw = x.view(torch.uint8).numpy().tobytes()
            entries[k] = {"shape": list(x.shape), "dtype": "BF16" if x.dtype == torch.bfloat16 else "F32",
                          "data_offsets": [len(payload), len(payload)+len(raw)]}
            payload.extend(raw)
        f = self._fake_shard(entries, bytes(payload))
        self.assertEqual(len(verify_mamba_widening(f, physical=54)), 4)
        objects[p + "mamba.core.D"] = torch.tensor([5., 6., 5., 9.], dtype=torch.float32)
        # Updating last vector only via reconstructed bytes would be cumbersome;
        # patch tensor read directly to simulate a corrupted second half.
        saved = f.tensor
        def corrupted(key):
            if key == p + "mamba.core.D":
                return np.array([5., 6., 5., 9.])
            return saved(key)
        with patch.object(f, "tensor", side_effect=corrupted):
            with self.assertRaisesRegex(RuntimeError, "Mamba replication mismatch"):
                verify_mamba_widening(f, physical=54)

    @staticmethod
    def _fake_shard(entries, payload):
        raw = json.dumps(entries).encode()
        data = struct.pack("<Q", len(raw)) + raw + payload
        # No network: use range reader over an exact byte buffer.
        obj = object.__new__(RemoteShard)
        obj.name, obj.size = "synthetic.safetensors", len(data)
        obj.header_size, obj.entries = len(raw), entries
        obj._payload = data
        obj._range = lambda a, b: obj._payload[a:b+1]
        return obj


if __name__ == "__main__":
    unittest.main()
