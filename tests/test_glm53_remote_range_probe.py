"""CPU-only fail-closed tests for partial Hugging Face safetensors reads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_glm53_remote_ranges.py"
spec = importlib.util.spec_from_file_location("probe_glm53_remote_ranges", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestRemoteRanges(unittest.TestCase):
    def test_exact_http_206_range_is_required(self):
        content, size = mod.parse_range_response(
            status_code=206,
            content_range="bytes 0-7/5000000000",
            payload=struct.pack("<Q", 100),
            start=0, end=7,
        )
        self.assertEqual(len(content), 8)
        self.assertEqual(size, 5_000_000_000)
        with self.assertRaisesRegex(mod.RangeProbeError, "did not honor"):
            mod.parse_range_response(
                status_code=200,
                content_range=None,
                payload=b"too big",
                start=0, end=7,
            )

    def test_content_range_mismatch_rejected(self):
        with self.assertRaisesRegex(mod.RangeProbeError, "wrong byte range"):
            mod.parse_range_response(
                status_code=206,
                content_range="bytes 1-8/5000000000",
                payload=b"12345678",
                start=0, end=7,
            )
        with self.assertRaisesRegex(mod.RangeProbeError, "invalid Content-Range"):
            mod.parse_range_response(
                status_code=206,
                content_range="invalid",
                payload=b"12345678",
                start=0, end=7,
            )

    def test_parse_safetensors_header_without_data(self):
        header = json.dumps({
            "__metadata__": {"format": "pt"},
            "model.layers.0.input_layernorm.weight": {
                "dtype": "BF16", "shape": [4], "data_offsets": [0, 8],
            },
        }).encode()
        parsed = mod.parse_safetensors_header(len(header), header, 8 + len(header) + 8)
        self.assertIn("model.layers.0.input_layernorm.weight", parsed)

    def test_header_out_of_bounds_rejected(self):
        header = json.dumps({
            "expert": {"dtype": "BF16", "shape": [8], "data_offsets": [0, 1000]}
        }).encode()
        with self.assertRaisesRegex(mod.RangeProbeError, "past file boundary"):
            mod.parse_safetensors_header(len(header), header, 8 + len(header) + 16)

    def test_reject_large_or_corrupted_header(self):
        with self.assertRaises(mod.RangeProbeError):
            mod.parse_safetensors_header(mod.MAX_HEADER + 1, b"test", 100000000)
        with self.assertRaises(mod.RangeProbeError):
            mod.parse_safetensors_header(5, b"oops!", 100)


if __name__ == "__main__":
    unittest.main()
