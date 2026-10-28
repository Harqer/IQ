"""Small, CPU-only checks of no-tensor-allocation finalizer validators."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.finalize_glm53_full78_metadata import expected_names, safetensors_header


class TestMetadataFinalizer(unittest.TestCase):
    def test_expected_full78_shard_count(self):
        files = expected_names()
        self.assertEqual(len(files), 20283)
        self.assertIn("model-mamba-018.safetensors", files)
        self.assertIn("model-moe-077-expert-255.safetensors", files)
        self.assertIn("model-native.safetensors", files)

    def _write(self, root, item, payload=b"\x00\x01\x02\x03"):
        header = json.dumps({"tensor.weight": item}, separators=(",", ":")).encode()
        dest = Path(root) / "test.safetensors"
        dest.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        return dest

    def test_small_valid_safetensors_header(self):
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
            })
            self.assertEqual(safetensors_header(file)["tensor.weight"]["shape"], [1])

    def test_truncated_or_mismatched_tensor_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [2], "data_offsets": [0, 8],
            })
            with self.assertRaisesRegex(RuntimeError, "Invalid tensor byte extent|Truncated"):
                safetensors_header(file)

    def test_bad_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "bad.safetensors"
            file.write_bytes(struct.pack("<Q", 1024) + b"none")
            with self.assertRaisesRegex(RuntimeError, "Invalid safetensors header length"):
                safetensors_header(file)


if __name__ == "__main__":
    unittest.main()
