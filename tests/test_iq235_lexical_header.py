"""CPU-only SafeTensors lexical header regression checks (synthetic bytes)."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
import math
import struct
import unittest

from iq_cloud.checkpoint_metadata import IQ235MetadataError
from iq_cloud.lexical_header import EXPECTED_SHAPES, inspect_global_header


def make_header(*, bad_shapes=None, bad_dtype=None, bad_offsets=None):
    data = {}
    offset = 0
    for name, shape in EXPECTED_SHAPES.items():
        shape = list((bad_shapes or {}).get(name, shape))
        size = math.prod(shape) * 2
        data[name] = {
            "dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]
        }
        offset += size
    if bad_dtype:
        data["lm_head.weight"]["dtype"] = bad_dtype
    if bad_offsets:
        data["lm_head.weight"]["data_offsets"] = bad_offsets
    header = json.dumps(data, separators=(",", ":")).encode()
    # Test only constructs the header, NOT the multi-gigabyte tensor payload.
    prefix = struct.pack("<Q", len(header))
    return prefix + header, 8 + len(header) + offset


class LexicalHeaderTests(unittest.TestCase):
    def test_shapes_and_total_size_without_allocating_payload(self):
        data, size = make_header()
        result = inspect_global_header(BytesIO(data), file_size=size)
        self.assertEqual(result["original_lexical_tensors"]["embed_tokens.weight"]["shape"],
                         [128256, 4096])
        self.assertEqual(result["target_glm_embedding_shape"], [154880, 4096])
        self.assertFalse(result["tensor_payloads_decoded"])
        self.assertFalse(result["pretrained_inference_ready"])
        self.assertEqual(result["header_bytes_inspected"], len(data))

    def test_truncated_prefix_and_header_fail(self):
        data, size = make_header()
        for cut in (0, 3, 8, len(data) - 1):
            with self.subTest(cut=cut), self.assertRaises(IQ235MetadataError):
                inspect_global_header(BytesIO(data[:cut]), file_size=size)

    def test_wrong_embedding_shape_fails(self):
        data, size = make_header(
            bad_shapes={"embed_tokens.weight": [154880, 4096]}
        )
        with self.assertRaisesRegex(IQ235MetadataError, "shape"):
            inspect_global_header(BytesIO(data), file_size=size)

    def test_wrong_dtype_fails(self):
        data, size = make_header(bad_dtype="F32")
        with self.assertRaisesRegex(IQ235MetadataError, "dtype"):
            inspect_global_header(BytesIO(data), file_size=size)

    def test_corrupt_offsets_fail(self):
        data, size = make_header(bad_offsets=[0, 8])
        with self.assertRaisesRegex(IQ235MetadataError, "byte span"):
            inspect_global_header(BytesIO(data), file_size=size)

    def test_extra_header_tensor_and_size_mismatch_fail(self):
        data, size = make_header()
        header = json.loads(data[8:])
        header["fake.weight"] = {
            "dtype": "BF16", "shape": [1], "data_offsets": [0, 2]
        }
        raw = json.dumps(header).encode()
        altered = struct.pack("<Q", len(raw)) + raw
        with self.assertRaisesRegex(IQ235MetadataError, "names"):
            inspect_global_header(BytesIO(altered), file_size=size)
        with self.assertRaisesRegex(IQ235MetadataError, "file size"):
            inspect_global_header(BytesIO(data), file_size=size + 1)

    def test_huge_header_size_rejected_without_reading_body(self):
        data = struct.pack("<Q", 2**63)
        with self.assertRaisesRegex(IQ235MetadataError, "out of bounds"):
            inspect_global_header(BytesIO(data), file_size=2101354776)


if __name__ == "__main__":
    unittest.main()
