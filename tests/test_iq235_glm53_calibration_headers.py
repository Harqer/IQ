"""Synthetic only: no 200 MB maps or pretrained model weights downloaded."""
from __future__ import annotations

from io import BytesIO
import json
from hashlib import sha256
import struct
import unittest

from iq_cloud.checkpoint_metadata import IQ235MetadataError
from iq_cloud.glm53_calibration_headers import (
    inspect_coordinate_map_header, inspect_map_metadata, EXPECTED_SPACES,
)


class GLMCalibrationHeaderTests(unittest.TestCase):
    def test_small_synthetic_coordinate_map_header(self):
        payload = json.dumps({
            "matrix": {
                "dtype": "F64", "shape": [3, 2],
                "data_offsets": [0, 48],
            },
        }).encode()
        header = struct.pack("<Q", len(payload))+payload
        result = inspect_coordinate_map_header(
            BytesIO(header), file_size=len(header)+48,
            source_dims=3, target_dims=2,
        )
        self.assertEqual(result["matrix_shape"], [3, 2])
        self.assertFalse(result["matrix_payload_read"])

    def test_wrong_dtype_shape_and_short_prefix_rejected(self):
        for shape, dtype in [([2, 3], "F64"), ([3, 2], "BF16")]:
            data = json.dumps({
                "matrix": {
                    "dtype": dtype, "shape": shape,
                    "data_offsets": [0, 48],
                },
            }).encode()
            packet = struct.pack("<Q", len(data)) + data
            with self.assertRaisesRegex(IQ235MetadataError, "shape/dtype"):
                inspect_coordinate_map_header(
                    BytesIO(packet), file_size=len(packet)+48,
                    source_dims=3, target_dims=2,
                )
        with self.assertRaisesRegex(IQ235MetadataError, "prefix"):
            inspect_coordinate_map_header(BytesIO(b""), file_size=100)

    def test_source_provenance_verified_but_quality_unproven(self):
        metadata = {
            "schema_version":1, "ridge":1e-12,
            "source_space": EXPECTED_SPACES["lexical-input"][0],
            "target_space": EXPECTED_SPACES["lexical-input"][1],
            "diagnostics":None,
        }
        raw = (json.dumps(metadata, sort_keys=True)+"\n").encode()
        out = inspect_map_metadata("lexical-input", raw, sha256(raw).hexdigest())
        self.assertFalse(out["calibration_quality_validated"])
        with self.assertRaisesRegex(IQ235MetadataError, "checksum"):
            inspect_map_metadata("lexical-input", raw+b" ", sha256(raw).hexdigest())
        metadata["diagnostics"] = {"validation_rmse": 0.0}
        bad = (json.dumps(metadata, sort_keys=True)+"\n").encode()
        with self.assertRaisesRegex(IQ235MetadataError, "provenance"):
            inspect_map_metadata("lexical-input", bad, sha256(bad).hexdigest())


if __name__ == "__main__":
    unittest.main()
