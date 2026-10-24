"""CPU-only strict remote safetensors reader tests; no network or model weights."""
from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from iq_transfer.donor import DonorError
from iq_transfer.http_range_source import HubSafetensorsSource


class _Response:
    def __init__(self, blob: bytes, start: int, end: int, *, status: int = 206):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}
        self.raw = io.BytesIO(blob[start:end + 1])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Session:
    def __init__(self, blob: bytes, *, status: int = 206):
        self.blob = blob
        self.status = status
        self.ranges: list[tuple[int, int]] = []

    def get(self, url, *, headers, stream, allow_redirects, timeout):
        assert stream and allow_redirects and timeout
        assert "resolve" in url
        assert headers.get("Accept-Encoding") == "identity"
        assert headers["Range"].startswith("bytes=")
        first, last = map(int, headers["Range"][6:].split("-"))
        self.ranges.append((first, last))
        return _Response(self.blob, first, last, status=self.status)


def _fixture(root: Path, *, bad_shape: bool = False) -> tuple[bytes, str]:
    shard = "model-00001-of-00001.safetensors"
    # Two BF16 rows of two elements: [[1, 2], [3, 4]].
    weights = struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)
    header = {
        "model.layers.0.weight": {
            "dtype": "BF16",
            "shape": [2, 3] if bad_shape else [2, 2],
            "data_offsets": [0, len(weights)],
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(encoded)) + encoded + weights
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": shard}}),
        encoding="utf-8",
    )
    return blob, "model.layers.0.weight"


class TestHTTPRangeSource(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.blob, self.key = _fixture(self.root)
        self.session = _Session(self.blob)
        self.token_patch = patch("iq_transfer.http_range_source.get_token", return_value=None)
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)

    def source(self, **kwargs):
        return HubSafetensorsSource(
            self.root, repo_id="zai-org/GLM-5.3-BF16",
            revision="abcdef0123456789abcdef0123456789abcdef01",
            session=self.session, **kwargs,
        )

    def test_shape_is_header_only_and_cached(self):
        source = self.source()
        self.assertEqual(source.shape(self.key), (2, 2))
        self.assertEqual(source.shape(self.key), (2, 2))
        self.assertEqual(len(self.session.ranges), 2)
        self.assertEqual(self.session.ranges[0], (0, 7))

    def test_bf16_tensor_values_and_chunked_range_requests(self):
        import torch
        source = self.source(max_request_bytes=3)
        weights = source.get(self.key)
        self.assertEqual(weights.dtype, torch.bfloat16)
        self.assertEqual(weights.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(self.session.ranges), 5)  # two header + three data
        self.assertTrue(all(b - a + 1 <= 3 for a, b in self.session.ranges[2:]))
        self.assertFalse((self.root / "model-00001-of-00001.safetensors").exists())

    def test_rows_preserve_order_duplicates_and_dtype(self):
        import torch
        source = self.source(max_request_bytes=4)
        got = source.get_rows(self.key, [1, 0, 1])
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(got.tolist(), [[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(source.get_rows(self.key, []).shape, (0, 2))
        self.assertFalse((self.root / "model-00001-of-00001.safetensors").exists())

    def test_out_of_range_row_rejected(self):
        with self.assertRaises(DonorError):
            self.source().get_rows(self.key, [2])

    def test_reject_full_shard_http_200(self):
        self.session = _Session(self.blob, status=200)
        with self.assertRaisesRegex(DonorError, "remote shard metadata rejected"):
            self.source().shape(self.key)

    def test_reject_bad_tensor_shape_before_payload_read(self):
        self.blob, self.key = _fixture(self.root, bad_shape=True)
        self.session = _Session(self.blob)
        with self.assertRaisesRegex(DonorError, "byte-length mismatch"):
            self.source().get(self.key)
        self.assertEqual(len(self.session.ranges), 2)

    def test_max_tensor_bytes_prevents_materialization(self):
        with self.assertRaisesRegex(DonorError, "chunk-compatible transform required"):
            self.source(max_tensor_bytes=4).get(self.key)
        self.assertEqual(len(self.session.ranges), 2)

    def test_missing_tensor_is_rejected(self):
        with self.assertRaisesRegex(DonorError, "unknown checkpoint tensor"):
            self.source().shape("does-not-exist")

    def test_cleanup_does_not_delete_local_files(self):
        source = self.source()
        sentinel = self.root / "keep.txt"
        sentinel.write_text("keep")
        source.clear_ephemeral_shards(completed_layer=77)
        self.assertTrue(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
