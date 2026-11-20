"""Small byte-level fixtures: no 539 GB download, GPU, or credentials."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import struct
import tempfile
from pathlib import Path
import unittest

from iq_cloud.remote_shard_loader import (
    RemoteShardError, validate_digest_manifest,
    verified_remote_shard, snapshot_remote_shard, verify_safetensors_header,
)

def make_shard():
    header = {
        "x.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "y.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [16, 20]},
    }
    data = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(data)) + data + b"\x00"*20


class FakeRemote:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def info(self, uri):
        return {"size": len(self.data)}

    def open(self, uri, mode):
        self.calls += 1
        return BytesIO(self.data)


class RemoteShardTests(unittest.TestCase):
    uri = "hf://buckets/owner/project/run/model-fake.safetensors"

    def test_header_validates_without_reading_payload(self):
        blob = make_shard()
        stream = BytesIO(blob)
        h = verify_safetensors_header(
            stream, file_size=len(blob), expected_keys={"x.weight", "y.weight"}
        )
        self.assertEqual(set(h), {"x.weight", "y.weight"})
        self.assertLess(stream.tell(), len(blob))

    def test_verified_shard_is_ephemeral_and_digest_checked(self):
        blob = make_shard()
        fs = FakeRemote(blob)
        with tempfile.TemporaryDirectory() as root:
            with verified_remote_shard(
                fs, self.uri, sha256(blob).hexdigest(),
                expected_keys={"x.weight", "y.weight"}, scratch_dir=root,
                chunk_bytes=7,
            ) as (path, header):
                self.assertEqual(path.read_bytes(), blob)
                self.assertIn("x.weight", header)
                self.assertEqual(len(list(Path(root).iterdir())), 1)
            self.assertEqual(list(Path(root).iterdir()), [])
        self.assertEqual(fs.calls, 1)

    def test_experimental_single_pass_snapshot_is_unreviewed(self):
        blob = make_shard()
        fs = FakeRemote(blob)
        with tempfile.TemporaryDirectory() as root:
            with snapshot_remote_shard(
                fs, self.uri, expected_keys={"x.weight", "y.weight"},
                scratch_dir=root, chunk_bytes=7,
            ) as (path, header, digest):
                self.assertEqual(digest, sha256(blob).hexdigest())
                self.assertEqual(path.read_bytes(), blob)
                self.assertEqual(set(header), {"x.weight", "y.weight"})
            self.assertEqual(list(Path(root).iterdir()), [])
        self.assertEqual(fs.calls, 1)

    def test_experimental_corrupted_header_does_not_yield(self):
        blob = make_shard()
        with self.assertRaisesRegex(RemoteShardError, "tensor names"):
            with snapshot_remote_shard(
                FakeRemote(blob), self.uri,
                expected_keys={"unexpected.weight"},
            ):
                self.fail("unverified/invalid structure was exposed")

    def test_verified_loading_never_accepts_unpinned_hash(self):
        with self.assertRaisesRegex(RemoteShardError, "pinned SHA-256"):
            with verified_remote_shard(
                FakeRemote(make_shard()), self.uri, None,
                expected_keys={"x.weight", "y.weight"},
            ):
                self.fail("unpinned input accepted by verified loader")

    def test_checksum_failure_does_not_yield_or_leave_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RemoteShardError, "SHA-256"):
                with verified_remote_shard(
                    FakeRemote(make_shard()), self.uri, "0"*64,
                    expected_keys={"x.weight", "y.weight"}, scratch_dir=root,
                ):
                    self.fail("corrupt shard was yielded")
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_header_corruption_and_extra_name_fail(self):
        blob = make_shard()
        with self.assertRaisesRegex(RemoteShardError, "tensor names"):
            verify_safetensors_header(
                BytesIO(blob), file_size=len(blob), expected_keys={"not-there"}
            )
        header = json.loads(blob[8:8+struct.unpack("<Q", blob[:8])[0]])
        header["y.weight"]["data_offsets"] = [12, 16]
        data = json.dumps(header).encode()
        broken = struct.pack("<Q", len(data))+data+b"\0"*20
        with self.assertRaisesRegex(RemoteShardError, "overlap"):
            verify_safetensors_header(BytesIO(broken), file_size=len(broken))

    def test_untrusted_manifest_fails_closed(self):
        valid = {
            "schema_version": 1, "source_index_sha256": "a"*64,
            "sha256_algorithm": "sha256",
            "model_source_status": "compiled_not_inference_validated",
            "shards_sha256": {"model-a.safetensors": "b"*64},
        }
        self.assertEqual(
            validate_digest_manifest(valid, {"model-a.safetensors"},
                                     source_index_digest="a"*64),
            valid["shards_sha256"],
        )
        for invalid in (
            dict(valid, source_index_sha256="b"*64),
            dict(valid, shards_sha256={}),
            dict(valid, shards_sha256={"model-../a.safetensors": "b"*64}),
        ):
            with self.assertRaises(RemoteShardError):
                validate_digest_manifest(
                    invalid, {"model-a.safetensors"}, source_index_digest="a"*64
                )

    def test_malformed_stream_sizes_and_unsafe_path(self):
        data = make_shard()
        for uri in ("file:///tmp/model.safetensors",
                    "hf://buckets/a/../model.safetensors"):
            with self.assertRaises(RemoteShardError):
                with verified_remote_shard(
                    FakeRemote(data), uri, sha256(data).hexdigest(),
                    expected_keys={"x.weight", "y.weight"},
                ):
                    self.fail("unsafe URI accepted")
        with self.assertRaisesRegex(RemoteShardError, "cap"):
            with verified_remote_shard(
                FakeRemote(data), self.uri, sha256(data).hexdigest(),
                expected_keys={"x.weight", "y.weight"},
                max_shard_bytes=16,
            ):
                self.fail("oversize accepted")


if __name__ == "__main__":
    unittest.main()
