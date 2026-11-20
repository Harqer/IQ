"""Hash small synthetic in-memory SafeTensors shards; no network/GPU."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import struct
import unittest

from iq_cloud.shard_digest_snapshot import (
    hash_one_remote_shard, snapshot_bucket_shard_hashes,
)
from iq_cloud.remote_shard_loader import (
    RemoteShardError, validate_digest_manifest,
)

def synthetic():
    h = {"linear.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    encoded=json.dumps(h).encode()
    return struct.pack("<Q",len(encoded))+encoded+b"\0"*4

class DummyFS:
    def __init__(self, obj):
        self.obj=obj
    def info(self, uri):
        return {"size":len(self.obj)}
    def open(self, uri, mode):
        return BytesIO(self.obj)

class SnapshotTests(unittest.TestCase):
    def test_exact_shard_sha_and_index_coverage(self):
        blob=synthetic()
        fs=DummyFS(blob)
        digest,size=hash_one_remote_shard(
            fs,"hf://buckets/test/run/model-tiny.safetensors",
            expected_keys={"linear.weight"},chunk_bytes=5,
        )
        self.assertEqual(digest,sha256(blob).hexdigest())
        self.assertEqual(size,len(blob))
        index={"weight_map":{"linear.weight":"model-tiny.safetensors"}}
        report=snapshot_bucket_shard_hashes(
            fs,base="hf://buckets/test/run",source_index=index,
            expected_sha256="a"*64,
        )
        self.assertEqual(report["shards_sha256"]["model-tiny.safetensors"],digest)
        self.assertFalse(report["inference_ready"])
        self.assertEqual(
            validate_digest_manifest(report,{"model-tiny.safetensors"},
                                     source_index_digest="a"*64)["model-tiny.safetensors"],
            digest,
        )

    def test_unexpected_key_refused_before_hash(self):
        with self.assertRaisesRegex(RemoteShardError,"tensor names"):
            hash_one_remote_shard(
                DummyFS(synthetic()),
                "hf://buckets/test/run/model-tiny.safetensors",
                expected_keys={"bad.weight"},
            )

    def test_unsafe_index_name_refused(self):
        with self.assertRaisesRegex(RemoteShardError,"invalid source index"):
            snapshot_bucket_shard_hashes(
                DummyFS(synthetic()),base="hf://buckets/test/run",
                source_index={"weight_map":{"linear.weight":"../bad.safetensors"}},
            )


if __name__=="__main__":
    unittest.main()
