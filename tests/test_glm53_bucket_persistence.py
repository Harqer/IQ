"""No-network failure injection for durable Hugging Face checkpoint writes."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from iq_transfer.glm53_bucket_persistence import (
    BucketCheckpointError,
    durable_output, publish_file, read_checkpoint_json,
    remote_checkpoint_sizes, write_checkpoint_text,
)


class TestBucketPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "mounted" / "IQ" / "runs" / "glm53-full78-local-test"
        self.output.mkdir(parents=True)
        self.stage = self.root / "local-stage"
        self.saved = {}
        self.bad_readback = False
        self.environ = patch.dict(os.environ, {
            "IQ_HF_DURABLE_OUTPUT": str(self.output),
            "IQ_HF_DURABLE_BUCKET": "kofecheckmate/iq-glm53-all256",
            "IQ_HF_DURABLE_PREFIX": "IQ/runs/glm53-full78-local-test",
            "IQ_HF_DURABLE_STAGING": str(self.stage),
        })
        self.environ.start()
        self.addCleanup(self.environ.stop)

        def batch(bucket, *, add):
            self.assertEqual(bucket, "kofecheckmate/iq-glm53-all256")
            for source, name in add:
                self.saved[name] = Path(source).read_bytes()

        def info(bucket, names):
            return (
                SimpleNamespace(path=name, size=len(self.saved[name]))
                for name in names if name in self.saved
            )

        def download(bucket, *, files, raise_on_missing_files):
            self.assertTrue(raise_on_missing_files)
            for remote, target in files:
                if remote not in self.saved:
                    raise FileNotFoundError(remote)
                data = self.saved[remote]
                if self.bad_readback:
                    data = data[:-1] + bytes([data[-1] ^ 255])
                Path(target).write_bytes(data)

        for name, side_effect in (
            ("batch_bucket_files", batch),
            ("get_bucket_paths_info", info),
            ("download_bucket_files", download),
        ):
            mock = patch(f"huggingface_hub.{name}", side_effect=side_effect)
            mock.start()
            self.addCleanup(mock.stop)

    def test_direct_upload_independent_sha256_and_inventory(self):
        self.assertTrue(durable_output(self.output))
        source = self.root / "source.bin"
        source.write_bytes(b"real-shard" * 1000)
        publish_file(self.output, "model-mamba-018.safetensors", source)
        name = "IQ/runs/glm53-full78-local-test/model-mamba-018.safetensors"
        self.assertEqual(self.saved[name], source.read_bytes())
        sizes = remote_checkpoint_sizes(self.output, ["model-mamba-018.safetensors"])
        self.assertEqual(sizes["model-mamba-018.safetensors"], source.stat().st_size)
        with self.assertRaisesRegex(BucketCheckpointError, "Refusing to overwrite"):
            publish_file(self.output, "model-mamba-018.safetensors", source)

    def test_corrupt_remote_readback_fails_closed(self):
        source = self.root / "source.bin"
        source.write_bytes(b"A" * 128)
        self.bad_readback = True
        with self.assertRaisesRegex(BucketCheckpointError, "SHA256 readback mismatch"):
            publish_file(self.output, "model-mamba-018.safetensors", source)

    def test_missing_remote_shard_blocks_manifest(self):
        with self.assertRaisesRegex(BucketCheckpointError, "missing/invalid 1"):
            remote_checkpoint_sizes(self.output, ["model-mamba-018.safetensors"])

    def test_metadata_idempotent_and_preexisting_stage_receipt(self):
        write_checkpoint_text(self.output, "iq_config.json", '{"n": 1}\n')
        write_checkpoint_text(self.output, "iq_config.json", '{"n": 1}\n')
        self.assertEqual(read_checkpoint_json(self.output, "iq_config.json"), {"n": 1})
        with self.assertRaisesRegex(BucketCheckpointError, "readback mismatch"):
            write_checkpoint_text(self.output, "iq_config.json", '{"n": 2}\n')
        write_checkpoint_text(self.output, "model-moe-003-projection.json", '{"stage":3}\n')
        write_checkpoint_text(
            self.output, "model-moe-003-projection.json",
            '{"stage":3,"resumed":true}\n', preserve_existing=True,
        )
        self.assertEqual(
            read_checkpoint_json(self.output, "model-moe-003-projection.json"),
            {"stage": 3},
        )


if __name__ == "__main__":
    unittest.main()
