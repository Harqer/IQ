"""CPU-only integrity tests with synthetic safetensors; NEVER a model/inference test."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from iq_transfer.checkpoint_integrity import IQIntegrityError, verify_iq_checkpoint_integrity


class PublishedCheckpointIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.blobs = {
            "model-mamba-000.safetensors": ("layers.0.weight", struct.pack("<2f", 1.0, -2.0)),
            "model-moe-000.safetensors": ("layers.1.weight", struct.pack("<2f", 0.0, 3.5)),
        }
        for shard, (tensor, payload) in self.blobs.items():
            self._write_shard(shard, tensor, payload)
        self._republish()

    def _write_shard(self, filename, tensor, payload, *, offsets=None, dtype="F32"):
        if offsets is None:
            offsets = [0, len(payload)]
        header = {tensor: {"dtype": dtype, "shape": [2], "data_offsets": offsets}}
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        (self.root / filename).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    def _republish(self):
        weights = {tensor: filename for filename, (tensor, _payload) in self.blobs.items()}
        paths = list(self.blobs)
        total = sum((self.root / name).stat().st_size for name in paths)
        index = {"metadata": {"total_size": total}, "weight_map": weights}
        encoded = json.dumps(index, sort_keys=True).encode()
        (self.root / "model.safetensors.index.json").write_bytes(encoded)
        sha_lines = [
            f"{hashlib.sha256((self.root / name).read_bytes()).hexdigest()}  {name}"
            for name in sorted(paths)
        ]
        (self.root / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
        (self.root / "COMPLETE.json").write_text(json.dumps({
            "shards": len(paths), "index_sha256": hashlib.sha256(encoded).hexdigest(),
        }), encoding="utf-8")

    def test_complete_published_shards_verified(self):
        result = verify_iq_checkpoint_integrity(self.root)
        self.assertEqual(result.verified_sha256_count, 2)
        self.assertEqual(result.verified_header_count, 2)
        self.assertEqual(result.tensor_count, 2)
        self.assertEqual(result.shard_bytes, sum(x.stat().st_size for x in self.root.glob("*.safetensors")))

    def test_one_byte_corruption_fails(self):
        shard = self.root / "model-mamba-000.safetensors"
        content = bytearray(shard.read_bytes())
        content[-1] ^= 1
        shard.write_bytes(content)
        with self.assertRaisesRegex(IQIntegrityError, "SHA-256 mismatch"):
            verify_iq_checkpoint_integrity(self.root)

    def test_index_hash_tampering_fails(self):
        path = self.root / "model.safetensors.index.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(IQIntegrityError, "index SHA-256 mismatch"):
            verify_iq_checkpoint_integrity(self.root)

    def test_overlap_fails_even_if_republished(self):
        name, (tensor, payload) = next(iter(self.blobs.items()))
        self._write_shard(name, tensor, payload, offsets=[1, 9])
        self._republish()
        with self.assertRaisesRegex(IQIntegrityError, "out-of-range tensor data|overlapping/gapped"):
            verify_iq_checkpoint_integrity(self.root)

    def test_wrong_dtype_fails_even_if_signed(self):
        name, (tensor, payload) = next(iter(self.blobs.items()))
        self._write_shard(name, tensor, payload, dtype="I16")
        self._republish()
        with self.assertRaisesRegex(IQIntegrityError, "tensor byte count mismatch"):
            verify_iq_checkpoint_integrity(self.root)

    def test_missing_published_shard_fails(self):
        (self.root / "model-moe-000.safetensors").unlink()
        with self.assertRaises(IQIntegrityError):
            verify_iq_checkpoint_integrity(self.root)

    def test_duplicate_hash_manifest_fails(self):
        path = self.root / "SHA256SUMS"
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(IQIntegrityError, "invalid/duplicate"):
            verify_iq_checkpoint_integrity(self.root)

    def test_independent_manifest_pin_blocks_republished_corruption(self):
        name, (tensor, payload) = next(iter(self.blobs.items()))
        self._write_shard(name, tensor, payload[:-4] + struct.pack("<f", 999.0))
        self._republish()  # An attacker also rewrites the sidecar hash list.
        with patch.dict("os.environ", {"IQ_EXPECTED_SHA256SUMS_SHA256": "0" * 64}):
            with self.assertRaisesRegex(IQIntegrityError, "independently pinned"):
                verify_iq_checkpoint_integrity(self.root)

    def test_extra_tensor_header_fails(self):
        name, (tensor, payload) = next(iter(self.blobs.items()))
        self._write_shard(name, "unexpected.weight", payload)
        self._republish()
        with self.assertRaisesRegex(IQIntegrityError, "tensor-index disagreement"):
            verify_iq_checkpoint_integrity(self.root)


if __name__ == "__main__":
    unittest.main()
