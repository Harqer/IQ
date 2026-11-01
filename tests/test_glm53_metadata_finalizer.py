"""Small, CPU-only checks of no-tensor-allocation finalizer validators."""
import errno
import io
import json
from unittest.mock import patch
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

    def test_fast_weight_map_covers_every_expected_shard(self):
        from scripts.finalize_glm53_full78_metadata import build_fast_weight_map
        schedule = ["mamba3", "csa", "moe"] * 78 + ["mamba3"]
        stages = [
            {"stage": i, "source_layer": i,
             "context_physical_layer": i * 3 + 1,
             "moe_physical_layer": i * 3 + 2}
            for i in range(78)
        ]
        native = {
            "attnres.weights", "reasoning.weights", "reasoning_injector.weights"
        }
        mapping = build_fast_weight_map(schedule, stages, native)
        self.assertEqual(set(mapping.values()), expected_names())
        self.assertEqual(
            mapping["layers.0.mamba.core.dt_bias"],
            "model-mamba-000.safetensors"
        )
        self.assertEqual(
            mapping["layers.16.attention.q_a_proj.weight"],
            "model-context-005.safetensors"
        )
        self.assertEqual(
            mapping["layers.17.moe.routed_experts.255.gate_proj.weight"],
            "model-moe-005-expert-255.safetensors"
        )
        self.assertEqual(
            mapping["layers.17.moe.shared_experts.0.down_proj.weight"],
            "model-moe-005-shared.safetensors"
        )

    def test_fast_weight_map_rejects_overlapping_physical_stages(self):
        from scripts.finalize_glm53_full78_metadata import build_fast_weight_map
        schedule = ["mamba3", "csa", "moe"] * 78 + ["mamba3"]
        stages = [
            {"stage": i, "source_layer": i,
             "context_physical_layer": i * 3 + 1,
             "moe_physical_layer": i * 3 + 2}
            for i in range(78)
        ]
        stages[1]["context_physical_layer"] = stages[0]["context_physical_layer"]
        native = {
            "attnres.weights", "reasoning.weights", "reasoning_injector.weights"
        }
        with self.assertRaisesRegex(RuntimeError, "Duplicate expected tensor key"):
            build_fast_weight_map(schedule, stages, native)

    def test_fast_samples_fail_closed_on_key_mismatch(self):
        from scripts.finalize_glm53_full78_metadata import validate_fast_samples
        with patch(
            "scripts.finalize_glm53_full78_metadata.remote_header",
            return_value={"not_expected": {"dtype": "BF16", "shape": [1]}}
        ):
            with self.assertRaisesRegex(RuntimeError, "Sample tensor layout mismatch"):
                validate_fast_samples(
                    object(), {"model-global.safetensors": 20},
                    {"embed_tokens.weight": "model-global.safetensors"},
                    latent=2048, intermediate=2048,
                )

    def test_small_valid_safetensors_header(self):
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
            })
            self.assertEqual(safetensors_header(file)["tensor.weight"]["shape"], [1])

    def test_native_controller_scalar_safetensors_header(self):
        # attnres.mixers.*.mix_logit is a PyTorch scalar, shape [].
        # Scalars are valid safetensors entries and use one dtype element.
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [], "data_offsets": [0, 4],
            })
            entries = safetensors_header(file)
            self.assertEqual(entries["tensor.weight"]["shape"], [])
            self.assertEqual(entries["tensor.weight"]["dtype"], "F32")

    def test_native_scalar_rejects_wrong_byte_length(self):
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [], "data_offsets": [0, 8],
            })
            with self.assertRaisesRegex(RuntimeError, "Invalid tensor byte extent"):
                safetensors_header(file)

    def test_non_scalar_invalid_dimension_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [0], "data_offsets": [0, 4],
            })
            with self.assertRaisesRegex(RuntimeError, "Invalid tensor metadata"):
                safetensors_header(file)

    def test_truncated_or_mismatched_tensor_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = self._write(temp, {
                "dtype": "F32", "shape": [2], "data_offsets": [0, 8],
            })
            with self.assertRaisesRegex(RuntimeError, "Invalid tensor byte extent|Truncated"):
                safetensors_header(file)

    def test_remote_header_streams_only_prefix_and_header(self):
        from scripts.finalize_glm53_full78_metadata import remote_header

        class BoundedStream(io.BytesIO):
            def read(self, count=-1):
                if count == -1 or count > 4096:
                    raise AssertionError("Unbounded or excessive safetensors read")
                return super().read(count)

        class RemoteFS:
            def open(self, path, mode, *, block_size, cache_type):
                assert path.endswith("/test.safetensors")
                assert (mode, block_size, cache_type) == ("rb", 0, "none")
                return BoundedStream(data)

        header = json.dumps({"tensor.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
        }}).encode()
        data = struct.pack("<Q", len(header)) + header + b"\x00\x01\x02\x03"
        entries = remote_header(RemoteFS(), "test.safetensors", len(data))
        self.assertEqual(list(entries), ["tensor.weight"])

    def test_nfs_eio_falls_back_to_hf_small_json(self):
        from scripts.finalize_glm53_full78_metadata import checkpoint_json

        class RemoteFS:
            def open(self, path, mode, *, block_size, cache_type):
                assert path.endswith("/run-provenance.json")
                assert (mode, block_size, cache_type) == ("rb", 0, "none")
                return io.BytesIO(b'{"schema_version":1}')

        with tempfile.TemporaryDirectory() as td:
            with patch.object(Path, "read_text", side_effect=OSError(errno.EIO, "NFS failed")):
                record = checkpoint_json(
                    RemoteFS(), Path(td), "run-provenance.json",
                    read_mode="auto",
                )
        self.assertEqual(record["schema_version"], 1)

    def test_bad_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "bad.safetensors"
            file.write_bytes(struct.pack("<Q", 1024) + b"none" * 4)
            with self.assertRaisesRegex(RuntimeError, "Invalid safetensors header length"):
                safetensors_header(file)


if __name__ == "__main__":
    unittest.main()
