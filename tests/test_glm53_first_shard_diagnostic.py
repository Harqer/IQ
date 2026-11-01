"""Free first-shard diagnostic tests; synthetic weights only, no HF billing."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from scripts.diagnose_glm53_first_shard import (
    find_first_base_shard, read_shard, read_first_remote_bucket_shard,
)


class FirstShardTests(unittest.TestCase):
    def test_reads_first_base_shard_and_validates_weights(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            # Filename sorting matches the real CUDA loader, not JSON order.
            save_file({"layers.0.norm.weight": torch.ones(3)},
                      str(root / "model-context-000.safetensors"))
            save_file({"layers.1.norm.weight": torch.ones(3)},
                      str(root / "model-context-001.safetensors"))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "layers.1.norm.weight": "model-context-001.safetensors",
                    "layers.0.moe.routed_experts.5.gate_proj.weight": "moe-shard.safetensors",
                    "layers.0.norm.weight": "model-context-000.safetensors",
                }
            }), encoding="utf-8")
            name, keys = find_first_base_shard(root / "model.safetensors.index.json")
            self.assertEqual(name, "model-context-000.safetensors")
            self.assertEqual(keys, {"layers.0.norm.weight"})
            report = read_shard(root)
            self.assertEqual(report["tensor_count"], 1)
            self.assertGreater(report["bytes"], 0)

    def test_remote_bucket_reads_only_index_and_one_bounded_shard(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "source.safetensors"
            save_file({"layers.0.norm.weight": torch.ones(3)}, str(source))
            copied = []

            def download(bucket_id, *, files):
                self.assertEqual(bucket_id, "kofecheckmate/iq-glm53-all256")
                self.assertEqual(len(files), 1)
                remote, local = files[0]
                copied.append(remote)
                target = Path(local)
                if remote.endswith("model.safetensors.index.json"):
                    target.write_text(json.dumps({
                        "weight_map": {
                            "layers.0.norm.weight": "model-context-000.safetensors",
                        }
                    }))
                else:
                    self.assertTrue(remote.endswith("model-context-000.safetensors"))
                    target.write_bytes(source.read_bytes())

            with patch("huggingface_hub.get_token", return_value="test-token"):
                with patch("huggingface_hub.download_bucket_files", side_effect=download):
                    with patch("huggingface_hub.get_bucket_file_metadata",
                               return_value=SimpleNamespace(size=source.stat().st_size)):
                        result = read_first_remote_bucket_shard(
                            "kofecheckmate/iq-glm53-all256"
                        )
            self.assertEqual(len(copied), 2)
            self.assertEqual(result["tensor_count"], 1)
            self.assertIs(result["gpu_allocated"], False)

    def test_remote_bucket_size_cap_blocks_download(self):
        with TemporaryDirectory() as temp:
            def download(_bucket_id, *, files):
                remote, local = files[0]
                if remote.endswith(".json"):
                    Path(local).write_text(json.dumps({
                        "weight_map": {"x.weight": "model-context-000.safetensors"}
                    }))
                else:
                    self.fail("Oversized shard should never download")
            with patch("huggingface_hub.get_token", return_value="test-token"):
                with patch("huggingface_hub.download_bucket_files", side_effect=download):
                    with patch("huggingface_hub.get_bucket_file_metadata",
                               return_value=SimpleNamespace(size=513*1024**2)):
                        with self.assertRaisesRegex(ValueError, "512 MiB"):
                            read_first_remote_bucket_shard(
                                "kofecheckmate/iq-glm53-all256"
                            )

    def test_nonfinite_tensor_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            save_file({"x.weight": torch.tensor([float("nan")])},
                      str(root / "bad.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x.weight": "bad.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Nonfloating/nonfinite"):
                read_shard(root)

    def test_path_traversal_not_accepted(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "model.safetensors.index.json"
            p.write_text(json.dumps({
                "weight_map": {"x.weight": "../secret.safetensors"}
            }))
            with self.assertRaisesRegex(ValueError, "Unsafe shard filename"):
                find_first_base_shard(p)


if __name__ == "__main__":
    unittest.main()
