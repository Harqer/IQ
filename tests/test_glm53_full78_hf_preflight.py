"""Cost-free fake-metadata regression tests for the HF full78 Job preflight."""
from __future__ import annotations

import unittest

from scripts.preflight_glm53_full78_hf_inference import (
    EXPECTED_SHARDS, EXPECTED_TENSORS,
    hardware_plan, validate_full78,
)


class Full78HFPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exact expected counts but no model checkpoint allocation.
        files = [f"shard-{i:05d}.safetensors" for i in range(EXPECTED_SHARDS)]
        cls.index = {
            "metadata": {"total_size": 539388468802},
            "weight_map": {
                f"tensor.{i}": files[i % EXPECTED_SHARDS]
                for i in range(EXPECTED_TENSORS)
            },
        }
        cls.config = {
            "model": {"vocab_size": 128256, "hidden_size": 4096},
            "stable_moe": {"num_experts": 256, "top_k": 8},
            "schedule": {"layers": (["mamba3", "hca", "moe"] * 78) + ["mamba3"]},
        }
        cls.marker = {
            "status": "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED",
            "inference_ready": False,
            "mapped_routed_expert_slots": 19200,
            "tokenizer_packaged": False,
        }

    def test_real_full78_geometry_and_external_tokenizer(self):
        r = validate_full78(self.index, self.config, self.marker)
        self.assertEqual(r["weight_shards"], 20283)
        self.assertEqual(r["weight_tensors"], 63164)
        self.assertEqual(r["mamba_layers"], 79)
        self.assertEqual(r["moe_layers"], 78)
        self.assertFalse(r["tokenizer_packaged"])

    def test_no_unsupported_gpu_launch_is_marked_ready(self):
        for flavor in ("h200", "h200x4", "h200x8", "a100x8"):
            with self.subTest(flavor=flavor):
                p = hardware_plan(539388468802, flavor)
                self.assertFalse(p["current_loader_model_parallel"])
                self.assertFalse(p["current_loader_full78_inference_runnable"])
                self.assertIn("one CUDA device", p["reason"])

    def test_short_index_rejected(self):
        short = {
            "metadata": self.index["metadata"],
            "weight_map": {"tensor.1": "shard-00001.safetensors"},
        }
        with self.assertRaisesRegex(RuntimeError, "tensor count"):
            validate_full78(short, self.config, self.marker)

    def test_false_inference_completion_rejected(self):
        marker = {**self.marker, "inference_ready": True}
        with self.assertRaisesRegex(RuntimeError, "inference readiness"):
            validate_full78(self.index, self.config, marker)

    def test_wrong_vocab_rejected(self):
        c = {**self.config, "model": {"vocab_size": 201088, "hidden_size": 4096}}
        with self.assertRaisesRegex(RuntimeError, "vocabulary"):
            validate_full78(self.index, c, self.marker)


if __name__ == "__main__":
    unittest.main()
