"""CPU-only tests of planned GLM BF16 shard transfer size estimator."""
from __future__ import annotations

import unittest

from scripts.estimate_glm53_shard_bytes import (
    REVISION, ShardEstimateError, estimate_shard_bytes,
)


class TestGLM53ShardSizes(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "source_revision": REVISION,
            "target_experts": 64,
            "official_shards": [
                "model-00001-of-00282.safetensors",
                "model-00002-of-00282.safetensors",
            ],
        }

    def test_estimate_does_not_require_weights(self):
        estimate = estimate_shard_bytes(
            self.plan,
            {
                "model-00001-of-00282.safetensors": 5_000_000_000,
                "model-00002-of-00282.safetensors": 4_000_000_000,
            },
            capacity_bytes=4_000_000_000,
        )
        self.assertEqual(estimate["status"], "metadata_estimate_only")
        self.assertEqual(estimate["bytes_to_download_if_every_planned_shard_fetched_once"], 9_000_000_000)
        self.assertFalse(estimate["all_shards_fit_simultaneously"])

    def test_incomplete_shard_size_report_fails(self):
        with self.assertRaisesRegex(ShardEstimateError, "did not report sizes"):
            estimate_shard_bytes(
                self.plan, {"model-00001-of-00282.safetensors": 1},
                capacity_bytes=1,
            )

    def test_duplicate_and_unexpected_revision_fail(self):
        wrong = {**self.plan, "source_revision": "unverified"}
        with self.assertRaises(ShardEstimateError):
            estimate_shard_bytes(wrong, {}, capacity_bytes=0)
        duplicate = {**self.plan, "official_shards": [
            self.plan["official_shards"][0], self.plan["official_shards"][0]
        ]}
        with self.assertRaises(ShardEstimateError):
            estimate_shard_bytes(duplicate, {}, capacity_bytes=0)


if __name__ == "__main__":
    unittest.main()
