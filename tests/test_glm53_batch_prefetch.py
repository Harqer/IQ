"""Offline tests for bounded HF range prefetch; no network or paid GPU."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from iq_transfer.glm53_batch_prefetch import RoutedBatchPrefetch, StagedBatchSource
from iq_transfer.glm53_gpu_batch import ExpertSpec
from iq_transfer.glm53_gpu_projection import ProjectionComputeError
from iq_transfer.http_range_source import HubSafetensorsSource


class PrefetchTests(unittest.TestCase):
    def _local_hf_source(self, root: Path) -> HubSafetensorsSource:
        tensors = {}
        for donor in (7, 25):
            base = f"model.layers.3.mlp.experts.{donor}"
            for kind, shape in (
                ("gate", (3, 4)), ("up", (3, 4)), ("down", (4, 3)),
            ):
                tensors[f"{base}.{kind}_proj.weight"] = (
                    torch.full(shape, donor, dtype=torch.bfloat16)
                )
        save_file(tensors, str(root / "model-00001-of-00001.safetensors"))
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                key: "model-00001-of-00001.safetensors" for key in tensors
            }}), encoding="utf-8",
        )
        return HubSafetensorsSource(
            root, repo_id="test/fake", revision="pinned-test",
            token="not-a-real-token",
        )

    def test_one_batch_ahead_uses_isolated_local_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._local_hf_source(Path(tmp))
            root = "model.layers.3.mlp"
            with patch.dict(os.environ, {"IQ_MOE_GPU_PREFETCH": "1"}):
                with RoutedBatchPrefetch(source, donor_root=root) as prefetch:
                    self.assertTrue(prefetch.enabled)
                    first = prefetch.submit([ExpertSpec(0, 7)])
                    second = prefetch.submit([ExpertSpec(1, 25)])
                    a = prefetch.source_for(first, source)
                    b = prefetch.source_for(second, source)
                    for spec, staged in ((7, a), (25, b)):
                        for name in ("gate", "up", "down"):
                            key = f"{root}.experts.{spec}.{name}_proj.weight"
                            self.assertTrue(torch.all(staged.get(key) == spec))
                    with self.assertRaises(ProjectionComputeError):
                        a.get(f"{root}.experts.25.gate_proj.weight")
            source.close()

    def test_queue_backpressure_prevents_unbounded_host_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._local_hf_source(Path(tmp))
            with patch.dict(os.environ, {"IQ_MOE_GPU_PREFETCH": "1"}):
                with RoutedBatchPrefetch(
                    source, donor_root="model.layers.3.mlp",
                ) as pipeline:
                    first = pipeline.submit([ExpertSpec(0, 7)])
                    second = pipeline.submit([ExpertSpec(1, 25)])
                    with self.assertRaisesRegex(
                        ProjectionComputeError, "one-ahead",
                    ):
                        pipeline.submit([ExpertSpec(2, 7)])
                    a = pipeline.source_for(first, source)
                    self.assertTrue(a.get(
                        "model.layers.3.mlp.experts.7.gate_proj.weight"
                    ).numel() > 0)
                    third = pipeline.submit([ExpertSpec(2, 7)])
                    pipeline.source_for(second, source)
                    pipeline.source_for(third, source)
                    with self.assertRaisesRegex(
                        ProjectionComputeError, "already consumed",
                    ):
                        pipeline.source_for(first, source)
            source.close()

    def test_prefetch_disabled_does_not_create_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._local_hf_source(Path(tmp))
            with patch.dict(os.environ, {"IQ_MOE_GPU_PREFETCH": "0"}):
                with RoutedBatchPrefetch(source, donor_root="model.layers.3.mlp") as pipeline:
                    self.assertFalse(pipeline.enabled)
                    self.assertIsNone(pipeline.submit([ExpertSpec(0, 7)]))
                    self.assertIs(pipeline.source_for(None, source), source)
            source.close()

    def test_invalid_mode_fails_closed(self):
        with patch.dict(os.environ, {"IQ_MOE_GPU_PREFETCH": "many"}):
            with self.assertRaises(ProjectionComputeError):
                RoutedBatchPrefetch(object(), donor_root="model.layers.3.mlp")

    def test_staged_source_has_no_uncached_remote_fallback(self):
        src = StagedBatchSource({"present": torch.ones(1)})
        self.assertEqual(src.get("present").item(), 1.0)
        with self.assertRaises(ProjectionComputeError):
            src.get("missing")


if __name__ == "__main__":
    unittest.main()
