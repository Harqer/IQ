"""CPU-only checks for exact full78 routed-expert caching (no H200 Jobs)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file

from iq_model.hybrid import UnloadedRoutedExpert
from scripts.iq_h200_expert_cache import ExpertDiskCache
from scripts.iq_hf_full78_sparse_inference import ExpertStreamer, hydrate_expert


def write_expert(root: Path, physical: int, slot: int) -> str:
    filename = f"model-moe-{physical:03d}-expert-{slot:03d}.safetensors"
    prefix = f"layers.{physical}.moe.routed_experts.{slot}."
    save_file({
        prefix + "gate_proj.weight": torch.ones((3, 4), dtype=torch.bfloat16),
        prefix + "up_proj.weight": torch.ones((3, 4), dtype=torch.bfloat16),
        prefix + "down_proj.weight": torch.ones((4, 3), dtype=torch.bfloat16),
    }, str(root / filename))
    return filename


class FakeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = nn.ModuleList(
            UnloadedRoutedExpert() for _ in range(4)
        )
        self.config = SimpleNamespace(
            latent_size=4, expert_intermediate_size=3,
            situ_beta=4.0, situ_linear_beta=25.0, expert_bias=False,
        )
        self.selected = (1, 3)

    def _route(self, x):
        ids = torch.tensor([list(self.selected)], dtype=torch.long)
        return None, ids, None


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Identity()
        self.moe = FakeMoE()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer()])


class ExpertCacheTests(unittest.TestCase):
    def test_remote_shards_stage_concurrently_and_reuse_local_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [write_expert(root, 0, slot) for slot in (1, 3)]
            with_cache = ExpertDiskCache(root, max_bytes=4096, workers=2)
            try:
                paths = with_cache.stage(names)
                self.assertEqual(set(paths), set(names))
                self.assertEqual(with_cache.copies, 2)
                self.assertEqual(with_cache.hits, 0)
                paths = with_cache.stage(names)
                self.assertEqual(with_cache.copies, 2)
                self.assertEqual(with_cache.hits, 2)
            finally:
                cached_root = with_cache.root
                with_cache.close()
                self.assertFalse(cached_root.exists())

    def test_gpu_cache_reuses_exact_routed_experts_across_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {slot: write_expert(root, 0, slot) for slot in (1, 3)}
            model = FakeModel()
            cache = ExpertStreamer(
                root, model, {0: mapping}, max_active=4,
                gpu_budget_bytes=4096, disk_budget_bytes=4096, stage_workers=2,
            )
            layer = model.layers[0]
            try:
                for forward in range(2):
                    cache.before(0, layer, (torch.ones((1, 4)),))
                    self.assertFalse(isinstance(layer.moe.routed_experts[1], UnloadedRoutedExpert))
                    self.assertFalse(isinstance(layer.moe.routed_experts[3], UnloadedRoutedExpert))
                    cache.after(0, layer, SimpleNamespace(
                        selected_experts=torch.tensor([[1, 3]])
                    ))
                self.assertEqual(cache.gpu_misses, 2)
                self.assertEqual(cache.gpu_hits, 2)
                self.assertEqual(cache.disk.copies, 2)
                self.assertEqual(cache.evictions, 0)
                self.assertEqual(cache.forward_count, 2)
            finally:
                cache.close()
                self.assertIsInstance(layer.moe.routed_experts[1], UnloadedRoutedExpert)
                self.assertIsInstance(layer.moe.routed_experts[3], UnloadedRoutedExpert)

    def test_gpu_eviction_retains_real_routing_and_disk_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {slot: write_expert(root, 0, slot) for slot in (1, 3)}
            size = (root / mapping[1]).stat().st_size
            model = FakeModel()
            cache = ExpertStreamer(
                root, model, {0: mapping}, max_active=4,
                gpu_budget_bytes=size + 8,
                disk_budget_bytes=4096, stage_workers=2,
            )
            layer = model.layers[0]
            try:
                for slot in (1, 3, 1):
                    layer.moe.selected = (slot,)
                    cache.before(0, layer, (torch.ones((1, 4)),))
                    cache.after(0, layer, SimpleNamespace(
                        selected_experts=torch.tensor([[slot]])
                    ))
                self.assertEqual(cache.gpu_misses, 3)
                self.assertGreaterEqual(cache.evictions, 2)
                self.assertEqual(cache.disk.copies, 2)
                self.assertEqual(cache.disk.hits, 1)
            finally:
                cache.close()

    def test_missing_file_fails_and_nonfinite_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = ExpertDiskCache(root, max_bytes=4096)
            try:
                with self.assertRaises(FileNotFoundError):
                    missing.stage(["not-present.safetensors"])
                with self.assertRaises(ValueError):
                    missing.stage(["../escape.safetensors"])
            finally:
                missing.close()
            name = write_expert(root, 0, 1)
            prefix = "layers.0.moe.routed_experts.1."
            save_file({
                prefix + "gate_proj.weight": torch.full((3, 4), float("nan")),
                prefix + "up_proj.weight": torch.ones((3, 4)),
                prefix + "down_proj.weight": torch.ones((4, 3)),
            }, str(root / name))
            with self.assertRaisesRegex(RuntimeError, "Nonfinite selected expert"):
                hydrate_expert(
                    root / name, prefix=prefix, hidden=4, intermediate=3,
                    beta=4.0, linear_beta=25.0,
                    device=torch.device("cpu"), dtype=torch.float32,
                    bias=False,
                )


if __name__ == "__main__":
    unittest.main()
