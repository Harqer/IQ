"""CPU-only contract tests for genuine IQ transplant geometry and overlay mapping.

These tests use tiny fixture tensors strictly to validate loading mechanics;
production weights are fetched/converted only by the pinned donor CLI.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from iq_model import HybridLayerType, HybridSchedule, Mamba3MIMOConfig
from iq_transfer.mamba3_direct import (
    MAMBA3_MIMO_15B_BIN_SHA256,
    MAMBA3_MIMO_15B_REVISION,
    Mamba3DirectTransferError,
    apply_mamba3_transplant_overlay,
    evenly_spaced_layer_placements,
)
from iq_transfer.complete_transplant import nvidia_parallel_78_config


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = nn.Linear(2, 2, bias=False)


class _Mamba(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _Core()


class _ParallelLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2)
        self.mamba = _Mamba()


class _TinyParallelRecipient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_ParallelLayer() for _ in range(78))
        self.config = SimpleNamespace(
            schedule=HybridSchedule(tuple(
                HybridLayerType.PARALLEL_CSA if i % 2 == 0
                else HybridLayerType.PARALLEL_HCA
                for i in range(78)
            )),
            mamba3=Mamba3MIMOConfig(
                d_model=2, num_layers=78, d_state=2, headdim=2,
                mimo_rank=2, expand=1.0, rope_fraction=1.0,
                chunk_size=1,
            ),
        )
        self.model_config = SimpleNamespace(rms_norm_eps=1e-5)


def _write_fixture(root: Path, model: nn.Module, *, corrupt: bool = False) -> tuple[int, ...]:
    from safetensors.torch import save_file

    placements = evenly_spaced_layer_placements(24, 78)
    for source_layer, ordinal in enumerate(placements):
        filename = root / "mamba" / f"layer-{ordinal:03d}.safetensors"
        filename.parent.mkdir(parents=True, exist_ok=True)
        save_file({
            f"mamba_layers.{ordinal}.norm.weight": (
                torch.full((2,), 1.0 + source_layer / 100)
            )
        }, str(filename))
    identity = sorted(set(range(78)) - set(placements))
    manifest = {
        "artifact_type": "iq_mamba3_foundation",
        "target": {
            "mamba3": asdict(model.config.mamba3),
            "rms_norm_eps": 1e-5,
        },
        "donor": {
            "revision": MAMBA3_MIMO_15B_REVISION,
            "checkpoint_sha256": MAMBA3_MIMO_15B_BIN_SHA256,
        },
        "placements": [
            {
                "source_layer": i,
                "target_mamba_ordinal": ordinal,
                "shard": f"mamba/layer-{ordinal:03d}.safetensors",
            }
            for i, ordinal in enumerate(placements)
        ],
        "identity_mamba_ordinals": identity if not corrupt else identity[:-1],
    }
    (root / "mamba3_transplant.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return placements


class Mamba78ParallelTransplantTests(unittest.TestCase):
    def test_depth_78_has_24_pretrained_and_54_residual_neutral_mamba_slots(self):
        placements = evenly_spaced_layer_placements(24, 78)
        self.assertEqual(len(placements), 24)
        self.assertEqual(placements[0], 0)
        self.assertEqual(placements[-1], 77)
        self.assertEqual(len(set(placements)), 24)
        self.assertEqual(len(set(range(78)) - set(placements)), 54)

    def test_target_matches_actual_nvidia_parallel_config(self):
        config = nvidia_parallel_78_config()
        self.assertEqual(config.schedule.mamba_positions, tuple(range(78)))
        self.assertEqual(len(config.schedule.parallel_positions), 78)
        self.assertEqual(config.mamba3.num_layers, 78)

    def test_compiled_foundation_maps_to_fused_block_parameter_names(self):
        try:
            import safetensors
        except ImportError:
            self.skipTest("safetensors is required")
        model = _TinyParallelRecipient()
        with tempfile.TemporaryDirectory() as tmp:
            placements = _write_fixture(Path(tmp), model)
            copied = apply_mamba3_transplant_overlay(model, Path(tmp))
        self.assertEqual(len(copied.copied_parameters), 24)
        self.assertEqual(len(copied.identity_parameters), 54)
        self.assertEqual(
            set(copied.copied_parameters),
            {f"layers.{i}.norm.weight" for i in placements},
        )
        self.assertTrue(all(
            torch.count_nonzero(model.layers[i].mamba.core.out_proj.weight) == 0
            for i in range(78) if i not in placements
        ))
        self.assertEqual(
            model.layers[placements[0]].norm.weight.tolist(), [1.0, 1.0]
        )

    def test_corrupt_slot_partition_is_rejected_before_parameter_mutation(self):
        try:
            import safetensors
        except ImportError:
            self.skipTest("safetensors is required")
        model = _TinyParallelRecipient()
        previous = model.layers[0].norm.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), model, corrupt=True)
            with self.assertRaisesRegex(Mamba3DirectTransferError, "partition"):
                apply_mamba3_transplant_overlay(model, Path(tmp))
        self.assertTrue(torch.equal(previous, model.layers[0].norm.weight))

    def test_original_32_ordinal_mapping_is_still_supported(self):
        original = evenly_spaced_layer_placements(24, 32)
        self.assertEqual(len(original), 24)
        self.assertEqual(len(set(range(32)) - set(original)), 8)


if __name__ == "__main__":
    unittest.main()
