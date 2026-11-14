"""No-GPU tests for hardware-driven Google Colab preflight routing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import iq_colab_runtime_probe as probe


class ColabProbeTests(unittest.TestCase):
    def test_real_tpu_devices_take_precedence(self) -> None:
        tpu = [{"platform": "tpu", "device_kind": "TPU v6e", "id": 0}]
        gpu = [{"name": "NVIDIA T4", "memory_bytes": 16_000_000_000}]
        self.assertEqual(probe.select_backend(tpu, gpu), "tpu")

    def test_cuda_device_is_selected_without_tpu(self) -> None:
        self.assertEqual(
            probe.select_backend([], [{"name": "NVIDIA A100"}]),
            "cuda",
        )

    def test_cpu_fallback_is_only_used_without_accelerator(self) -> None:
        self.assertEqual(probe.select_backend([], [], tpu_hint=False), "cpu")

    def test_failed_tpu_initialization_not_misclassified_as_cpu(self) -> None:
        self.assertEqual(probe.select_backend([], [], tpu_hint=True), "tpu_unavailable")

    def test_tpu_never_attempts_cuda_mamba3_smoke(self) -> None:
        fake_tpu = [{"platform": "tpu", "device_kind": "TPU v5e", "id": 0}]
        with (
            patch.object(probe, "_tpu_devices", return_value=(fake_tpu, None)),
            patch.object(probe, "_nvidia_devices", side_effect=AssertionError("GPU probe should not run")),
            patch.object(probe, "_smoke_cuda", side_effect=AssertionError("CUDA smoke on TPU")),
            patch.object(probe, "_smoke_tpu", return_value={"passed": True, "kind": "tiny_tpu_test"}),
        ):
            result = probe.run(smoke=True)
        self.assertEqual(result["runtime_backend"], "tpu")
        self.assertTrue(result["smoke"]["passed"])
        self.assertFalse(result["model_inference_executed"])

    def test_cuda_smoke_runs_only_on_nvidia_branch(self) -> None:
        nvidia = [{"index": 0, "name": "NVIDIA T4", "memory_bytes": 16_000_000_000}]
        with (
            patch.object(probe, "_tpu_devices", return_value=([], None)),
            patch.object(probe, "_nvidia_devices", return_value=(nvidia, None)),
            patch.object(probe, "_tpu_hint_present", return_value=False),
            patch.object(probe, "_probe_mamba3_dependency", return_value={"mimo_combined_present": False}),
            patch.object(probe, "_smoke_cuda", return_value={"passed": True, "kind": "tiny_cuda_test"}),
            patch.object(probe, "_smoke_tpu", side_effect=AssertionError("TPU smoke on CUDA")),
        ):
            result = probe.run(smoke=True)
        self.assertEqual(result["runtime_backend"], "cuda")
        self.assertEqual(result["smoke"]["kind"], "tiny_cuda_test")

    def test_full256_map_does_not_fit_single_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "map.json"
            file.write_text(json.dumps({
                "device_count": 8,
                "memory_reserve_GB": 20,
                "placements": [{"gpu": i, "weight_GB": 68.9} for i in range(8)],
            }))
            gpus = [{"index": 0, "memory_bytes": 80_000_000_000}]
            report = probe._device_map_fit(file, "cuda", gpus)
            self.assertFalse(report["gpu_placement_capacity"])
            self.assertIn("Not enough", report["reason"])

    def test_tpu_does_not_claim_cuda_checkpoint_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "map.json"
            file.write_text(json.dumps({
                "device_count": 8,
                "placements": [{"gpu": i, "weight_GB": 60} for i in range(8)],
            }))
            report = probe._device_map_fit(file, "tpu", [])
            self.assertFalse(report["gpu_placement_capacity"])
            self.assertIn("CUDA", report["reason"])

    def test_hardware_kind_is_not_inferred_from_cli_flag(self) -> None:
        tpu = [{"platform": "tpu", "device_kind": "TPU v7", "id": 0}]
        self.assertEqual(probe.select_backend(tpu, []), "tpu")


if __name__ == "__main__":
    unittest.main()
