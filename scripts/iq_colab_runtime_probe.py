"""Auto-detect an attached Google Colab CPU, NVIDIA GPU, or TPU runtime.

No installations, model weight downloads, TPU provisioning, paid GPU allocations,
or IQ inference. When --smoke is passed, test only tiny synthetic arrays.

Google Colab CLI 2026 accepts --tpu=v5e1|v6e1 and
--gpu=T4|L4|G4|A100|H100. Those names are *requested* accelerator labels,
not a reliable means of detecting the hardware actually attached.
Google documents checking JAX device.platform == 'tpu' for TPU detection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:320]}"


def _ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _tpu_hint_present() -> bool:
    # A hint is NOT proof of an initialized TPU device.
    return any(os.getenv(k) for k in (
        "TPU_NAME", "COLAB_TPU_ADDR", "TPU_ACCELERATOR_TYPE",
        "TPU_WORKER_ID", "TPU_HOST_BOUNDS",
    ))


def _tpu_devices() -> tuple[list[dict[str, Any]], str | None]:
    try:
        import jax
    except (ImportError, OSError) as exc:
        return [], _error(exc)
    try:
        # The actual backend device.platform, not a guessed TPU model name.
        devices = jax.devices("tpu")
    except Exception as exc:
        return [], _error(exc)
    result = []
    for d in devices:
        if getattr(d, "platform", None) != "tpu":
            continue
        result.append({
            "platform": d.platform,
            "device_kind": str(getattr(d, "device_kind", "unknown")),
            "platform_version": str(getattr(d, "platform_version", "unknown")),
            "process_index": int(getattr(d, "process_index", 0)),
            "id": int(getattr(d, "id", 0)),
        })
    return result, None


def _nvidia_devices() -> tuple[list[dict[str, Any]], str | None]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return [], _error(exc)
    try:
        if not torch.cuda.is_available():
            return [], None
        result = []
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            major, minor = torch.cuda.get_device_capability(i)
            result.append({
                "index": i,
                "name": str(prop.name),
                "memory_bytes": int(prop.total_memory),
                "compute_capability": f"{major}.{minor}",
                "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
            })
        return result, None
    except (RuntimeError, AssertionError) as exc:
        return [], _error(exc)


def select_backend(
    tpu_devices: list[dict[str, Any]],
    nvidia_devices: list[dict[str, Any]],
    *,
    tpu_hint: bool = False,
) -> str:
    """Pure, testable decision tree; hardware wins over naming guesses."""
    if tpu_devices:
        return "tpu"
    elif nvidia_devices:
        return "cuda"
    elif tpu_hint:
        return "tpu_unavailable"
    return "cpu"


def _probe_mamba3_dependency() -> dict[str, Any]:
    # Import-only prerequisite check; never pretend this proves kernel execution.
    try:
        from mamba_ssm.modules import mamba3
        return {
            "upstream_mamba3_imported": True,
            "mimo_combined_present": getattr(mamba3, "mamba3_mimo_combined", None) is not None,
            "mimo_executed": False,
        }
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        return {
            "upstream_mamba3_imported": False,
            "mimo_combined_present": False,
            "mimo_executed": False,
            "dependency_error": _error(exc),
        }


def _smoke_tpu() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    d = next(d for d in jax.devices("tpu") if d.platform == "tpu")
    a = jax.device_put(jnp.ones((4, 4), dtype=jnp.float32), d)
    out = (a @ a).block_until_ready()
    ok = out.device.platform == "tpu" and bool(jnp.all(out == 4))
    return {
        "kind": "synthetic_tpu_jax_matmul",
        "passed": ok,
        "iq_mamba3_executed": False,
    }


def _smoke_cuda() -> dict[str, Any]:
    import torch
    device = torch.device("cuda:0")
    bf16 = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else torch.float32
    a = torch.ones((4, 4), device=device, dtype=dtype)
    out = a @ a
    torch.cuda.synchronize(device)
    ok = out.device.type == "cuda" and bool((out.float() == 4).all())
    return {
        "kind": "synthetic_cuda_matmul",
        "dtype": str(dtype),
        "passed": ok,
        "iq_mamba3_executed": False,
    }


def _smoke_cpu() -> dict[str, Any]:
    return {
        "kind": "cpu_metadata_only",
        "passed": True,
        "iq_mamba3_executed": False,
    }


def _device_map_fit(plan_path: Path, backend: str, gpu_devices: list[dict[str, Any]]) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    required = plan["device_count"]
    result = {
        "required_cuda_devices": required,
        "available_cuda_devices": len(gpu_devices),
        "gpu_placement_capacity": False,
        "inference_ready": False,
    }
    if backend != "cuda":
        result["reason"] = "IQ's pinned Mamba-3 MIMO runtime requires CUDA; TPU is not a substitute"
        return result
    if len(gpu_devices) < required:
        result["reason"] = "Not enough NVIDIA GPUs for the full256 device map"
        return result
    for slot in plan["placements"]:
        i = slot["gpu"]
        minimum_gb = slot["weight_GB"] + float(plan.get("memory_reserve_GB", 20))
        if gpu_devices[i]["memory_bytes"] / 1e9 < minimum_gb:
            result["reason"] = f"cuda:{i} lacks planned weight + reserve capacity"
            return result
    result["gpu_placement_capacity"] = True
    result["reason"] = "Nominal VRAM capacity only; CUDA kernel, real loading and forward remain unverified"
    return result


def run(
    *,
    smoke: bool = False,
    device_map: Path | None = None,
    checkpoint: Path | None = None,
    tokenizer_dir: Path | None = None,
) -> dict[str, Any]:
    tpu, tpu_error = _tpu_devices()
    gpu, gpu_error = _nvidia_devices() if not tpu else ([], None)
    backend = select_backend(tpu, gpu, tpu_hint=_tpu_hint_present())
    result: dict[str, Any] = {
        "runtime_backend": backend,
        "tpu": {"devices": tpu, "initialization_error": tpu_error},
        "nvidia_gpu": {"devices": gpu, "initialization_error": gpu_error},
        "system_ram_bytes": _ram_bytes(),
        "colab_runtime_name_inferred": False,
        "weights_loaded": False,
        "model_inference_executed": False,
        "warnings": [],
    }

    if backend == "tpu":
        result["checks"] = ["JAX TPU backend available", "metadata preflight if mounted"]
        result["warnings"].append(
            "IQ's existing Mamba-3 MIMO TileLang/CUDA kernel does not run on TPU; "
            "do not switch to CPU/SISO or rewrite IQ architecture"
        )
        if smoke:
            try:
                result["smoke"] = _smoke_tpu()
            except Exception as exc:
                result["smoke"] = {"kind": "synthetic_tpu_jax_matmul", "passed": False, "error": _error(exc)}
    elif backend == "cuda":
        result["checks"] = [
            "NVIDIA CUDA accessible", "Mamba-3 MIMO dependency import",
            "metadata preflight if mounted",
        ]
        result["mamba3_dependency"] = _probe_mamba3_dependency()
        if smoke:
            try:
                result["smoke"] = _smoke_cuda()
            except Exception as exc:
                result["smoke"] = {"kind": "synthetic_cuda_matmul", "passed": False, "error": _error(exc)}
    elif backend == "tpu_unavailable":
        result["checks"] = ["diagnose TPU initialization before any accelerator work"]
        result["warnings"].append(
            "TPU environment hints detected, but no JAX TPU device initialized. "
            "Do not silently classify this as usable CPU/TPU inference."
        )
    else:
        result["checks"] = ["CPU-only checkpoint metadata preflight"]
        if smoke:
            result["smoke"] = _smoke_cpu()

    if device_map is not None:
        result["device_map_capacity"] = _device_map_fit(device_map, backend, gpu)
    if checkpoint is not None:
        if device_map is None:
            result["warnings"].append("--checkpoint requires --device-map to run metadata preflight")
        else:
            from check_iq_device_map import run as metadata_run
            result["metadata_preflight"] = metadata_run(checkpoint, device_map, tokenizer_dir)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="Tiny JAX TPU / PyTorch CUDA test; no model weights")
    p.add_argument("--device-map", type=Path, help="Existing hybrid JSON; optional")
    p.add_argument("--checkpoint", type=Path, help="Mounted checkpoint; metadata scan only")
    p.add_argument("--tokenizer-dir", type=Path, help="Optional existing tokenizer mount")
    args = p.parse_args()
    try:
        report = run(smoke=args.smoke, device_map=args.device_map, checkpoint=args.checkpoint, tokenizer_dir=args.tokenizer_dir)
    except (OSError, ValueError, KeyError, ImportError, json.JSONDecodeError) as exc:
        report = {"runtime_backend": "undetermined", "preflight_error": _error(exc), "model_inference_executed": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    if "preflight_error" in report:
        return 1
    if report.get("metadata_preflight") and not report["metadata_preflight"]["metadata_passed"]:
        return 1
    if report.get("smoke") and not report["smoke"]["passed"]:
        return 1
    if report["runtime_backend"] == "tpu_unavailable":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
