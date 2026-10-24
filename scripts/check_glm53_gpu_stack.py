#!/usr/bin/env python3
"""Read-only dependency, CUDA, and device preflight for GLM-5.3 -> IQ transfer.

No donor downloads, CUDA builds, checkpoint mutations, or HF Jobs are started.
This probes the *active interpreter* in the worker, not an assumed container.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from typing import Any

from packaging.version import InvalidVersion, Version

# Keep the compiler separate from the much heavier Mamba-3 inference runtime.
TRANSFER_REQUIREMENTS = {
    "numpy": ("1.26", None),
    "safetensors": ("0.4.5", None),
    "huggingface-hub": ("0.35", None),
    "requests": ("2.32", "3"),
    "packaging": ("24", None),
    "torch": ("2.10", None),
}
MODEL_REQUIREMENTS = {"transformers": ("5.17.0", "5.17.1")}
OPTIONAL_KERNEL_LIBRARIES = (
    "triton", "tilelang", "apache-tvm-ffi", "quack-kernels",
    "nvidia-cutlass-dsl", "mamba-ssm",
)


def check_packages(
    installed: dict[str, str | None], *, model_runtime: bool = False
) -> list[str]:
    problems: list[str] = []
    needed = dict(TRANSFER_REQUIREMENTS)
    if model_runtime:
        needed.update(MODEL_REQUIREMENTS)
    for package, (minimum, exclusive_maximum) in needed.items():
        found = installed.get(package)
        if found is None:
            problems.append(f"missing required package: {package}")
            continue
        try:
            v = Version(found)
            if v < Version(minimum) or (
                exclusive_maximum is not None and v >= Version(exclusive_maximum)
            ):
                problems.append(
                    f"incompatible {package}={found}; require >= {minimum}"
                    + (f", < {exclusive_maximum}" if exclusive_maximum else "")
                )
        except InvalidVersion:
            problems.append(f"invalid version for {package}: {found}")
    return problems


def check_cuda_device(
    snapshot: dict[str, Any], *, require_cuda: bool, min_vram_gib: float,
    require_bf16: bool,
) -> list[str]:
    problems: list[str] = []
    if not snapshot.get("cuda_available", False):
        if require_cuda:
            problems.append("CUDA unavailable in active PyTorch; refusing CPU fallback")
        return problems
    if not snapshot.get("torch_cuda_build"):
        problems.append("torch.cuda available without reported CUDA build")
    if snapshot.get("sm_major", 0) < 7:
        problems.append("unsupported CUDA compute capability")
    if snapshot.get("vram_gib", 0) < min_vram_gib:
        problems.append(
            f"GPU VRAM {snapshot.get('vram_gib', 0):.2f} GiB < required {min_vram_gib:.2f} GiB"
        )
    if require_bf16 and not snapshot.get("bf16_supported", False):
        problems.append("native BF16 computation unavailable on selected GPU")
    return problems


def package_versions() -> dict[str, str | None]:
    names = {*TRANSFER_REQUIREMENTS, *MODEL_REQUIREMENTS, *OPTIONAL_KERNEL_LIBRARIES}
    result = {}
    for name in sorted(names):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def cuda_snapshot(*, smoke: bool) -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"cuda_available": False, "import_error": str(exc)}
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not result["cuda_available"]:
        return result
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        result.update({
            "device_index": device,
            "gpu_name": props.name,
            "sm_major": int(props.major),
            "sm_minor": int(props.minor),
            "vram_gib": round(props.total_memory / (1024 ** 3), 3),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
        if smoke:
            with torch.inference_mode():
                # Tiny representative FP32 GEMM and BF16 storage round-trip.
                left = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8) / 8
                right = torch.eye(8, device="cuda", dtype=torch.float32)
                product = left @ right
                torch.cuda.synchronize()
                if not torch.allclose(product, left, rtol=0, atol=0):
                    raise RuntimeError("FP32 CUDA GEMM parity failed")
                storage = left.to(torch.bfloat16).to(torch.float32)
                if not torch.allclose(storage, left, rtol=0, atol=0):
                    raise RuntimeError("BF16 storage round-trip failed")
                result["cuda_math_smoke"] = "passed"
    except (RuntimeError, AssertionError) as exc:
        result["cuda_math_smoke"] = "failed"
        result["device_error"] = str(exc)
    return result


def tool_version(executable: str) -> str | None:
    path = shutil.which(executable)
    if not path:
        return None
    try:
        args = ([path, "-V"] if executable == "nvcc" else
                [path, "--query-gpu=driver_version", "--format=csv,noheader"])
        proc = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        return (proc.stdout or proc.stderr).strip()[:500] if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def audit(*, require_cuda: bool, min_vram_gib: float, require_bf16: bool,
          model_runtime: bool, smoke: bool, pip_check: bool) -> dict[str, Any]:
    versions = package_versions()
    device = cuda_snapshot(smoke=smoke)
    errors = check_packages(versions, model_runtime=model_runtime)
    errors.extend(check_cuda_device(device, require_cuda=require_cuda,
                                    min_vram_gib=min_vram_gib, require_bf16=require_bf16))
    if device.get("cuda_math_smoke") == "failed":
        errors.append(f"CUDA math smoke failed: {device.get('device_error')}")
    if require_cuda and platform.system() != "Linux":
        errors.append("GPU transfer worker must run Linux")
    if sys.version_info < (3, 10):
        errors.append("Python >=3.10 required")
    if pip_check:
        p = subprocess.run([sys.executable, "-m", "pip", "check"],
                           capture_output=True, text=True, timeout=90, check=False)
        if p.returncode:
            errors.append(f"pip check failed: {(p.stdout + p.stderr).strip()[:2000]}")
    return {
        "status": "ready" if not errors else "blocked",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "cuda": device,
        "nvcc": tool_version("nvcc"),
        "nvidia_driver": tool_version("nvidia-smi"),
        "requirements": {
            "require_cuda": require_cuda,
            "min_vram_gib": min_vram_gib,
            "require_bf16": require_bf16,
            "model_runtime": model_runtime,
            "smoke": smoke,
        },
        "issues": errors,
        "notes": [
            "BF16 source storage does not require native BF16 GEMM; native BF16 is an optional gate.",
            "Mamba-3 TileLang/CuTe runtime kernels are not exercised by weight-compiler preflight.",
            "Passing preflight is not a benchmark or a model-quality guarantee.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-vram-gib", type=float, default=0.0)
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--model-runtime", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pip-check", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if args.min_vram_gib < 0:
        parser.error("--min-vram-gib must be nonnegative")
    report = audit(require_cuda=args.require_cuda, min_vram_gib=args.min_vram_gib,
                   require_bf16=args.require_bf16, model_runtime=args.model_runtime,
                   smoke=args.smoke, pip_check=args.pip_check)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(output, end="", flush=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output, encoding="utf-8")
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
