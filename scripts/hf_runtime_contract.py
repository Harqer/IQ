"""H200 CUDA/PyTorch environment contract; metadata checks, not a kernel test.

Run inside the actual HF inference process before and after pip installation.
No tensors, model weights, benchmarks, or independent GPU jobs are created.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from importlib import metadata


class RuntimeContractError(RuntimeError):
    """A host-image or installed-package invariant is not satisfied."""


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeContractError(f"Required package {name!r} is not installed") from exc


def _native_stack() -> dict[str, str]:
    import torch
    import triton

    actual_torch = torch.__version__.split("+", 1)[0]
    expected_torch = os.getenv("IQ_EXPECTED_TORCH", "2.10.0")
    if actual_torch != expected_torch:
        raise RuntimeContractError(
            f"Container must provide CUDA-enabled torch {expected_torch}, got {torch.__version__}. "
            "Select the matching PyTorch devel image; do not upgrade Torch through pip."
        )
    actual_cuda = torch.version.cuda
    expected_cuda = os.getenv("IQ_EXPECTED_CUDA", "12.8")
    if actual_cuda != expected_cuda:
        raise RuntimeContractError(
            f"PyTorch CUDA runtime is {actual_cuda!r}, expected {expected_cuda}. "
            "A CPU wheel or the wrong CUDA image cannot serve IQ MIMO."
        )
    if sys.prefix == sys.base_prefix or not os.environ.get("VIRTUAL_ENV"):
        raise RuntimeContractError("Activate a venv (with the container's native torch)")
    return {
        "torch": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "triton": _version("triton"),
        "triton_path": str(Path(triton.__file__).resolve()),
        "torch_cuda": str(actual_cuda),
    }


def _cuda_toolkit_and_gpu() -> dict[str, str]:
    import torch

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            nvcc = str(candidate)
    if nvcc is None:
        raise RuntimeContractError(
            "nvcc not found: use a CUDA *devel* image, not a runtime-only image"
        )
    try:
        output = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeContractError(f"Cannot query nvcc: {exc}") from exc
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        raise RuntimeContractError("Could not parse nvcc --version")
    compiled_cuda = match.group(1)
    if compiled_cuda != torch.version.cuda:
        raise RuntimeContractError(
            f"CUDA compiler is {compiled_cuda}, but PyTorch was built for "
            f"{torch.version.cuda}; correct CUDA_HOME/PATH or select a matching image"
        )
    if not torch.cuda.is_available():
        raise RuntimeContractError(
            "CUDA driver/device unavailable inside the HF job; do not fall back to CPU/SISO"
        )
    gpu = torch.cuda.get_device_name(0)
    required_gpu = os.getenv("IQ_EXPECTED_GPU_NAME", "H200")
    if required_gpu and required_gpu.lower() not in gpu.lower():
        raise RuntimeContractError(f"Expected {required_gpu!r} device, got {gpu!r}")
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeContractError(
            f"IQ H200 image expects Hopper SM90, device is SM{capability[0]}{capability[1]}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeContractError("Device does not advertise required BF16 support")
    arch_list = torch.cuda.get_arch_list()
    if arch_list and not any(
        name.startswith(("sm_90", "compute_90")) for name in arch_list
    ):
        raise RuntimeContractError(
            f"Image PyTorch does not include Hopper SM90 kernels: {arch_list}"
        )
    requested_arches = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    if requested_arches and "9.0" not in requested_arches:
        raise RuntimeContractError(
            f"TORCH_CUDA_ARCH_LIST={requested_arches!r} does not target H200 SM90"
        )
    return {
        "nvcc": compiled_cuda,
        "nvcc_path": str(Path(nvcc).resolve()),
        "gpu": gpu,
        "sm": f"{capability[0]}.{capability[1]}",
    }


def _installed_mamba() -> dict[str, str]:
    from iq_model.state.mamba3 import (
        MAMBA3_UPSTREAM_COMMIT,
        inspect_mamba3_mimo_runtime,
        require_mamba3_mimo_runtime,
    )
    import torch

    for name, expected in (("tilelang", "0.1.8"), ("apache-tvm-ffi", "0.1.9")):
        actual = _version(name)
        if actual != expected:
            raise RuntimeContractError(
                f"Unsupported {name}={actual}; this pinned Mamba-3 requires {expected}"
            )
    info = inspect_mamba3_mimo_runtime(torch.device("cuda:0"))
    require_mamba3_mimo_runtime(torch.device("cuda:0"))
    if info.source_commit != MAMBA3_UPSTREAM_COMMIT:
        raise RuntimeContractError("Mamba-3 source commit differs from IQ's pinned source")
    # A present CuTe import is NOT proof of working recurrent decoding.
    return {
        "mamba_commit": str(info.source_commit),
        "tilelang": _version("tilelang"),
        "tvm_ffi": _version("apache-tvm-ffi"),
        "quack": _version("quack-kernels"),
        "decode_strategy": "full_prefix",
        "decode_note": "CuTe step() remains unverified; retain native rank-4 MIMO full-prefix recomputation",
    }


def _snapshot(path: Path, current: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after", "constraints"))
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        base = _native_stack()
        if args.phase == "constraints":
            print(f"torch=={base['torch']}")
            print(f"triton=={base['triton']}")
            return 0
        extra = _cuda_toolkit_and_gpu()
        if args.phase == "before":
            if args.snapshot is None:
                raise RuntimeContractError("--snapshot is required for 'before'")
            _snapshot(args.snapshot, base)
        else:
            if args.snapshot is None or not args.snapshot.is_file():
                raise RuntimeContractError(
                    "Missing pre-install snapshot; the original CUDA Torch/Triton build "
                    "must be captured before installing IQ packages"
                )
            original = json.loads(args.snapshot.read_text())
            if base != original:
                raise RuntimeContractError(
                    "pip modified the native PyTorch/Triton stack or import resolution. "
                    f"Before: {original}; after: {base}"
                )
            extra.update(_installed_mamba())
        print(json.dumps({"phase": args.phase, **base, **extra}, sort_keys=True))
    except (RuntimeContractError, ImportError, OSError, ValueError) as exc:
        print(f"IQ H200 RUNTIME BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
