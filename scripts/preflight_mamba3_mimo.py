"""Compile and execute the pinned Mamba-3 MIMO TileLang CUDA forward kernel.

Run only on a CUDA system, after activating the IQ virtual environment
and installing dependencies via scripts/install_model_runtime.sh.
An import-only smoke test is not sufficient to catch TileLang/TVM-FFI errors.
"""

from __future__ import annotations

from importlib.metadata import version

import torch

from iq_model import Mamba3MIMOConfig, Mamba3MIMOState, require_mamba3_mimo_runtime


def main() -> None:
    expected_versions = {"tilelang": "0.1.8", "apache-tvm-ffi": "0.1.9"}
    for package, expected in expected_versions.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(
                f"Unsupported {package}=={actual}; required {package}=={expected}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA kernel preflight requires a CUDA GPU")

    device = torch.device("cuda:0")
    require_mamba3_mimo_runtime(device)

    model = Mamba3MIMOState(
        Mamba3MIMOConfig(
            d_model=4096,
            num_layers=32,
            d_state=128,
            headdim=64,
            mimo_rank=4,
            expand=2.0,
            rope_fraction=0.5,
            chunk_size=16,
            outproj_norm=False,
        ),
        layer_idx=0,
        dtype=torch.bfloat16,
        device=device,
    ).eval()

    inputs = torch.randn((1, 16, 4096), dtype=torch.bfloat16, device=device)
    with torch.inference_mode():
        output = model(inputs)
        torch.cuda.synchronize(device)

    if output.shape != inputs.shape or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Mamba-3 MIMO CUDA preflight returned invalid outputs")

    print(
        "Mamba-3 MIMO TileLang kernel PASS:",
        {
            "gpu": torch.cuda.get_device_name(device),
            "shape": list(output.shape),
            "tilelang": version("tilelang"),
            "apache-tvm-ffi": version("apache-tvm-ffi"),
        },
    )


if __name__ == "__main__":
    main()
