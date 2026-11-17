"""Fail-closed guard for pinned upstream Mamba-3 MIMO short-chunk V reads.

The official e9594c... TileLang forward source reads V beyond the valid
sequence for short/incomplete chunks. This script patches ONLY that exact
source expression. Run on a clone of the pinned upstream source before
building mamba_ssm, never as a silent inference-time edit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

KERNEL_RELATIVE_PATH = "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py"
UPSTREAM_V_READ = (
    "                for cs, p in T.Parallel(chunk_size, P):\n"
    "                    v_shared[cs, p] = V[i_b, chunk_start+cs, i_h, p]"
)
SAFE_V_READ = (
    "                for cs, p in T.Parallel(chunk_size, P):\n"
    "                    if chunk_start + cs < S:\n"
    "                        v_shared[cs, p] = V[i_b, chunk_start+cs, i_h, p]\n"
    "                    else:\n"
    "                        v_shared[cs, p] = 0.0"
)


class NativeMIMOKernelSafetyError(RuntimeError):
    pass


def kernel_tail_guard_present(source: str) -> bool:
    return source.count(SAFE_V_READ) == 1 and UPSTREAM_V_READ not in source


def verify_kernel_guard(path: Path) -> None:
    if not path.is_file():
        raise NativeMIMOKernelSafetyError(f"native MIMO kernel is missing: {path}")
    source = path.read_text(encoding="utf-8")
    if not kernel_tail_guard_present(source):
        raise NativeMIMOKernelSafetyError(
            "native MIMO fixed-length TileLang V-read tail guard absent or changed; "
            "refuse CUDA inference until patched, pinned source is installed"
        )


def patch_kernel(root: Path) -> Path:
    path = root / KERNEL_RELATIVE_PATH
    source = path.read_text(encoding="utf-8")
    if source.count(UPSTREAM_V_READ) != 1 or SAFE_V_READ in source:
        raise NativeMIMOKernelSafetyError(
            "unrecognized or already patched Mamba-3 source; refusing edits"
        )
    result = source.replace(UPSTREAM_V_READ, SAFE_V_READ)
    path.write_text(result, encoding="utf-8")
    verify_kernel_guard(path)
    return path


def check_installed_kernel_guard() -> None:
    """Read-only check before importing or invoking the original CUDA model."""
    from importlib import util
    spec = util.find_spec("mamba_ssm")
    if spec is None or not spec.submodule_search_locations:
        raise NativeMIMOKernelSafetyError("native mamba_ssm package not installed")
    root = Path(next(iter(spec.submodule_search_locations))).parent
    verify_kernel_guard(root / KERNEL_RELATIVE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patch-checkout", type=Path,
                       help="Apply exact one-line native kernel safety patch to a pinned source clone")
    group.add_argument("--check-installed", action="store_true",
                       help="Read-only verify installed native kernel contains the tail guard")
    args = parser.parse_args()
    if args.check_installed:
        check_installed_kernel_guard()
        print("IQ_NATIVE_MIMO_TAIL_GUARD_OK")
    else:
        print(f"IQ_NATIVE_MIMO_TAIL_PATCHED={patch_kernel(args.patch_checkout)}")


if __name__ == "__main__":
    main()
