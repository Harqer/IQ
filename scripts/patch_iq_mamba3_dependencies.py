"""Reproducible compatibility and short-chunk CUDA safety patch for pinned Mamba3.

Preserve the exact upstream revision, pinned dependency compatibility and
native rank-4 MIMO architecture. The only kernel modification is an explicit
boundary guard on the V tile's scalar global-memory load in the fixed-length
TileLang forward kernel. The original code reads cs=13..15 for a 13-token
input when chunk_size=16; unlike T.copy, this scalar read is not automatically
masked. Require SHA-verified installed kernel bytes and native CUDA parity.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path

UPSTREAM_TILELANG = '"tilelang==0.1.8"'
UPSTREAM_TVM = '"apache-tvm-ffi<=0.1.12"'
IQ_TILELANG = '"tilelang==0.1.12"'
IQ_TVM = '"apache-tvm-ffi==0.1.11"'
TARGETS = ("pyproject.toml", "setup.py")
KERNEL_RELATIVE_PATH = "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py"
KERNEL_PATCH = "mimo-fwd-valid-length-guard-v1"
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



def apply_patch(root: Path) -> list[str]:
    """Edit only exact known dependency literals; fail on upstream drift."""
    originals: dict[Path, str] = {}
    patched: dict[Path, str] = {}
    for name in TARGETS:
        path = root / name
        value = path.read_text(encoding="utf-8")
        for expected in (UPSTREAM_TILELANG, UPSTREAM_TVM):
            if value.count(expected) != 1:
                raise RuntimeError(
                    f"Unrecognized Mamba3 upstream dependency in {name}: "
                    f"expected exactly one {expected}"
                )
        if IQ_TILELANG in value or IQ_TVM in value:
            raise RuntimeError(f"Unexpected pre-patched Mamba3 file: {name}")
        originals[path] = value
        patched[path] = (
            value.replace(UPSTREAM_TILELANG, IQ_TILELANG)
            .replace(UPSTREAM_TVM, IQ_TVM)
        )
    # Validate every file before writing any file.
    for path, content in patched.items():
        path.write_text(content, encoding="utf-8")
    return [p.name for p in patched]


def apply_kernel_tail_guard(root: Path) -> str:
    """Fix the exact unguarded input load; reject any source drift."""
    path = root / KERNEL_RELATIVE_PATH
    original = path.read_text(encoding="utf-8")
    if original.count(UPSTREAM_V_READ) != 1 or SAFE_V_READ in original:
        raise RuntimeError(
            "Unexpected pinned upstream Mamba3 MIMO fixed-kernel V read: "
            "cannot apply exact bounds patch safely"
        )
    path.write_text(original.replace(UPSTREAM_V_READ, SAFE_V_READ), encoding="utf-8")
    return KERNEL_RELATIVE_PATH


def record_installed(root: Path, source_commit: str) -> Path:
    """Record SHA-checked unmodified Mamba mixer implementation in installed venv.

    The metadata-only source patch means pip emits a local-path direct_url
    rather than a Git VCS SHA. This separate provenance record keeps IQ's
    original upstream-commit verification active without pretending a local
    path is an unchanged VCS wheel.
    """
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    source = root / "mamba_ssm" / "modules" / "mamba3.py"
    if not source.is_file():
        raise FileNotFoundError("Pinned upstream Mamba-3 module missing")
    dist = metadata.distribution("mamba_ssm")
    installed = Path(dist.locate_file("mamba_ssm/modules/mamba3.py"))
    if not installed.is_file():
        raise FileNotFoundError("Installed Mamba-3 module missing")
    expected = sha256(source.read_bytes()).hexdigest()
    actual = sha256(installed.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("Installed Mamba-3 source differs from pinned checkout")
    kernel_source = root / KERNEL_RELATIVE_PATH
    kernel_installed = Path(dist.locate_file(KERNEL_RELATIVE_PATH))
    if not kernel_source.is_file() or not kernel_installed.is_file():
        raise FileNotFoundError("Installed pinned MIMO TileLang kernel missing")
    original_kernel_text = kernel_source.read_text(encoding="utf-8")
    if (original_kernel_text.count(SAFE_V_READ) != 1
            or UPSTREAM_V_READ in original_kernel_text):
        raise RuntimeError("MIMO tail guard is not present in pinned checkout")
    source_kernel_sha = sha256(kernel_source.read_bytes()).hexdigest()
    installed_kernel_sha = sha256(kernel_installed.read_bytes()).hexdigest()
    if source_kernel_sha != installed_kernel_sha:
        raise RuntimeError("Installed MIMO TileLang kernel differs from patched source")
    provenance = Path(dist.locate_file("mamba_ssm/iq_mamba3_provenance.json"))
    if not provenance.parent.is_dir():
        raise FileNotFoundError("Installed Mamba-3 package directory missing")
    provenance.write_text(json.dumps({
        "source_commit": source_commit,
        "module_sha256": expected,
        "kernel_sha256": source_kernel_sha,
        "kernel_patch": KERNEL_PATCH,
        "metadata_patch": "tilelang0.1.12-tvmffi0.1.11",
    }, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, nargs="?")
    parser.add_argument("--record-installed", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.record_installed is not None:
        if args.checkout is not None or args.source_commit is None:
            parser.error("--record-installed requires --source-commit and no checkout")
        file = record_installed(args.record_installed, args.source_commit)
        print(f"IQ_MAMBA3_PATCHED_PROVENANCE_RECORDED={file}", flush=True)
    else:
        if args.checkout is None or args.source_commit is not None:
            parser.error("provide checkout, or --record-installed with --source-commit")
        patched = apply_patch(args.checkout)
        patched.append(apply_kernel_tail_guard(args.checkout))
        print(f"IQ_MAMBA3_COMPAT_PATCH_APPLIED={','.join(patched)}",
              flush=True)


if __name__ == "__main__":
    main()
