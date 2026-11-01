"""Reproducible, fail-closed dependency compatibility patch for pinned Mamba3.

The model code remains byte-for-byte upstream at the verified commit; only
the package's runtime dependency declarations are changed. This addresses
an independently reproduced TileLang 0.1.8 / TVM FFI 0.1.9 / CuTe 4.8.1
incompatibility. H200 MUST still pass native prefill+step correctness.
"""
from __future__ import annotations

import argparse
from pathlib import Path

UPSTREAM_TILELANG = '"tilelang==0.1.8"'
UPSTREAM_TVM = '"apache-tvm-ffi<=0.1.12"'
IQ_TILELANG = '"tilelang==0.1.12"'
IQ_TVM = '"apache-tvm-ffi==0.1.11"'
TARGETS = ("pyproject.toml", "setup.py")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    print(f"IQ_MAMBA3_METADATA_PATCH_APPLIED={','.join(apply_patch(args.checkout))}",
          flush=True)


if __name__ == "__main__":
    main()
