"""Reproducible, fail-closed dependency compatibility patch for pinned Mamba3.

The model code remains byte-for-byte upstream at the verified commit; only
the package's runtime dependency declarations are changed. This addresses
an independently reproduced TileLang 0.1.8 / TVM FFI 0.1.9 / CuTe 4.8.1
incompatibility. H200 MUST still pass native prefill+step correctness.
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
    provenance = Path(dist.locate_file("mamba_ssm/iq_mamba3_provenance.json"))
    if not provenance.parent.is_dir():
        raise FileNotFoundError("Installed Mamba-3 package directory missing")
    provenance.write_text(json.dumps({
        "source_commit": source_commit,
        "module_sha256": expected,
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
        print(f"IQ_MAMBA3_METADATA_PATCH_APPLIED={','.join(apply_patch(args.checkout))}",
              flush=True)


if __name__ == "__main__":
    main()
