#!/usr/bin/env python3
"""Independently verify the complete 78-stage calibration from HF read-only.

No bucket writes, no CUDA needed. Downloads only one artifact at a time and
checks the committed 866-file SHA-256 receipt against actual remote bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi

from scripts.assemble_glm53_full78_calibration_local import (
    OUTPUT, PREFIX, RECEIPT, inventory, read,
)
from scripts.glm53_full78_gate import expected_calibration_files


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path.home() / "IQ-local-warm-state" / "verification")
    a = parser.parse_args()
    owner = HfApi().whoami()["name"]
    if owner.casefold() != "kofecheckmate":
        parser.error(f"expected kofecheckmate, signed in as {owner}")
    root = a.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Do not trust listing or SHA in isolation: check strict canonical inventory
    # and every byte from a fresh independent read.
    receipt_path = read(OUTPUT, RECEIPT, root / "audit-receipt.json")
    manifest_path = read(
        OUTPUT, f"{PREFIX}/glm53_calibration.json", root / "audit-manifest.json"
    )
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected = expected_calibration_files(manifest)
    hashes = receipt.get("files")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("source_layers") != 78
        or receipt.get("hash") != "sha256"
        or not isinstance(hashes, dict)
        or set(hashes) != expected
        or len(expected) != 866
    ):
        raise RuntimeError("final receipt/manifest mismatch: refuse compilation")
    remote = inventory(OUTPUT, PREFIX)
    absent = [name for name in sorted(expected)
              if remote.get(f"{PREFIX}/{name}", 0) <= 0]
    if absent:
        raise RuntimeError(f"missing/zero-size canonical artifacts: {absent[:8]}")

    for i, name in enumerate(sorted(expected), 1):
        target_hash = hashes[name]
        if not isinstance(target_hash, str) or len(target_hash) != 64:
            raise RuntimeError(f"malformed SHA256: {name}")
        if name == "glm53_calibration.json":
            actual = digest(manifest_path)
        else:
            with tempfile.TemporaryDirectory(prefix="iq-audit-", dir=str(root)) as tmp:
                actual = digest(read(
                    OUTPUT, f"{PREFIX}/{name}", Path(tmp) / name
                ))
        if actual != target_hash:
            raise RuntimeError(f"INDEPENDENT_FULL78_SHA_MISMATCH: {name}")
        if i % 25 == 0 or i == len(expected):
            print(f"IQ_INDEPENDENT_REMOTE_SHA256={i}/{len(expected)}", flush=True)
    print("IQ_FULL78_INDEPENDENT_CALIBRATION_VERIFIED files=866 layers=78", flush=True)


if __name__ == "__main__":
    main()
