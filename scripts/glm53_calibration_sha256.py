#!/usr/bin/env python3
"""Hash every expected WARM artifact; verify remotely from a DIFFERENT job.

Run 'make' on the local COMPLETE calibration before publishing. Then upload.
Run 'verify' in an independent read-only worker after cloud sync. Never trust
a job's exit status or an unflushed FUSE mount as confirmation of persistence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.glm53_full78_gate import (
    GateFailure, check_calibration, expected_calibration_files, _load_json,
)

RECEIPT = "calibration_sha256.json"
CHUNK_SIZE = 4 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            data = stream.read(CHUNK_SIZE)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def receipt(root: Path) -> dict[str, str]:
    status = check_calibration(root)
    if not status["ready"]:
        raise GateFailure(
            f"incomplete calibration: missing={status['missing_count']} "
            f"invalid={status['invalid_count']}"
        )
    manifest = _load_json(root / "glm53_calibration.json")
    expected = sorted(expected_calibration_files(manifest))
    return {name: digest(root / name) for name in expected}


def write_receipt(root: Path) -> int:
    target = root / RECEIPT
    if target.exists() or target.is_symlink():
        raise GateFailure("receipt already exists; refuse to overwrite historical proof")
    records = receipt(root)
    body = {"schema_version": 1, "source_layers": 78,
            "hash": "sha256", "files": records}
    temporary = root / (RECEIPT + ".partial")
    if temporary.exists():
        raise GateFailure("unfinished receipt exists; audit before retry")
    try:
        with temporary.open("x", encoding="utf-8") as out:
            json.dump(body, out, sort_keys=True, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(records)


def verify_receipt(root: Path) -> int:
    body = _load_json(root / RECEIPT)
    if body.get("schema_version") != 1 or body.get("source_layers") != 78 or body.get("hash") != "sha256":
        raise GateFailure("wrong receipt schema")
    expected = expected_calibration_files(_load_json(root / "glm53_calibration.json"))
    hashes = body.get("files")
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise GateFailure("receipt file inventory does not match GLM manifest")
    report = check_calibration(root)
    if not report["ready"]:
        raise GateFailure("some calibration artifacts are absent from remote bucket")
    for idx, name in enumerate(sorted(expected), 1):
        if not isinstance(hashes[name], str) or digest(root / name) != hashes[name]:
            raise GateFailure(f"CALIBRATION_HASH_MISMATCH: {name}")
        if idx % 100 == 0:
            print(f"CALIBRATION_REMOTE_HASH_PROGRESS={idx}/{len(expected)}", flush=True)
    return len(expected)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("make", "verify"))
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    try:
        count = write_receipt(args.root) if args.mode == "make" else verify_receipt(args.root)
        print(f"FULL78_CALIBRATION_{args.mode.upper()}_VERIFIED files={count}", flush=True)
        return 0
    except (GateFailure, OSError) as exc:
        p.exit(3, f"BLOCKED_{args.mode.upper()}_CALIBRATION: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
