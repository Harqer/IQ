#!/usr/bin/env python3
"""Read-only independent Hub bucket inventory of a full78 IQ checkpoint.

Checks the authoritative HF bucket API, NEVER an NFS/FUSE directory listing.
Run before resuming a legacy asynchronous-mount checkpoint and after finish.
Does not rename, remove, or upload anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from huggingface_hub import get_bucket_paths_info
from scripts.glm53_full78_gate import expected_stage_shards

BUCKET = "kofecheckmate/iq-glm53-all256"


def expected_files() -> set[str]:
    files = set()
    for stage in range(78):
        files.update(expected_stage_shards(stage))
    files.update(f"model-mamba-{n:03d}.safetensors" for n in range(79))
    files.update((
        "model-global.safetensors", "model-native.safetensors",
        "iq_config.json", "run-provenance.json",
    ))
    return files


def audit(run_id: str, *, expect_missing: set[str]) -> dict:
    if not re.fullmatch(r"glm53-full78-[a-z0-9._-]+", run_id):
        raise ValueError("invalid IQ run id")
    prefix = f"IQ/runs/{run_id}/"
    expected = expected_files()
    info = {
        obj.path.removeprefix(prefix): obj
        for obj in get_bucket_paths_info(BUCKET, [prefix + n for n in sorted(expected)])
    }
    missing = sorted(set(expected) - set(info))
    bad = sorted(
        n for n, item in info.items()
        if item.size < (16 if n.endswith(".safetensors") else 1)
    )
    print(f"IQ_REMOTE_EXPECTED={len(expected)}", flush=True)
    print(f"IQ_REMOTE_PRESENT={len(info)}", flush=True)
    print(f"IQ_REMOTE_MISSING={len(missing)} {missing[:25]}", flush=True)
    print(f"IQ_REMOTE_INVALID_SIZES={len(bad)} {bad[:25]}", flush=True)
    if set(missing) != expect_missing or bad:
        raise RuntimeError(
            "Remote checkpoint inventory does not match expected missing "
            f"set {sorted(expect_missing)}. Do not claim completion."
        )
    if expect_missing:
        print("IQ_REMOTE_INVENTORY_READY_FOR_TARGETED_RESUME", flush=True)
    else:
        print("IQ_REMOTE_INVENTORY_COMPLETE_METADATA_ONLY", flush=True)
    return {"expected": len(expected), "present": len(info), "missing": missing}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="glm53-full78-local-20261025a")
    p.add_argument("--expect-missing", action="append", default=[])
    args = p.parse_args()
    audit(args.run_id, expect_missing=set(args.expect_missing))


if __name__ == "__main__":
    main()
