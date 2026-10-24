#!/usr/bin/env python3
"""Estimate network and temporary-disk cost of planned GLM BF16 shards.

Reads only Hub file metadata. Does not download any model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


REPO = "zai-org/GLM-5.3-BF16"
REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


class ShardEstimateError(ValueError):
    pass


def estimate_shard_bytes(
    plan: dict, sizes: dict[str, int], *, capacity_bytes: int
) -> dict:
    if plan.get("source_revision") != REVISION or plan.get("target_experts") not in (64, 128, 256):
        raise ShardEstimateError("unexpected donor revision or expert capacity")
    names = plan.get("official_shards")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ShardEstimateError("missing/duplicate source shard inventory")
    if not all(
        isinstance(x, str)
        and x.startswith("model-")
        and x.endswith(".safetensors")
        and "/" not in x
        for x in names
    ):
        raise ShardEstimateError("invalid shard filename")
    missing = sorted(set(names) - set(sizes))
    if missing:
        raise ShardEstimateError(f"Hub did not report sizes for {len(missing)} planned shards: {missing[:5]}")
    if any(type(sizes[name]) is not int or sizes[name] <= 0 for name in names):
        raise ShardEstimateError("Hub returned invalid shard byte lengths")
    total = sum(sizes[name] for name in names)
    largest = max(sizes[name] for name in names)
    return {
        "status": "metadata_estimate_only",
        "shards": len(names),
        "bytes_to_download_if_every_planned_shard_fetched_once": total,
        "total_download_GiB": round(total / (1024 ** 3), 2),
        "largest_single_shard_GiB": round(largest / (1024 ** 3), 2),
        "disk_available_GiB": round(capacity_bytes / (1024 ** 3), 2),
        "all_shards_fit_simultaneously": capacity_bytes >= total,
        "notes": [
            "Not a quota/billing quote: provider network traffic and storage may differ.",
            "Stage-wise streaming can reduce peak disk but not total transferred bytes.",
            "The 17.5 GB original calibration and IQ checkpoint storage are additional.",
            "No weight shards downloaded or compute started.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--disk", type=Path, default=Path.home())
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        info = HfApi().model_info(
            repo_id=REPO,
            revision=REVISION,
            files_metadata=True,
        )
        sizes = {
            file.rfilename: file.size
            for file in info.siblings or []
            if file.size is not None
        }
        report = estimate_shard_bytes(
            plan,
            sizes,
            capacity_bytes=shutil.disk_usage(args.disk).free,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Shard storage estimation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
