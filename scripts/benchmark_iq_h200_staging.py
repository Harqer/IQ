#!/usr/bin/env python3
"""Free, CPU-only sampling of IQ baseline shard staging through an existing mount.

This benchmarks the LOCAL machine and mount, not H200 networking or startup.
Does not start HF Jobs, change the checkpoint, or require CUDA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

from safetensors.torch import load_file
import torch

from scripts.iq_h200_staging import iter_staged_shards

DEFAULT_CHECKPOINT = (Path.home() / "iq-hf-output-fuse" / "IQ" / "runs" /
                      "glm53-full78-local-20261025a")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--samples", type=int, default=4,
                   help="Bounded number of baseline shards to validate (1..16)")
    p.add_argument("--workers", type=int, default=4,
                   help="Bounded staging concurrency (1..8)")
    p.add_argument("--stage-dir", type=Path,
                   help="Existing writable ephemeral staging parent (default system temp)")
    args = p.parse_args()
    if not 1 <= args.samples <= 16 or not 1 <= args.workers <= 8:
        p.error("--samples must be 1..16; --workers must be 1..8")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid safetensors index")
    base: dict[str, set[str]] = defaultdict(set)
    for key, name in weight_map.items():
        if not isinstance(key, str) or not isinstance(name, str):
            raise ValueError("Invalid safetensors weight map")
        if ".moe.routed_experts." not in key:
            base[name].add(key)
    selected = sorted(base)[:args.samples]
    if len(selected) != args.samples:
        raise ValueError("Requested more shards than available")
    started = perf_counter()
    nbytes = 0
    print(f"IQ_STAGE_BENCHMARK_CPU_ONLY=shards={len(selected)} workers={args.workers}",
          flush=True)
    for i, (name, local, size, stage_sec) in enumerate(iter_staged_shards(
        args.checkpoint, selected, workers=args.workers,
        stage_parent=args.stage_dir,
    ), 1):
        read_start = perf_counter()
        tensors = load_file(str(local), device="cpu")
        if set(tensors) != base[name]:
            raise ValueError(f"Unexpected tensors in {name}")
        for key, value in tensors.items():
            if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid tensor {key}")
        del tensors
        nbytes += size
        print(f"IQ_STAGE_BENCHMARK_SHARD={i}/{len(selected)} "
              f"bytes={size} stage_seconds={stage_sec:.3f} "
              f"decode_validate_seconds={perf_counter()-read_start:.3f}", flush=True)
    elapsed = perf_counter() - started
    print("IQ_STAGE_BENCHMARK_PASS=" + json.dumps({
        "shards": len(selected), "workers": args.workers,
        "bytes": nbytes, "elapsed_seconds": round(elapsed, 3),
        "effective_mib_per_second": round(nbytes / 1024**2 / elapsed, 2),
        "note": "Laptop/mount benchmark, NOT H200 throughput or guaranteed readiness",
        "paid_job_started": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
