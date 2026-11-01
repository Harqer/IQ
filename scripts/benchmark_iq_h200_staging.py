#!/usr/bin/env python3
"""No-GPU benchmark of IQ baseline safetensors (existing mount OR direct HF bucket).

The local path mode tests an existing mount and bounded staging pipeline.
Direct bucket mode downloads only requested sample shards, without FUSE.
Neither measures H200 Job filesystem speed or proves model readiness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json

from safetensors.torch import load_file
import torch

from scripts.iq_h200_staging import iter_staged_shards

BUCKET = "kofecheckmate/iq-glm53-all256"
PREFIX = "IQ/runs/glm53-full78-local-20261025a/"
DEFAULT_CHECKPOINT = (Path.home() / "iq-hf-output-fuse" / "IQ" / "runs" /
                      "glm53-full78-local-20261025a")


def baseline_map(index: dict) -> dict[str, set[str]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid safetensors index")
    base: dict[str, set[str]] = defaultdict(set)
    for key, name in weight_map.items():
        if not isinstance(key, str) or not isinstance(name, str):
            raise ValueError("Invalid safetensors weight map")
        if ".moe.routed_experts." not in key:
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError("Unsafe baseline shard path")
            base[name].add(key)
    return dict(base)


def iter_bucket_samples(bucket: str, names: list[str], workers: int):
    """Download at most `workers` 512-MiB capped samples concurrently."""
    from huggingface_hub import download_bucket_files, get_bucket_file_metadata
    with TemporaryDirectory(prefix="iq-hf-direct-bucket-") as directory:
        root = Path(directory)
        jobs = iter(names)
        pending = deque()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iq-hf-download") as pool:
            def fetch(name: str):
                remote = PREFIX + name
                size = get_bucket_file_metadata(bucket, remote).size
                if type(size) is not int or not 0 < size <= 512 * 1024**2:
                    raise ValueError(f"HF sample shard exceeds 512 MiB cap: {name}")
                started = perf_counter()
                dest = root / name
                download_bucket_files(bucket, files=[(remote, str(dest))])
                if dest.is_symlink() or dest.stat().st_size != size:
                    raise IOError(f"HF download size mismatch: {name}")
                return dest, size, perf_counter() - started

            def add_one():
                name = next(jobs, None)
                if name is not None:
                    pending.append((name, pool.submit(fetch, name)))

            for _ in range(workers):
                add_one()
            while pending:
                name, task = pending.popleft()
                file, size, seconds = task.result()
                try:
                    yield name, file, size, seconds
                finally:
                    file.unlink(missing_ok=True)
                add_one()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=Path,
                       help="Existing local checkpoint mount (default if --bucket omitted)")
    group.add_argument("--bucket", type=str,
                       help=f"Direct HF bucket download without FUSE (only {BUCKET})")
    p.add_argument("--samples", type=int, default=4,
                   help="Bounded number of baseline shards (1..16)")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent staging/download workers (1..8)")
    p.add_argument("--stage-dir", type=Path,
                   help="Existing writable local staging parent (mount mode only)")
    args = p.parse_args()
    if not 1 <= args.samples <= 16 or not 1 <= args.workers <= 8:
        p.error("--samples must be 1..16; --workers must be 1..8")
    if args.bucket:
        if args.bucket != BUCKET:
            p.error("Unexpected bucket; refusing direct download")
        if args.stage_dir:
            p.error("--stage-dir applies only to local-mount mode")
        from huggingface_hub import download_bucket_files, get_token
        if not get_token():
            p.error("HF authentication required: hf auth login")
        with TemporaryDirectory(prefix="iq-hf-index-") as temp:
            index_path = Path(temp) / "model.safetensors.index.json"
            download_bucket_files(
                BUCKET,
                files=[(PREFIX + index_path.name, str(index_path))],
            )
            base = baseline_map(json.loads(index_path.read_text()))
        selected = sorted(base)[:args.samples]
        samples = iter_bucket_samples(args.bucket, selected, args.workers)
        mode = "direct_bucket"
    else:
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        index = checkpoint / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(
                f"Checkpoint index unavailable: {index}. "
                "Use --bucket to benchmark directly without a local mount."
            )
        base = baseline_map(json.loads(index.read_text()))
        selected = sorted(base)[:args.samples]
        samples = iter_staged_shards(
            checkpoint, selected, workers=args.workers, stage_parent=args.stage_dir,
        )
        mode = "local_mount"
    if len(selected) != args.samples:
        raise ValueError("Requested more shards than available")
    started = perf_counter()
    nbytes = 0
    print(f"IQ_STAGE_BENCHMARK_CPU_ONLY=mode={mode} "
          f"shards={len(selected)} workers={args.workers}", flush=True)
    for i, (name, local, size, stage_seconds) in enumerate(samples, 1):
        read_start = perf_counter()
        tensors = load_file(str(local), device="cpu")
        if set(tensors) != base[name]:
            raise ValueError(f"Unexpected tensor keys in {name}")
        for key, value in tensors.items():
            if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint tensor {key}")
        del tensors
        nbytes += size
        print(f"IQ_STAGE_BENCHMARK_SHARD={i}/{len(selected)} "
              f"bytes={size} fetch_seconds={stage_seconds:.3f} "
              f"decode_validate_seconds={perf_counter()-read_start:.3f}", flush=True)
    elapsed = perf_counter() - started
    print("IQ_STAGE_BENCHMARK_PASS=" + json.dumps({
        "mode": mode, "shards": len(selected), "workers": args.workers,
        "bytes": nbytes, "elapsed_seconds": round(elapsed, 3),
        "effective_mib_per_second": round(nbytes / 1024**2 / elapsed, 2),
        "note": "Local benchmark, NOT H200 throughput or guaranteed readiness",
        "paid_job_started": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
