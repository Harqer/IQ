#!/usr/bin/env python3
"""Free CPU-only diagnostic for IQ full78's FIRST non-expert checkpoint shard.

Replicates the beginning of load_base_weights using an existing LOCAL mount.
Never allocates a GPU or creates a Hugging Face Job. The checkpoint directory
must already exist: typically ~/iq-hf-output-fuse/IQ/runs/<RUN>.
Run with 'timeout 180s python -m scripts.diagnose_glm53_first_shard ...'
to cap an indefinitely stalled FUSE read. Logs appear before each I/O step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter


RUN = "glm53-full78-local-20261025a"
DEFAULT_CHECKPOINT = Path.home() / "iq-hf-output-fuse" / "IQ" / "runs" / RUN
MAX_SHARD_GIB = 3


def find_first_base_shard(index_path: Path) -> tuple[str, set[str]]:
    """Resolve the first file the real CUDA loader would attempt to read."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Invalid or empty safetensors weight_map")
    base: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not isinstance(filename, str):
            raise ValueError("Invalid weight_map key or filename")
        if ".moe.routed_experts." in key:
            continue
        if not filename.endswith(".safetensors"):
            raise ValueError("Non-safetensors base shard entry")
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("Unsafe shard filename")
        base.setdefault(filename, set()).add(key)
    if not base:
        raise ValueError("No non-expert shards")
    name = sorted(base)[0]
    return name, base[name]


def read_shard(checkpoint: Path, *, max_shard_gib: int = MAX_SHARD_GIB) -> dict:
    """Read and validate only the first baseline shard; do not touch remaining 314."""
    started = perf_counter()
    index = checkpoint / "model.safetensors.index.json"
    print(f"IQ_FIRST_SHARD_INDEX_START={index}", flush=True)
    name, expected = find_first_base_shard(index)
    print(f"IQ_FIRST_SHARD_INDEX_OK=file={name} expected_tensors={len(expected)}", flush=True)
    shard = checkpoint / name
    print(f"IQ_FIRST_SHARD_STAT_START={shard}", flush=True)
    if shard.is_symlink() or not shard.is_file():
        raise FileNotFoundError(f"Missing or symlinked checkpoint shard: {shard}")
    size = shard.stat().st_size
    if size <= 0 or size > max_shard_gib * 1024 ** 3:
        raise ValueError(f"Shard size outside 0..{max_shard_gib} GiB: {size}")
    print(f"IQ_FIRST_SHARD_READ_START=file={name} bytes={size}", flush=True)
    from safetensors.torch import load_file
    before = perf_counter()
    tensors = load_file(str(shard), device="cpu")
    elapsed = perf_counter() - before
    print(f"IQ_FIRST_SHARD_READ_PASS=seconds={elapsed:.3f} tensors={len(tensors)}", flush=True)
    if set(tensors) != expected:
        raise ValueError(f"Shard tensor keys mismatch: {name}")
    # Mirror loader's float/finite test without copying to a GPU.
    print(f"IQ_FIRST_SHARD_VALIDATE_START=tensors={len(tensors)}", flush=True)
    import torch
    for key, value in tensors.items():
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfloating/nonfinite checkpoint tensor: {key}")
    print(f"IQ_FIRST_SHARD_VALIDATED=seconds={perf_counter()-started:.3f}", flush=True)
    return {"shard": name, "bytes": size, "seconds_to_read": round(elapsed, 3),
            "tensor_count": len(tensors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Existing LOCAL filesystem mount of IQ full78 run")
    parser.add_argument("--max-shard-gib", type=int, default=MAX_SHARD_GIB)
    args = parser.parse_args()
    if not 1 <= args.max_shard_gib <= 8:
        parser.error("max-shard-gib must be 1..8")
    print("IQ_FIRST_SHARD_DIAGNOSTIC=CPU_ONLY_NO_PAID_JOB", flush=True)
    result = read_shard(args.checkpoint, max_shard_gib=args.max_shard_gib)
    print("IQ_FIRST_SHARD_SUCCESS=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
