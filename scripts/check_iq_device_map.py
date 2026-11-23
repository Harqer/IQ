"""Read-only inference preflight for the IQ GLM-5.3/Mamba-3 full256 checkpoint.

Runs against HF bucket mounts; no weight arrays, GPU, or local model downloads.
Never updates COMPILE_COMPLETE.json or calls .generate().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import defaultdict
from pathlib import Path


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid SafeTensors file: {path.name}")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 10_000_000:
            raise ValueError(f"Invalid SafeTensors header length: {path.name}")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"Truncated SafeTensors header: {path.name}")
    return json.loads(raw)


def run(checkpoint: Path, device_map: Path, tokenizer_dir: Path | None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    cfg_bytes = (checkpoint / "iq_config.json").read_bytes()
    idx_bytes = (checkpoint / "model.safetensors.index.json").read_bytes()
    cfg = json.loads(cfg_bytes)
    index = json.loads(idx_bytes)
    plan = json.loads(device_map.read_text())
    layers = cfg["schedule"]["layers"]
    owners = plan["layer_to_gpu"]
    ndev = plan["device_count"]
    mapping = index["weight_map"]
    groups = defaultdict(list)
    for key, name in mapping.items():
        groups[name].append(key)

    if plan["config_sha256"] != hashlib.sha256(cfg_bytes).hexdigest():
        errors.append("Device map configuration SHA-256 mismatch")
    if plan["index_sha256"] != hashlib.sha256(idx_bytes).hexdigest():
        errors.append("Device map checkpoint index SHA-256 mismatch")
    if len(layers) != cfg["model"]["num_hidden_layers"] or len(owners) != len(layers):
        errors.append("Physical layer count or device-map length mismatch")
    if cfg["mamba3"]["num_layers"] != layers.count("mamba3"):
        errors.append("Mamba-3 physical layer count mismatch")
    if cfg["mamba3"]["mimo_rank"] != 4 or cfg["mamba3"]["chunk_size"] != 16:
        errors.append("Pinned Mamba-3 rank/chunk parameters differ")
    if cfg["attnres"]["num_layers"] != len(layers):
        errors.append("Block AttnRes layer count mismatch")
    if not isinstance(ndev, int) or ndev < 1 or sorted(set(owners)) != list(range(ndev)):
        errors.append("Incomplete GPU assignment")
    if any(a > b for a, b in zip(owners, owners[1:])):
        errors.append("GPU stages are non-contiguous")
    if plan.get("inference_ready") or not plan.get("plan_only"):
        errors.append("Unexecuted device map must remain plan_only and inference_ready=false")
    if set(groups) != set(plan["shard_to_gpu"]):
        errors.append("Device map does not cover every referenced shard")

    total = 0
    missing = []
    placement_errors = []
    for shard, keys in groups.items():
        if Path(shard).name != shard:
            errors.append(f"Unsafe shard name {shard}")
            continue
        path = checkpoint / shard
        if not path.is_file():
            missing.append(shard)
            continue
        size = path.stat().st_size
        if size < 9:
            errors.append(f"Empty/truncated shard {shard}")
        total += size
        desired = {}
        for key in keys:
            m = re.match(r"layers\.(\d+)\.", key)
            if m:
                stage = int(m.group(1))
                gpu = owners[stage] if stage < len(owners) else -1
            elif key.startswith("attnres.mixers."):
                m = re.match(r"attnres\.mixers\.(\d+)\.", key)
                gpu = owners[int(m.group(1))] if m and int(m.group(1)) < len(owners) else -1
            elif key == "embed_tokens.weight":
                gpu = 0
            else:
                gpu = ndev - 1
            desired[key] = gpu
        unique = sorted(set(desired.values()))
        expected = unique[0] if len(unique) == 1 else unique
        if plan["shard_to_gpu"].get(shard) != expected:
            placement_errors.append(shard)
        if len(unique) > 1:
            if plan.get("mixed_shard_tensor_to_gpu", {}).get(shard) != desired:
                placement_errors.append(f"{shard}: tensor map")
            if set(read_header(path)) - {"__metadata__"} != set(keys):
                placement_errors.append(f"{shard}: header/index disagreement")

    if missing:
        errors.append(f"{len(missing)} missing shards (first: {missing[:3]})")
    if placement_errors:
        errors.append(f"{len(placement_errors)} incorrect assignments (first: {placement_errors[:3]})")
    if total != index["metadata"]["total_size"] or total != plan["total_weight_bytes"]:
        errors.append("Physical shard sizes differ from checkpoint index/device map")
    if len(mapping) != plan["indexed_parameters"]:
        errors.append("Tensor key count differs from device map")
    if tokenizer_dir is None:
        warnings.append("Tokenizer mount not supplied; generation blocked")
        tokenizer_found = False
    else:
        tokenizer_found = any((tokenizer_dir / name).is_file() for name in ("tokenizer.json", "tokenizer.model"))
        if not tokenizer_found:
            warnings.append("Tokenizer mount missing tokenizer.json/tokenizer.model")
        else:
            warnings.append("Tokenizer exists; full 128256-vocabulary identity check is still required")
    return {
        "metadata_passed": not errors,
        "inference_ready": False,
        "tensor_values_read": False,
        "checkpoints_modified": False,
        "physical_layers": len(layers),
        "mamba3_layers": layers.count("mamba3"),
        "moe_layers": layers.count("moe"),
        "device_count": ndev,
        "checkpoint_GB": round(total / 1e9, 3),
        "indexed_parameter_tensors": len(mapping),
        "indexed_weight_shards": len(groups),
        "trainable_parameter_count": "Not recorded in SafeTensors; requires_grad is not an inference criterion",
        "tokenizer_found": tokenizer_found,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--device-map", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument("--require-tokenizer", action="store_true")
    args = p.parse_args()
    try:
        report = run(args.checkpoint, args.device_map, args.tokenizer_dir)
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        report = {"metadata_passed": False, "inference_ready": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    if args.require_tokenizer and not report.get("tokenizer_found"):
        report.setdefault("errors", []).append("Tokenizer required for token generation")
        report["metadata_passed"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["metadata_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
