#!/usr/bin/env python3
"""Cost-free, read-only HF preflight for the COMPLETE GLM-5.3 full78 IQ checkpoint.

NO Hugging Face Jobs, GPU allocations, checkpoint downloads, or bucket writes.
Fail closed on incorrect model/index/weight-inventory. Reports why the current
single-device loader cannot run the 502 GiB checkpoint, even on multi-GPU Jobs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "glm53-full78-local-20261025a"
PREFIX = f"IQ/runs/{RUN}"
INDEX = "model.safetensors.index.json"
CONFIG = "iq_config.json"
MARKER = "COMPILE_COMPLETE.json"
EXPECTED_SHARDS = 20283
EXPECTED_TENSORS = 63164

# Approximate advertised aggregate VRAM, in GiB, not usable capacity.
TIERS = {
    "h200": (141, 5.00),
    "h200x4": (564, 20.00),
    "h200x8": (1128, 40.00),
    "a100x8": (640, 20.00),
}


def validate_full78(index: dict[str, Any], config: dict[str, Any],
                    complete: dict[str, Any]) -> dict[str, Any]:
    if complete.get("status") != "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED":
        raise RuntimeError("Incorrect/unpublished full78 completion marker")
    if complete.get("inference_ready") is not False:
        raise RuntimeError("Completion marker must not claim inference readiness")
    if config.get("model", {}).get("vocab_size") != 128256:
        raise RuntimeError("Wrong IQ foundation/tokenizer vocabulary")
    if config.get("model", {}).get("hidden_size") != 4096:
        raise RuntimeError("Wrong IQ hidden width")
    schedule = config.get("schedule", {}).get("layers", [])
    if (len(schedule) != 235 or
        sum(v == "mamba3" for v in schedule) != 79 or
        sum(v == "moe" for v in schedule) != 78):
        raise RuntimeError("Wrong full78 physical layer schedule")
    if (config.get("stable_moe", {}).get("num_experts") != 256 or
        config.get("stable_moe", {}).get("top_k") != 8):
        raise RuntimeError("Wrong routed expert configuration")
    mapping = index.get("weight_map", {})
    total_bytes = index.get("metadata", {}).get("total_size")
    if not isinstance(mapping, dict) or len(mapping) != EXPECTED_TENSORS:
        raise RuntimeError("Unexpected weight-map tensor count")
    if type(total_bytes) is not int or total_bytes < 450 * 1024**3:
        raise RuntimeError("Invalid full78 checkpoint total weight size")
    filenames = set()
    for key, value in mapping.items():
        if (not isinstance(key, str) or not key or
            not isinstance(value, str) or not value.endswith(".safetensors")
            or value != Path(value).name or ".." in value):
            raise RuntimeError("Unsafe/corrupt checkpoint weight map")
        filenames.add(value)
    if len(filenames) != EXPECTED_SHARDS:
        raise RuntimeError("Unexpected full78 checkpoint shard count")
    if len(mapping) != complete.get("mapped_routed_expert_slots", 0) + (
        EXPECTED_TENSORS - 19200
    ):
        raise RuntimeError("Wrong mapped-routed-expert slot count")
    return {
        "checkpoint_bucket": BUCKET,
        "run": RUN,
        "physical_layers": len(schedule),
        "mamba_layers": 79,
        "moe_layers": 78,
        "routed_expert_slots": complete["mapped_routed_expert_slots"],
        "weight_shards": len(filenames),
        "weight_tensors": len(mapping),
        "total_weight_bytes": total_bytes,
        "total_weight_gib": round(total_bytes / 1024**3, 3),
        "tokenizer": "Llama 3.1 (external mount, vocab 128256)",
        "tokenizer_packaged": bool(complete.get("tokenizer_packaged")),
        "already_inference_validated": False,
    }


def hardware_plan(total_bytes: int, flavor: str) -> dict[str, Any]:
    if flavor not in TIERS:
        raise ValueError(f"Unknown flavor: {flavor}")
    vram_gib, hourly = TIERS[flavor]
    weight_gib = total_bytes / 1024**3
    # Single GPU model copy, not parameter sharding. Cannot run regardless
    # of how many GPUs exist in a Job.
    return {
        "flavor": flavor,
        "advertised_aggregate_vram_gib": vram_gib,
        "weights_gib": round(weight_gib, 3),
        "nominal_vram_margin_gib": round(vram_gib - weight_gib, 3),
        "advertised_usd_per_hour": hourly,
        "current_loader_model_parallel": False,
        "current_loader_full78_inference_runnable": False,
        "reason": "current load_complete_iq_checkpoint constructs IQHybridForCausalLM on one CUDA device",
    }


def load_bucket_metadata() -> tuple[dict, dict, dict]:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or None)
    results = []
    for name in (INDEX, CONFIG, MARKER):
        uri = f"hf://buckets/{BUCKET}/{PREFIX}/{name}"
        with fs.open(uri, "rb", block_size=0, cache_type="none") as fp:
            limit = 8 * 1024 * 1024 if name == INDEX else 256 * 1024
            data = fp.read(limit + 1)
            if len(data) > limit:
                raise RuntimeError(f"Oversize metadata: {name}")
            results.append(json.loads(data))
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=tuple(TIERS), default="h200x4")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional local JSON report; refuses to overwrite")
    args = parser.parse_args()
    # User's local HF_TOKEN determines access. A linked ChatGPT plugin may
    # authenticate as a different account, so a volume mount still needs proof.
    index, config, complete = load_bucket_metadata()
    report = validate_full78(index, config, complete)
    report["hardware"] = hardware_plan(report["total_weight_bytes"], args.flavor)
    report["hf_job_submitted"] = False
    report["hf_mount_usable_inside_job"] = "not_yet_proven"
    report["blocking_steps"] = [
        "Implement and test multi-GPU sharded or streamed full78 IQ model loader",
        "Validate pinned Mamba-3 MIMO kernels on selected HF GPU hardware",
        "Verify target HF Job account can mount source bucket read-only",
        "Validate external Llama 3.1 tokenizer and token IDs",
        "Explicitly authorize paid GPU usage before launch",
    ]
    print("IQ_HF_FULL78_METADATA_PREFLIGHT_PASS=" + json.dumps({
        k: report[k] for k in (
            "weight_shards", "weight_tensors", "total_weight_gib",
            "physical_layers", "moe_layers", "mamba_layers"
        )
    }, sort_keys=True), flush=True)
    print("IQ_HF_FULL78_GPU_LAUNCH_BLOCKED=" +
          json.dumps(report["hardware"], sort_keys=True), flush=True)
    print("IQ_HF_FULL78_NO_PAID_JOB_SUBMITTED=1", flush=True)
    if args.output:
        if args.output.exists():
            raise RuntimeError(f"Refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
