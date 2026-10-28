#!/usr/bin/env python3
"""Finalize an ALREADY COMPLETE GLM53 full78 run without loading model weights.

Strictly checks all expected shards and safetensors headers one at a time.
No CUDA, no matrix inverses, no donor weight loading. Default is read-only.
--publish publishes immutable manifest, index, and then non-inference-ready
completion marker using the independently SHA256-verified HF bucket API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "glm53-full78-local-20261025a"
GLM_REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAMBA_REV = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
MAX_HEADER = 4 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safetensors_header(path: Path) -> dict[str, dict]:
    """Validate the complete offsets, shapes, dtypes and byte spans.

    Read ONLY the small header; never mmap or deserialize the large tensors.
    Does not validate numerical values or whole-file SHA256.
    """
    file_size = path.stat().st_size
    if file_size < 16 or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Invalid checkpoint object {path.name}")
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"Short safetensors prefix {path.name}")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= MAX_HEADER or size + 8 > file_size:
            raise RuntimeError(f"Invalid safetensors header length {path.name}")
        data = json.loads(f.read(size))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid safetensors JSON {path.name}")
    data.pop("__metadata__", None)
    if not data:
        raise RuntimeError(f"Empty safetensors shard {path.name}")
    spans = []
    for key, item in data.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            raise RuntimeError(f"Invalid safetensors entry {path.name}")
        shape, dtype, offsets = item.get("shape"), item.get("dtype"), item.get("data_offsets")
        if (not isinstance(shape, list) or not shape or
            any(type(dim) is not int or dim < 1 for dim in shape) or
            dtype not in DTYPE_BYTES or not isinstance(offsets, list) or
            len(offsets) != 2 or any(type(v) is not int for v in offsets)):
            raise RuntimeError(f"Invalid tensor metadata: {path.name}:{key}")
        start, end = offsets
        expected_bytes = DTYPE_BYTES[dtype]
        for dim in shape:
            expected_bytes *= dim
        if start < 0 or end <= start or end - start != expected_bytes:
            raise RuntimeError(f"Invalid tensor byte extent: {path.name}:{key}")
        spans.append((start, end))
    previous = 0
    for start, end in sorted(spans):
        if start != previous:
            raise RuntimeError(f"Noncontiguous or overlapping tensor data: {path.name}")
        previous = end
    if previous != file_size - 8 - size:
        raise RuntimeError(f"Truncated or trailing tensor data: {path.name}")
    return data


def expected_names() -> set[str]:
    from scripts.glm53_full78_gate import expected_stage_shards
    names = {"model-global.safetensors", "model-native.safetensors"}
    for stage in range(78):
        names.update(n for n in expected_stage_shards(stage) if n.endswith(".safetensors"))
    names.update(f"model-mamba-{i:03d}.safetensors" for i in range(79))
    if len(names) != 20283:
        raise RuntimeError(f"Unexpected number of weight shards: {len(names)}")
    return names


def validate_bucket(root: Path, *, api_sizes: dict[str, int],
                    schedule: list[str], stages: list[dict], latent: int,
                    intermediate: int) -> dict[str, str]:
    expected = expected_names()
    if set(api_sizes) != expected:
        raise RuntimeError(f"Incorrect remote weight inventory: missing={sorted(expected-set(api_sizes))[:5]}")
    index: dict[str, str] = {}
    for j, name in enumerate(sorted(expected), 1):
        path = root / name
        if not path.is_file() or path.stat().st_size != api_sizes[name]:
            raise RuntimeError(f"Remote API vs NFS metadata mismatch: {name}")
        entries = safetensors_header(path)
        if name == "model-global.safetensors":
            if set(entries) != {"embed_tokens.weight", "norm.weight", "lm_head.weight"}:
                raise RuntimeError("Invalid lexical foundation tensor keys")
        elif name == "model-native.safetensors":
            if any(key.startswith("layers.") for key in entries):
                raise RuntimeError("Unexpected layer tensors in native-controller shard")
        elif name.startswith("model-mamba-"):
            ordinal = int(name[12:15])
            physical = [i for i, kind in enumerate(schedule) if kind == "mamba3"][ordinal]
            prefix = f"layers.{physical}."
            if len(entries) != 12 or any(not key.startswith(prefix) for key in entries):
                raise RuntimeError(f"Invalid Mamba physical placement: {name}")
        elif name.startswith("model-context-"):
            stage = stages[int(name[14:17])]
            physical = int(stage["context_physical_layer"])
            if (schedule[physical] not in ("csa", "hca") or
                any(not k.startswith(f"layers.{physical}.") for k in entries)):
                raise RuntimeError(f"Invalid context physical placement: {name}")
        elif name.startswith("model-moe-"):
            stage = stages[int(name[10:13])]
            physical = int(stage["moe_physical_layer"])
            if schedule[physical] != "moe" or any(
                not k.startswith(f"layers.{physical}.") for k in entries
            ):
                raise RuntimeError(f"Invalid MoE physical placement: {name}")
            if "-expert-" in name:
                slot = int(name[-15:-12])
                prefix = f"layers.{physical}.moe.routed_experts.{slot}."
                expected_shapes = {
                    prefix + "gate_proj.weight": [intermediate, latent],
                    prefix + "up_proj.weight": [intermediate, latent],
                    prefix + "down_proj.weight": [latent, intermediate],
                }
                if set(entries) != set(expected_shapes):
                    raise RuntimeError(f"Wrong expert tensor inventory: {name}")
                for key, shape in expected_shapes.items():
                    if entries[key]["shape"] != shape or entries[key]["dtype"] != "BF16":
                        raise RuntimeError(f"Wrong expert layout/dtype: {name}:{key}")
            elif name.endswith("-common.safetensors"):
                prefix = f"layers.{physical}.moe."
                common = {
                    f"layers.{physical}.norm.weight",
                    prefix + "latent_norm.weight",
                    prefix + "latent_down.weight",
                    prefix + "latent_up.weight",
                    prefix + "router.weight",
                    prefix + "routing_bias",
                }
                if set(entries) != common:
                    raise RuntimeError(f"Wrong common MoE tensor inventory: {name}")
            elif name.endswith("-shared.safetensors"):
                prefix = f"layers.{physical}.moe.shared_experts.0."
                if set(entries) != {
                    prefix + "gate_proj.weight",
                    prefix + "up_proj.weight",
                    prefix + "down_proj.weight",
                }:
                    raise RuntimeError(f"Wrong shared MoE tensor inventory: {name}")
            else:
                raise RuntimeError(f"Unexpected MoE shard {name}")
        for key in entries:
            if key in index:
                raise RuntimeError(f"Duplicate tensor key {key} in {name} and {index[key]}")
            index[key] = name
        if j % 1000 == 0 or j == len(expected):
            print(f"IQ_METADATA_HEADER_VERIFIED={j}/{len(expected)}", flush=True)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true",
                        help="Upload index/manifest/completion after full dry-run checks")
    parser.add_argument("--mount", type=Path,
                        default=Path.home() / "iq-hf-output")
    parser.add_argument("--root", type=Path,
                        default=Path.home() / "IQ-transfer-full78")
    args = parser.parse_args()
    output = args.mount / "IQ" / "runs" / RUN
    calibration = args.root / "calibration"
    if not output.is_dir():
        raise RuntimeError("Existing readonly checkpoint mount unavailable")
    # Do not silently complete a potentially concurrent compile.
    if os.environ.get("IQ_CONFIRM_NO_OTHER_WRITER") != "1":
        raise RuntimeError("Set IQ_CONFIRM_NO_OTHER_WRITER=1 after checking no compiler is running")
    config_bytes = (output / "iq_config.json").read_bytes()
    config = json.loads(config_bytes)
    fingerprint = sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    run_provenance = json.loads((output / "run-provenance.json").read_text())
    receipt = calibration / "calibration_sha256.json"
    receipt_obj = json.loads(receipt.read_text())
    if receipt_obj.get("schema_version") != 1 or len(receipt_obj.get("files", {})) != 866:
        raise RuntimeError("Bad calibration receipt: expected 866 verified entries")
    if run_provenance != {
        "schema_version": 1,
        "calibration_receipt_sha256": sha256(receipt.read_bytes()),
        "teacher_revision": GLM_REV,
        "mamba_revision": MAMBA_REV,
        "target_fingerprint": fingerprint,
    }:
        raise RuntimeError("Pinned run provenance or config fingerprint mismatch")

    stages = json.loads((calibration / "glm53_calibration.json").read_text())["stages"]
    schedule = config["schedule"]["layers"]
    stable = config["stable_moe"]
    if (len(stages) != 78 or len(schedule) != 235 or
        schedule.count("mamba3") != 79 or
        schedule.count("moe") != 78 or
        schedule.count("csa") + schedule.count("hca") != 78 or
        stable["num_experts"] != 256):
        raise RuntimeError("Noncanonical 78-stage / 79-Mamba configuration")
    source_map = {}
    selected_ids = {}
    for i, stage in enumerate(stages):
        if stage.get("stage") != i or stage.get("source_layer") != i:
            raise RuntimeError(f"Noncanonical stage source mapping: {i}")
        physical_context = int(stage["context_physical_layer"])
        physical_moe = int(stage["moe_physical_layer"])
        if schedule[physical_context] not in ("csa", "hca") or schedule[physical_moe] != "moe":
            raise RuntimeError(f"Wrong physical schedule for stage {i}")
        source_map[str(i)] = i
        if i < 3:
            if stage.get("expert_usage") is not None or not stage.get("dense_intermediate_map"):
                raise RuntimeError(f"Bad dense stage calibration: {i}")
            selected_ids[str(i)] = None
        else:
            import numpy as np
            usage_file = stage.get("expert_usage")
            if not usage_file:
                raise RuntimeError(f"Missing expert usage at stage {i}")
            usage = np.load(calibration / usage_file, allow_pickle=False).reshape(-1)
            if len(usage) != 256 or not np.isfinite(usage).all() or (usage < 0).any() or float(usage.sum()) <= 0:
                raise RuntimeError(f"Bad expert usage data at stage {i}")
            selected_ids[str(i)] = [int(k) for k in np.lexsort(
                (np.arange(len(usage)), -usage)
            )]
        report = json.loads((output / f"model-moe-{i:03d}-projection.json").read_text())
        if report.get("stage") != i or report.get("source_layer") != i:
            raise RuntimeError(f"Invalid saved stage projection receipt {i}")

    # Only reads official donor CONFIG and INDEX. Never reads any donor weight.
    from iq_transfer.glm53_job import validate_glm53_streaming_donor
    donor = validate_glm53_streaming_donor(
        checkpoint=args.root / "donors" / "glm53",
        checkpoint_revision=GLM_REV, donor_license="GLM-5.3",
    )
    print("IQ_FINALIZER_DONOR_METADATA_VERIFIED=1", flush=True)

    from huggingface_hub import get_bucket_paths_info
    names = sorted(expected_names())
    prefix = f"IQ/runs/{RUN}"
    api = {item.path.removeprefix(prefix + "/"): item.size for item in
           get_bucket_paths_info(BUCKET, [f"{prefix}/{n}" for n in names])}
    weight_map = validate_bucket(
        output, api_sizes=api, schedule=schedule, stages=stages,
        latent=int(stable["latent_size"]),
        intermediate=int(stable["expert_intermediate_size"]),
    )
    total_size = sum(api.values())
    provenance = {
        "schema_version": 2,
        "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
        "foundation": {
            "repo": "state-spaces/mamba3-mimo-1.5b",
            "revision": MAMBA_REV,
            "role": "pretrained lexical + recurrent foundation",
            "width_transfer": "2048->4096 exact replication embedding",
        },
        "capability_source": {
            "repo": "zai-org/GLM-5.3-BF16",
            "revision": GLM_REV,
            "fingerprint": donor.manifest.fingerprint,
            "checkpoint_hash": donor.manifest.checkpoint_hash,
            "role": "WARM-remapped context/DSA/MoE capability weights",
        },
        "recipient_config_fingerprint": fingerprint,
        "calibration_mode": json.loads((calibration / "glm53_calibration.json").read_text())["calibration_mode"],
        "routed_experts_per_sparse_stage": 256,
        "selected_source_expert_ids": selected_ids,
        "source_layer_map": source_map,
        "context": "MLA compressed-latent refactor + DSA direct-token indexer",
        "moe": "usage-selected expert subcloning into Stable LatentMoE",
        "mamba": "official Mamba-3 MIMO foundation; no GLM QKV-to-recurrence fabrication",
    }
    index = {"metadata": {"total_size": total_size},
             "weight_map": dict(sorted(weight_map.items()))}
    complete = {
        "status": "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED",
        "source_layers": 78, "mapped_layers": 78,
        "routed_source_layers": 75, "routed_experts_per_layer": 256,
        "mapped_routed_expert_slots": 19200,
        "not_exact_bitwise_weight_reuse": True,
        "tokenizer_packaged": (output / "tokenizer.json").is_file() and
                              (output / "tokenizer_config.json").is_file(),
        "inference_ready": False,
        "end_to_end_knowledge_retention_proven": False,
    }
    print(f"IQ_METADATA_FINALIZER_READY shards={len(names)} tensors={len(weight_map)} "
          f"bytes={total_size}", flush=True)
    if not args.publish:
        print("IQ_METADATA_FINALIZER_DRY_RUN_OK_NO_WRITES", flush=True)
        return
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Set HF_TOKEN to publish via authorized Hub API")
    from iq_transfer.glm53_bucket_persistence import (
        write_checkpoint_text, remote_checkpoint_sizes,
    )
    os.environ.update({
        "IQ_HF_DURABLE_OUTPUT": str(output.resolve()),
        "IQ_HF_DURABLE_BUCKET": BUCKET,
        "IQ_HF_DURABLE_PREFIX": prefix,
        "IQ_HF_DURABLE_STAGING": str((args.root / "shard-staging").resolve()),
    })
    # Idempotent publishing verifies exact content through SHA256 readback.
    for filename, data in (
        ("model.safetensors.index.json", index),
        ("transplant_manifest.json", provenance),
    ):
        write_checkpoint_text(output, filename, json.dumps(
            data, indent=2, sort_keys=True
        ) + "\n")
    remote_checkpoint_sizes(output, names + [
        "model.safetensors.index.json", "transplant_manifest.json",
        "iq_config.json", "run-provenance.json"
    ])
    write_checkpoint_text(output, "COMPILE_COMPLETE.json", json.dumps(
        complete, sort_keys=True
    ) + "\n")
    remote_checkpoint_sizes(output, [
        "model.safetensors.index.json", "transplant_manifest.json",
        "COMPILE_COMPLETE.json"
    ])
    print("IQ_METADATA_FINALIZER_COMMITTED_INFERENCE_READY_FALSE", flush=True)


if __name__ == "__main__":
    main()
