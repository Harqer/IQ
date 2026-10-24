#!/usr/bin/env python3
"""Fail-closed, read-only checkpoints for GLM-5.3 full78 cloud execution.

Must run on a SEPARATE read-only mount after a previous upload/worker ends.
A manifest or successful job status is NOT proof of persistent completeness.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

GLM_LAYERS = 78
ROUTED_LAYERS = 75
EXPERTS = 256
ROUTED_SLOTS = ROUTED_LAYERS * EXPERTS
ALL_EXPERT_SLOTS = GLM_LAYERS * EXPERTS
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class GateFailure(RuntimeError):
    pass


def _safe_filename(name: str) -> str:
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in (".", ".."):
        raise GateFailure(f"unsafe artifact basename: {name!r}")
    return name


def _load_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateFailure(f"missing or invalid JSON: {path.name}") from exc
    if not isinstance(obj, dict):
        raise GateFailure(f"expected JSON object: {path.name}")
    return obj


def expected_calibration_files(manifest: dict) -> set[str]:
    if manifest.get("schema_version") != 2 or manifest.get("source_layers") != GLM_LAYERS:
        raise GateFailure("calibration version or source layer count mismatch")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != GLM_LAYERS:
        raise GateFailure("calibration must contain exactly 78 stage records")
    result = {"glm53_calibration.json"}
    map_keys = ("residual_map", "attention_input_map", "moe_input_map",
                "compressed_kv_map", "latent_map", "dense_intermediate_map")
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get("stage") != i or stage.get("source_layer") != i:
            raise GateFailure(f"incorrect source mapping at stage {i}")
        if (stage.get("dense_intermediate_map") is not None) != (i < 3):
            raise GateFailure(f"incorrect dense intermediate mapping at {i}")
        usage = stage.get("expert_usage")
        if (usage is not None) != (i >= 3):
            raise GateFailure(f"incorrect routed expert usage map at {i}")
        for field in map_keys:
            name = stage.get(field)
            if name is None:
                continue
            name = _safe_filename(name)
            if not name.startswith(f"stage-{i:02d}-"):
                raise GateFailure(f"stage {i}: wrong coordinate filename for {field}")
            result.add(name + ".json")
            result.add(name + ".safetensors")
        if usage is not None:
            usage = _safe_filename(usage)
            if usage != f"stage-{i:02d}-expert-usage.npy":
                raise GateFailure(f"stage {i}: unexpected expert usage filename")
            result.add(usage)
    for field, expected in (("lexical_input_map", "lexical-input"),
                            ("final_output_map", "final-output")):
        if manifest.get(field) != expected:
            raise GateFailure(f"wrong lexical map {field}")
        result.update((expected + ".json", expected + ".safetensors"))
    return result


def inspect_files(root: Path, expected: set[str], *, min_shard_bytes: int = 16) -> dict:
    if not root.is_dir():
        raise GateFailure(f"directory does not exist: {root}")
    present = {p.name: p.stat().st_size for p in root.iterdir()
               if p.is_file() and not p.is_symlink()}
    missing = sorted(expected.difference(present))
    short = sorted(name for name in expected.intersection(present)
                   if present[name] < (min_shard_bytes if name.endswith(".safetensors") else 1))
    return {"expected_files": len(expected), "present_files": len(expected)-len(missing),
            "missing_count": len(missing), "missing_examples": missing[:25],
            "invalid_count": len(short), "invalid_examples": short[:25]}


def check_calibration(root: Path) -> dict:
    manifest = _load_json(root / "glm53_calibration.json")
    result = inspect_files(root, expected_calibration_files(manifest))
    result["stage"] = "calibration"
    result["source_layers"] = GLM_LAYERS
    result["ready"] = result["missing_count"] == 0 and result["invalid_count"] == 0
    return result


def expected_stage_shards(i: int) -> set[str]:
    if i not in range(GLM_LAYERS):
        raise GateFailure("stage must be in 0..77")
    prefix = f"model-moe-{i:03d}-"
    return {
        f"model-context-{i:03d}.safetensors",
        prefix + "common.safetensors",
        prefix + "shared.safetensors",
        prefix + "projection.json",
        *(prefix + f"expert-{slot:03d}.safetensors" for slot in range(EXPERTS)),
    }


def check_checkpoint_stage(root: Path, i: int) -> dict:
    result = inspect_files(root, expected_stage_shards(i))
    result["stage"] = "checkpoint"
    result["source_layer"] = i
    result["ready"] = result["missing_count"] == 0 and result["invalid_count"] == 0
    return result


def check_checkpoint_complete(root: Path) -> dict:
    index = _load_json(root / "model.safetensors.index.json")
    finished = _load_json(root / "COMPILE_COMPLETE.json")
    if finished.get("source_layers") != GLM_LAYERS or finished.get("mapped_routed_expert_slots") != ROUTED_SLOTS:
        raise GateFailure("completion record does not claim 78 original and 19,200 routed slots")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise GateFailure("missing final safetensors index mapping")
    all_stage_files = set()
    for i in range(GLM_LAYERS):
        all_stage_files.update(expected_stage_shards(i))
    expected = {x for x in all_stage_files if x.endswith(".safetensors")}
    expected.update(("model-global.safetensors", "model-native.safetensors"))
    present = inspect_files(root, expected)
    referenced = set(mapping.values())
    if not all(isinstance(n, str) and SAFE_NAME.fullmatch(n) for n in referenced):
        raise GateFailure("invalid final safetensors index filename")
    index_files = inspect_files(root, referenced)
    result = {"stage": "completed_checkpoint", "source_layers": GLM_LAYERS,
              "expert_slots": ALL_EXPERT_SLOTS, "files": present,
              "index_files": index_files}
    result["ready"] = (present["missing_count"] == present["invalid_count"] == 0
                       and index_files["missing_count"] == index_files["invalid_count"] == 0)
    return result


def check_capacity(attestation: Path, required_bytes: int, bucket: str) -> dict:
    data = _load_json(attestation)
    if data.get("bucket") != bucket:
        raise GateFailure("storage attestation names a different bucket")
    if not data.get("source") or data.get("source") in ("user_guess", "unknown"):
        raise GateFailure("storage must be verified with provider/account evidence")
    timestamp = data.get("checked_at")
    try:
        checked = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise GateFailure("storage attestation needs timestamp with timezone") from exc
    age = dt.datetime.now(dt.timezone.utc) - checked
    if checked.utcoffset() is None or age.total_seconds() < 0 or age > dt.timedelta(hours=24):
        raise GateFailure("stale/future storage attestation")
    free = data.get("confirmed_free_bytes")
    if type(free) is not int or free < required_bytes:
        raise GateFailure(f"confirmed free storage {free} insufficient; need {required_bytes}")
    return {"stage": "storage", "ready": True, "free_bytes": free,
            "required_bytes": required_bytes, "source": data["source"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=("calibration", "checkpoint-stage",
                                        "checkpoint-complete", "storage"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--required-bytes", type=int)
    parser.add_argument("--bucket", default="kofecheckmate/iq-glm53-all256")
    args = parser.parse_args()
    try:
        if args.gate == "storage":
            if not args.attestation or not args.required_bytes:
                raise GateFailure("storage requires --attestation and --required-bytes")
            result = check_capacity(args.attestation, args.required_bytes, args.bucket)
        else:
            if not args.root:
                raise GateFailure("--root is required")
            if args.gate == "calibration":
                result = check_calibration(args.root)
            elif args.gate == "checkpoint-stage":
                result = check_checkpoint_stage(args.root, args.layer)
            else:
                result = check_checkpoint_complete(args.root)
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result["ready"]:
            raise GateFailure(f"{args.gate}: incomplete or corrupt persistent artifacts")
        return 0
    except GateFailure as exc:
        print(json.dumps({"gate": args.gate, "status": "BLOCKED", "reason": str(exc)},
                         sort_keys=True), file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
