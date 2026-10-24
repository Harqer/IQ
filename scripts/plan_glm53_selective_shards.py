#!/usr/bin/env python3
"""Plan exact pinned GLM-5.3-BF16 shards needed for a 64-expert IQ compile.

No donor weight downloads, model loading, CUDA, or writes to the user's bucket.
Reads only official config/index and old calibration manifest + 23 usage arrays.
The separate old 82-shard index MUST NOT be used for the donor source.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from iq_transfer.complete_transplant import canonical_glm53_config, donor_layer_positions
from iq_transfer.glm53 import GLM53Inspector
from iq_transfer.glm53_moe import select_experts_by_usage
from iq_transfer.glm53_shards import _compile_tensor_keys, _shards_for_keys


REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


class SelectivePlanError(ValueError):
    pass


def build_selective_plan(
    donor_config: dict,
    official_index: dict,
    calibration_manifest: dict,
    integrity_manifest: dict,
    usage_payloads: dict[str, bytes],
    transplant_manifest: dict,
    *,
    routed_experts: int,
) -> dict:
    if routed_experts not in (64, 128, 256) or type(routed_experts) is not int:
        raise SelectivePlanError("expanded expert count must be 64, 128, or 256")
    inspector = GLM53Inspector.from_config_mapping(donor_config)
    if (
        donor_config.get("model_type") != "glm_moe_dsa"
        or donor_config.get("num_hidden_layers") != 78
        or donor_config.get("n_routed_experts") != 256
        or donor_config.get("first_k_dense_replace") != 3
        or donor_config.get("quantization_config") is not None
    ):
        raise SelectivePlanError("donor is not the pinned BF16 GLM-5.3 architecture")
    if (
        transplant_manifest.get("capability_source", {}).get("revision") != REVISION
        or integrity_manifest.get("donor_revision") != REVISION
        or transplant_manifest.get("capability_source", {}).get("repo")
        != "zai-org/GLM-5.3-BF16"
    ):
        raise SelectivePlanError("mismatched original GLM donor revision")
    base = canonical_glm53_config()
    target = canonical_glm53_config(routed_experts=routed_experts)
    if (
        calibration_manifest.get("schema_version") != 2
        or calibration_manifest.get("calibration_mode") != "weight_only"
        or calibration_manifest.get("source_layers") != 78
        or calibration_manifest.get("target_config_fingerprint") != base.fingerprint
    ):
        raise SelectivePlanError("old GLM calibration is not the canonical weight-only one")
    if transplant_manifest.get("recipient_config_fingerprint") != base.fingerprint:
        raise SelectivePlanError("old IQ checkpoint fingerprint differs from calibration")
    stages = calibration_manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != 24:
        raise SelectivePlanError("expected exactly 24 mapped GLM stages")
    old_map = transplant_manifest.get("source_layer_map")
    positions = donor_layer_positions(base)
    files = integrity_manifest.get("files")
    if not isinstance(files, dict):
        raise SelectivePlanError("original calibration has no file hash inventory")
    parsed = []
    chosen_report = []
    source_layers = []
    for i, row in enumerate(stages):
        if type(row.get("stage")) is not int or row["stage"] != i:
            raise SelectivePlanError("non-contiguous source stages")
        layer = row.get("source_layer")
        if type(layer) is not int or not 0 <= layer < 78:
            raise SelectivePlanError("invalid source layer")
        source_layers.append(layer)
        if old_map is None or old_map.get(str(i)) != layer:
            raise SelectivePlanError("source-layer mapping differs from compiled IQ checkpoint")
        if (
            row.get("context_physical_layer"), row.get("moe_physical_layer")
        ) != positions[i]:
            raise SelectivePlanError("physical context/MoE stage mapping differs")
        usage_name = row.get("expert_usage")
        if layer < 3:
            if usage_name is not None:
                raise SelectivePlanError("dense source layer cannot have expert usage")
            usage = None
            chosen_ids = None
        else:
            if not isinstance(usage_name, str) or Path(usage_name).name != usage_name:
                raise SelectivePlanError("unsafe or absent usage filename")
            payload = usage_payloads.get(usage_name)
            digest = files.get(usage_name)
            if not payload or not digest or sha256(payload).hexdigest() != digest:
                raise SelectivePlanError(f"unverified calibration usage: {usage_name}")
            usage = np.load(BytesIO(payload), allow_pickle=False)
            if usage.shape != (256,) or not np.isfinite(usage).all():
                raise SelectivePlanError(f"invalid expert usage vector: {usage_name}")
            chosen_ids = select_experts_by_usage(usage, target_experts=routed_experts)
        parsed.append(SimpleNamespace(
            stage=i, source_layer=layer,
            context_physical_layer=row["context_physical_layer"],
            expert_usage=usage,
        ))
        chosen_report.append(None if chosen_ids is None else list(chosen_ids))
    if source_layers != sorted(set(source_layers)):
        raise SelectivePlanError("GLM source mapping must be monotonic and unique")
    selected = _compile_tensor_keys(
        calibration=SimpleNamespace(stages=parsed),
        inspector=inspector, target_config=target,
    )
    weight_map = official_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SelectivePlanError("official donor index has no weight map")
    shards = _shards_for_keys(weight_map, selected)
    return {
        "schema_version": 1,
        "status": "selective_metadata_plan_only",
        "source_revision": REVISION,
        "target_experts": routed_experts,
        "target_config_fingerprint": target.fingerprint,
        "source_layers": source_layers,
        "selected_donor_expert_ids": chosen_report,
        "tensor_keys_required": len(selected),
        "official_total_shards": len(set(weight_map.values())),
        "selective_shards_required": len(shards),
        "official_shards": list(shards),
        "note": (
            "No weights downloaded. Shard-level downloads may contain additional "
            "unused tensors sharing the same file. Local 82-shard index is not the "
            "official donor and must not be used as the compiler source."
        ),
    }


def read_text_uri(uri: str) -> bytes:
    if uri.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(uri, "rb") as handle:
            return handle.read()
    return Path(uri).read_bytes()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor-config", required=True, help="official pinned donor config.json")
    p.add_argument("--donor-index", required=True, help="official pinned 282-shard index JSON")
    p.add_argument("--calibration", required=True, help="local path or HF bucket prefix (not manifest file)")
    p.add_argument("--transplant-manifest", required=True, help="original IQ transplant_manifest.json")
    p.add_argument("--routed-experts", type=int, choices=(64, 128, 256), default=64)
    p.add_argument("--output", type=Path, help="new JSON filename for exact shard plan")
    a = p.parse_args()
    try:
        base = a.calibration.rstrip("/")
        calibration_raw = read_text_uri(base + "/glm53_calibration.json")
        calibration = json.loads(calibration_raw)
        integrity = json.loads(read_text_uri(base + "/.calibration_integrity.json"))
        if sha256(calibration_raw).hexdigest() != integrity.get("files", {}).get("glm53_calibration.json"):
            raise SelectivePlanError("original calibration manifest SHA mismatch")
        names = [
            x["expert_usage"] for x in calibration["stages"]
            if x.get("expert_usage") is not None
        ]
        if len(names) != 23 or len(set(names)) != 23:
            raise SelectivePlanError("expected 23 independently captured/prior expert-usage arrays")
        usage_bytes = {name: read_text_uri(base + "/" + name) for name in names}
        result = build_selective_plan(
            json.loads(read_text_uri(a.donor_config)),
            json.loads(read_text_uri(a.donor_index)),
            calibration,
            integrity,
            usage_bytes,
            json.loads(read_text_uri(a.transplant_manifest)),
            routed_experts=a.routed_experts,
        )
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("official_shards", "selected_donor_expert_ids")},
                         indent=2, sort_keys=True))
        if a.output is not None:
            if a.output.exists():
                raise SelectivePlanError("output exists; cannot overwrite existing plan")
            with a.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
            print(f"Wrote exact source shard names and expert IDs to {a.output}")
        else:
            print("Pass --output PATH to save exact source shard filenames.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Selective GLM expansion planning FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
