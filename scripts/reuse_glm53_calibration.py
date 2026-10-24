#!/usr/bin/env python3
"""Reuse verified weight-only GLM coordinate-map assets for a wider expert bank.

Creates a NEW calibration directory with unchanged maps/usage arrays linked
read-only by path to the original local directory. Nothing is rewritten in the
original. This is NOT activation alignment and does NOT transfer any weights.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from iq_transfer.complete_transplant import canonical_glm53_config, donor_layer_positions


class CalibrationReuseError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_reuse(source: Path, output: Path, *, routed_experts: int, verify_hashes: bool = False) -> dict:
    old = canonical_glm53_config()
    target = canonical_glm53_config(routed_experts=routed_experts)
    if routed_experts <= 32:
        raise CalibrationReuseError("reuse operation requires expanded routed experts")
    if target.schedule != old.schedule or target.mamba3 != old.mamba3:
        raise CalibrationReuseError("Mamba foundation or physical schedule changed")
    if target.model != old.model or target.compressed_context != old.compressed_context:
        raise CalibrationReuseError("lexical, residual, or context geometry changed")
    if output.exists():
        raise CalibrationReuseError("output already exists; refusing to overwrite")
    source = source.resolve(strict=True)
    original_path = source / "glm53_calibration.json"
    original_bytes = original_path.read_bytes()
    manifest = json.loads(original_bytes)
    if manifest.get("schema_version") != 2 or manifest.get("calibration_mode") != "weight_only":
        raise CalibrationReuseError("source must be a valid weight-only GLM calibration")
    if manifest.get("target_config_fingerprint") != old.fingerprint:
        raise CalibrationReuseError("calibration was not built against the canonical 32-expert IQ")
    stages = manifest.get("stages")
    if manifest.get("source_layers") != 78 or not isinstance(stages, list) or len(stages) != 24:
        raise CalibrationReuseError("unexpected GLM source layer count or stage count")
    positions = donor_layer_positions(old)
    required = set()
    for field in ("lexical_input_map", "final_output_map"):
        if not isinstance(manifest.get(field), str):
            raise CalibrationReuseError(f"missing {field}")
        required.update((manifest[field] + ".json", manifest[field] + ".safetensors"))
    all_sources = []
    for stage, record in enumerate(stages):
        if type(record.get("stage")) is not int or record["stage"] != stage:
            raise CalibrationReuseError("source stage numbers not contiguous")
        source_layer = record.get("source_layer")
        if type(source_layer) is not int or not 0 <= source_layer < 78:
            raise CalibrationReuseError("source layer outside GLM donor")
        all_sources.append(source_layer)
        if (record.get("context_physical_layer"), record.get("moe_physical_layer")) != positions[stage]:
            raise CalibrationReuseError("physical stage mapping changed")
        for field in (
            "residual_map", "attention_input_map", "moe_input_map",
            "compressed_kv_map", "latent_map", "dense_intermediate_map",
        ):
            name = record.get(field)
            if name is None:
                if field == "dense_intermediate_map" and source_layer >= 3:
                    continue
                raise CalibrationReuseError(f"missing {field} for stage {stage}")
            if not isinstance(name, str):
                raise CalibrationReuseError(f"invalid {field} path")
            required.update((name + ".json", name + ".safetensors"))
        usage = record.get("expert_usage")
        if source_layer < 3:
            if usage is not None:
                raise CalibrationReuseError("dense donor stage has routed usage")
        elif not isinstance(usage, str):
            raise CalibrationReuseError("sparse donor stage missing expert-usage array")
        if usage:
            required.add(usage)
    if all_sources != sorted(set(all_sources)):
        raise CalibrationReuseError("source layer mapping is not monotonic/unique")
    inventory_path = source / ".calibration_integrity.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    checks = inventory.get("files")
    if not isinstance(checks, dict) or checks.get("glm53_calibration.json") != sha256(original_bytes).hexdigest():
        raise CalibrationReuseError("original calibration manifest not pinned by integrity inventory")
    if not required <= set(checks):
        raise CalibrationReuseError("integrity inventory missing required map/usage files")
    for name in required:
        if not name or Path(name).name != name or name.startswith("."):
            raise CalibrationReuseError("calibration path is not a safe relative filename")
        file = source / name
        if not file.is_file() or not (file.resolve() == source / name):
            raise CalibrationReuseError(f"original calibration file missing or symlinked: {name}")
        if verify_hashes and sha256_file(file) != checks[name]:
            raise CalibrationReuseError(f"original calibration integrity mismatch: {name}")
    updated = dict(manifest)
    updated["target_config_fingerprint"] = target.fingerprint
    updated["calibration_mode"] = "weight_only"
    output.mkdir(parents=True)
    for name in sorted(required):
        os.symlink(source / name, output / name)
    (output / "glm53_calibration.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        "schema_version": 1,
        "status": "reused_weight_only_maps_no_weights_transferred",
        "donor_revision": inventory.get("donor_revision"),
        "source_calibration_manifest_sha256": sha256(original_bytes).hexdigest(),
        "source_config_fingerprint": old.fingerprint,
        "candidate_config_fingerprint": target.fingerprint,
        "routed_experts": routed_experts,
        "unchanged_map_and_usage_files": len(required),
        "original_file_hashes_verified": bool(verify_hashes),
        "relies_on_original_calibration_path": str(source),
        "not_self_contained": True,
        "activation_refinement_performed": False,
    }
    (output / "reuse_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="LOCAL original calibration folder")
    p.add_argument("--output", type=Path, required=True, help="NEW local candidate folder")
    p.add_argument("--routed-experts", type=int, choices=(64, 128, 256), default=64)
    p.add_argument("--verify-hashes", action="store_true", help="SHA256 all map tensors (slow; strongly recommended)")
    args = p.parse_args()
    try:
        result = prepare_reuse(
            args.source, args.output,
            routed_experts=args.routed_experts,
            verify_hashes=args.verify_hashes,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Refusing calibration reuse: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Calibration reused, NOT weights transferred. Output symlinks require source to remain accessible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
