#!/usr/bin/env python3
"""Plan/restore only missing GLM full78 calibration maps from verified 24-stage originals.

No donor downloads, GPU actions or changed existing files. Every restored
file is byte-for-byte copied from the SHA256-protected original calibration,
and published only when its destination is absent.

Remaining source layers must be generated separately by WARM. An existing
partial checkpoint remains UNTRUSTED until source-provenance and output-value
checks succeed; this tool deliberately does not adopt or overwrite it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


MAP_FIELDS = (
    "residual_map", "attention_input_map", "moe_input_map",
    "compressed_kv_map", "latent_map", "dense_intermediate_map",
)
STAGE_EXPERT_RE = re.compile(r"model-moe-(\d{3})-expert-(\d{3})\.safetensors$")
DONOR_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


class RecoveryError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def stage_artifacts(record: dict) -> dict[str, str]:
    """Map destination filenames to the map role, independent of stage number."""
    artifacts = {}
    for field in MAP_FIELDS:
        basename = record.get(field)
        if basename:
            artifacts[basename + ".json"] = field + ".json"
            artifacts[basename + ".safetensors"] = field + ".safetensors"
    if record.get("expert_usage"):
        artifacts[record["expert_usage"]] = "expert_usage"
    return artifacts


def legacy_sources(root: Path, *, integrity: dict) -> dict[int, dict]:
    old_manifest_path = root / "glm53_calibration.json"
    if digest(old_manifest_path) != integrity["files"].get("glm53_calibration.json"):
        raise RecoveryError("original 24-stage calibration manifest hash mismatch")
    j = json.loads(old_manifest_path.read_text())
    if (j.get("schema_version") != 2 or j.get("source_layers") != 78 or
        j.get("calibration_mode") != "weight_only" or
        len(j.get("stages", [])) != 24):
        raise RecoveryError("incorrect 24-stage original calibration")
    return {int(s["source_layer"]): s for s in j["stages"]}


def inspect(full_dir: Path, original: Path, checkpoint: Path,
            *, expected_old_fingerprint: str | None = None,
            existing_calibration: Path | None = None) -> dict:
    full_file = full_dir / "glm53_calibration.json"
    full = json.loads(full_file.read_text())
    records = full.get("stages", [])
    if (full.get("schema_version") != 2 or full.get("source_layers") != 78 or
        full.get("calibration_mode") != "weight_only" or len(records) != 78 or
        [s.get("stage") for s in records] != list(range(78)) or
        [s.get("source_layer") for s in records] != list(range(78))):
        raise RecoveryError("invalid full78 calibration manifest")
    integrity = json.loads((original / ".calibration_integrity.json").read_text())
    if integrity.get("donor_revision") != DONOR_REVISION:
        raise RecoveryError("original calibration donor revision mismatch")
    old_source = legacy_sources(original, integrity=integrity)
    if expected_old_fingerprint:
        old = json.loads((original / "glm53_calibration.json").read_text())
        if old.get("target_config_fingerprint") != expected_old_fingerprint:
            raise RecoveryError("original IQ calibration target fingerprint mismatch")
    reusable: list[int] = []
    needs_compute: list[int] = []
    existing: list[int] = []
    incomplete: list[int] = []
    old_copy_files = 0
    for record in records:
        stage = int(record["stage"])
        available = stage_artifacts(record)
        def exists(name: str) -> bool:
            return any(
                (directory / name).is_file() and (directory / name).stat().st_size > 0
                for directory in (full_dir, existing_calibration)
                if directory is not None
            )
        present = [name for name in available if exists(name)]
        if len(present) == len(available):
            existing.append(stage)
        elif present:
            incomplete.append(stage)
        elif stage in old_source:
            old = stage_artifacts(old_source[stage])
            if set(available.values()) != set(old.values()):
                raise RecoveryError(f"original stage {stage} uses incompatible map roles")
            reusable.append(stage)
            old_copy_files += len(available)
        else:
            needs_compute.append(stage)
    if incomplete:
        raise RecoveryError(f"partially written stages require manual audit: {incomplete}")
    shards = sorted(checkpoint.glob("model-moe-*-expert-*.safetensors"))
    donor_slots: dict[int, list[int]] = {}
    for path in shards:
        match = STAGE_EXPERT_RE.fullmatch(path.name)
        if not match or path.is_symlink():
            raise RecoveryError("invalid expert shard in checkpoint")
        donor_slots.setdefault(int(match[1]), []).append(int(match[2]))
    if len(donor_slots) > 75 or any(len(set(ids)) != len(ids) for ids in donor_slots.values()):
        raise RecoveryError("invalid saved expert slot inventory")
    return {
        "schema_version": 1,
        "donor_revision": DONOR_REVISION,
        "target_fingerprint": full["target_config_fingerprint"],
        "calibration": {
            "present_stages": existing,
            "old24_sha256_reusable_stages": reusable,
            "must_compute_stages": needs_compute,
            "old24_reusable_files": old_copy_files,
            "missing_files_before": sum(
                len(stage_artifacts(s)) for s in records
                if s["stage"] in reusable or s["stage"] in needs_compute
            ),
            "calibration_receipt_present": (full_dir / "calibration_sha256.json").is_file(),
        },
        "checkpoint": {
            "candidate_expert_slots": {str(k): sorted(v) for k, v in donor_slots.items()},
            "candidate_expert_count": len(shards),
            "provenance_present": (checkpoint / "run-provenance.json").is_file(),
            "promoted_shards": 0,
        },
    }


def restore_original_maps(full_dir: Path, original: Path, plan: dict) -> int:
    """Copy only SHA256-protected original maps; fail on unexpected originals."""
    integrity = json.loads((original / ".calibration_integrity.json").read_text())
    originals = legacy_sources(original, integrity=integrity)
    full = json.loads((full_dir / "glm53_calibration.json").read_text())
    restored = 0
    for stage in plan["calibration"]["old24_sha256_reusable_stages"]:
        new = stage_artifacts(full["stages"][stage])
        old = stage_artifacts(originals[stage])
        by_role = {role: filename for filename, role in old.items()}
        for name, role in new.items():
            src_name = by_role[role]
            source = original / src_name
            destination = full_dir / name
            if destination.exists() or destination.is_symlink():
                raise RecoveryError(f"refusing overwrite of existing calibration: {name}")
            expected_hash = integrity["files"].get(src_name)
            if expected_hash is None or digest(source) != expected_hash:
                raise RecoveryError(f"original map integrity mismatch: {src_name}")
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".partial", dir=str(full_dir),
            )
            try:
                with os.fdopen(fd, "wb") as target, source.open("rb") as inp:
                    shutil.copyfileobj(inp, target, length=4 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if digest(Path(temp_name)) != expected_hash:
                    raise RecoveryError(f"copied calibration SHA mismatch: {name}")
                if destination.exists() or destination.is_symlink():
                    raise RecoveryError(f"concurrent writer detected: {name}")
                os.replace(temp_name, destination)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            restored += 1
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-calibration", required=True, type=Path)
    parser.add_argument("--original-calibration", required=True, type=Path)
    parser.add_argument("--existing-calibration", type=Path,
                        help="Read-only previously uploaded stages, not copied to staging")
    parser.add_argument("--partial-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-old-fingerprint", default=None)
    parser.add_argument("--apply-old24", action="store_true",
                        help="Only copy missing SHA-verified old24 stage maps; no overwrites")
    args = parser.parse_args()
    try:
        plan = inspect(args.partial_calibration, args.original_calibration,
                       args.partial_checkpoint,
                       expected_old_fingerprint=args.expected_old_fingerprint,
                       existing_calibration=args.existing_calibration)
        if args.apply_old24:
            plan["restored_original_files"] = restore_original_maps(
                args.partial_calibration, args.original_calibration, plan,
            )
            # Do not silently declare full calibration complete; other stages
            # still require WARM and independent 866-file SHA verification.
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RecoveryError) as exc:
        parser.exit(3, f"RECOVERY_BLOCKED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
