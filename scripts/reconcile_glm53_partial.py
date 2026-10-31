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
            if not isinstance(basename, str) or not re.fullmatch(
                r"stage-[0-9]{2}-(?:residual|attention-input|moe-input|kv|latent|dense-intermediate)",
                basename,
            ):
                raise RecoveryError(f"unsafe calibration map name: {basename}")
            artifacts[basename + ".json"] = field + ".json"
            artifacts[basename + ".safetensors"] = field + ".safetensors"
    if record.get("expert_usage"):
        name = record["expert_usage"]
        if not isinstance(name, str) or not re.fullmatch(
            r"stage-[0-9]{2}-expert-usage\.npy", name
        ):
            raise RecoveryError(f"unsafe calibration usage path: {name}")
        artifacts[name] = "expert_usage"
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
            if source.is_symlink() or not source.is_file():
                raise RecoveryError(f"unsafe original map source: {src_name}")
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


def link_overlay(full_dir: Path, original: Path, existing: Path,
                 plan: dict) -> dict[str, int]:
    """Local-only zero-copy workspace; never publish symlinks as durable artifacts.

    Original 24-stage content is checked against its SHA256 integrity receipt.
    Existing full78 stage-50..77 files remain *unverified provenance candidates*.
    """
    if full_dir.resolve() == existing.resolve():
        raise RecoveryError("overlay workspace cannot equal the original HF bucket")
    old_integrity = json.loads((original / ".calibration_integrity.json").read_text())
    originals = legacy_sources(original, integrity=old_integrity)
    full = json.loads((full_dir / "glm53_calibration.json").read_text())
    linked_original = 0
    linked_existing = 0
    verified_existing_overlap = 0
    for stage in plan["calibration"]["present_stages"]:
        record = full["stages"][stage]
        # The 24-stage canonical WARM source maps offer independent,
        # SHA-protected provenance for nine of the saved late stages.
        overlap = originals.get(stage)
        overlap_map = (
            {role: old_name for old_name, role in stage_artifacts(overlap).items()}
            if overlap is not None else {}
        )
        for name, role in stage_artifacts(record).items():
            if overlap is not None:
                old_name = overlap_map[role]
                expected_digest = old_integrity["files"].get(old_name)
                if not expected_digest or digest(existing / name) != expected_digest:
                    raise RecoveryError(
                        f"saved stage {stage} is incompatible with protected 24-stage WARM: {name}"
                    )
                verified_existing_overlap += 1
            dst = full_dir / name
            source = existing / name
            if (not source.is_file() or source.is_symlink() or
                dst.exists() or dst.is_symlink()):
                raise RecoveryError(f"unsafe existing calibration overlay link {name}")
            dst.symlink_to(source.resolve())
            linked_existing += 1
    for stage in plan["calibration"]["old24_sha256_reusable_stages"]:
        newer = stage_artifacts(full["stages"][stage])
        by_role = {
            role: name
            for name, role in stage_artifacts(originals[stage]).items()
        }
        for dest_name, role in newer.items():
            old_name = by_role[role]
            src = original / old_name
            dst = full_dir / dest_name
            if (not src.is_file() or src.is_symlink() or
                dst.exists() or dst.is_symlink()):
                raise RecoveryError(f"unsafe original calibration overlay link {old_name}")
            receipt_hash = old_integrity["files"].get(old_name)
            if not receipt_hash or digest(src) != receipt_hash:
                raise RecoveryError(f"original map hash mismatch {old_name}")
            dst.symlink_to(src.resolve())
            linked_original += 1
    # Lexical/final maps were already saved in the partial full78 calibration.
    for prefix in ("lexical-input", "final-output"):
        for extension in (".json", ".safetensors"):
            name = prefix + extension
            dst = full_dir / name
            src = existing / name
            if (not src.is_file() or src.is_symlink() or
                dst.exists() or dst.is_symlink()):
                raise RecoveryError(f"missing/unsafe full78 global map: {name}")
            original_hash = old_integrity["files"].get(name)
            if original_hash is None or digest(src) != original_hash:
                raise RecoveryError(
                    f"saved {name} mismatches original SHA-verified lexical/final frame"
                )
            dst.symlink_to(src.resolve())
            linked_existing += 1
    return {"hash_verified_original_links": linked_original,
            "existing_linked_files": linked_existing,
            "hash_verified_existing_overlap": verified_existing_overlap,
            "unverified_existing_files": linked_existing - verified_existing_overlap - 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-calibration", required=True, type=Path)
    parser.add_argument("--original-calibration", required=True, type=Path)
    parser.add_argument("--existing-calibration", type=Path,
                        help="Read-only previously uploaded stages, not copied to staging")
    parser.add_argument("--partial-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-old-fingerprint", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply-old24", action="store_true",
                      help="Copy only missing SHA-verified old maps; no overwrites")
    mode.add_argument("--link-overlay", action="store_true",
                      help="Create TEMPORARY symlinks to RO sources; never sync these to HF")
    args = parser.parse_args()
    try:
        if args.link_overlay:
            if args.existing_calibration is None:
                raise RecoveryError("--link-overlay requires --existing-calibration")
            args.partial_calibration.mkdir(parents=True, exist_ok=True)
            manifest = args.partial_calibration / "glm53_calibration.json"
            if not manifest.exists():
                shutil.copyfile(
                    args.existing_calibration / "glm53_calibration.json",
                    manifest,
                )
        plan = inspect(args.partial_calibration, args.original_calibration,
                       args.partial_checkpoint,
                       expected_old_fingerprint=args.expected_old_fingerprint,
                       existing_calibration=args.existing_calibration)
        if args.apply_old24:
            plan["restored_original_files"] = restore_original_maps(
                args.partial_calibration, args.original_calibration, plan,
            )
        if args.link_overlay:
            if args.existing_calibration is None:
                raise RecoveryError("--link-overlay requires --existing-calibration")
            plan["temporary_overlay_links"] = link_overlay(
                args.partial_calibration, args.original_calibration,
                args.existing_calibration, plan,
            )
        # Neither mode promotes the existing partial checkpoint. Full
        # calibration and independent checksums are still mandatory.
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RecoveryError) as exc:
        parser.exit(3, f"RECOVERY_BLOCKED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
