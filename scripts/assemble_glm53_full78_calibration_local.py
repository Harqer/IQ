#!/usr/bin/env python3
"""Finalize complete full78 WARM calibration on a local machine with HF access.

Read the pinned original and 35 saved stage receipts; copy only missing
artifacts server-side to the original public calibration directory, verify
every resulting artifact by SHA-256 readback, and publish one final receipt.
Does NOT compute or publish model weights. Original artifacts are never
deleted or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from huggingface_hub import HfApi, batch_bucket_files, download_bucket_files, list_bucket_tree

from iq_transfer.glm53_partial_repair import _files
from iq_transfer.glm53_partial_recovery_receipts import PINNED_DONOR_REVISION
from scripts.glm53_full78_gate import expected_calibration_files
from scripts.reconcile_glm53_partial import stage_artifacts
from scripts.select_glm53_missing_gpu_stages import ORIGINAL_FINGERPRINT, TARGET_FINGERPRINT

OWNER = "kofecheckmate"
OUTPUT = f"{OWNER}/iq-glm53-all256"
ORIGINAL = f"{OWNER}/mocha"
ORIGINAL_PREFIX = "legacy/visionboxx-macha/calibration"
RUN = "glm53-full78-20261024-c"
PREFIX = f"calibration78/{RUN}"
RECOVERY = f"{PREFIX}/recovery-stage-uploads"
RECEIPT = f"{PREFIX}/calibration_sha256.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(bucket: str, remote: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        bucket, files=[(remote, str(local))], raise_on_missing_files=True
    )
    if not local.is_file() or not local.stat().st_size:
        raise RuntimeError(f"Missing/empty remote artifact: {bucket}/{remote}")
    return local


def inventory(bucket: str, prefix: str) -> dict[str, int]:
    return {
        item.path: item.size
        for item in list_bucket_tree(bucket, prefix=prefix + "/", recursive=True)
        if item.type == "file"
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=Path.home() / "IQ-local-warm-state" / "assembly")
    args = p.parse_args()
    if os.environ.get("IQ_CONFIRM_NO_OTHER_WRITER") != "1":
        p.error("Set IQ_CONFIRM_NO_OTHER_WRITER=1 only after confirming no other checkpoint writer.")
    user = HfApi().whoami()["name"]
    if user.lower() != OWNER.lower():
        p.error(f"HF account is {user}, expected {OWNER}")
    from huggingface_hub import bucket_info
    if bucket_info(OUTPUT).private:
        p.error("Public bucket visibility changed; refusing to assume sufficient storage.")
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    print("IQ_CPU_ASSEMBLY_START bucket=" + OUTPUT, flush=True)

    original_manifest_file = read(
        ORIGINAL, f"{ORIGINAL_PREFIX}/glm53_calibration.json",
        work / "metadata" / "old.json"
    )
    original_integrity_file = read(
        ORIGINAL, f"{ORIGINAL_PREFIX}/.calibration_integrity.json",
        work / "metadata" / "old-integrity.json"
    )
    full_manifest_file = read(
        OUTPUT, f"{PREFIX}/glm53_calibration.json",
        work / "metadata" / "full.json"
    )
    old = json.loads(original_manifest_file.read_text())
    integrity = json.loads(original_integrity_file.read_text())
    full = json.loads(full_manifest_file.read_text())
    if (
        integrity.get("donor_revision") != PINNED_DONOR_REVISION
        or integrity.get("files", {}).get("glm53_calibration.json") != sha(original_manifest_file)
        or old.get("target_config_fingerprint") != ORIGINAL_FINGERPRINT
        or old.get("source_layers") != 78
        or len(old.get("stages", [])) != 24
        or full.get("target_config_fingerprint") != TARGET_FINGERPRINT
        or full.get("source_layers") != 78
        or len(full.get("stages", [])) != 78
        or [int(s["source_layer"]) for s in full["stages"]] != list(range(78))
    ):
        raise RuntimeError("pinned manifest/integrity mismatch: refusing assembly")
    expected = expected_calibration_files(full)
    if len(expected) != 866:
        raise RuntimeError(f"unexpected expected artifact count: {len(expected)}")

    originals = {int(stage["source_layer"]): stage for stage in old["stages"]}
    remote = inventory(OUTPUT, PREFIX)
    original_tree = inventory(ORIGINAL, ORIGINAL_PREFIX)
    if RECEIPT in remote:
        raise RuntimeError("final receipt already exists; verify separately; never overwrite")
    names = set()
    recovered: dict[int, dict[str, str]] = {}
    receipt_paths = [
        name for name in remote
        if re.fullmatch(re.escape(RECOVERY) + r"/stage-\d{2}/receipt\.json", name)
    ]
    if len(receipt_paths) != 35:
        raise RuntimeError(f"requires 35 recovered stage receipts, found {len(receipt_paths)}")
    for remote_name in sorted(receipt_paths):
        match = re.fullmatch(re.escape(RECOVERY) + r"/stage-(\d{2})/receipt\.json", remote_name)
        assert match is not None
        stage = int(match.group(1))
        if stage in originals or stage >= 50:
            raise RuntimeError(f"unexpected recovered stage {stage}")
        data = json.loads(read(OUTPUT, remote_name, work / "receipts" / f"{stage:02d}.json").read_text())
        required = _files(full["stages"][stage])
        if (
            data.get("schema_version") != 1
            or data.get("source_layer") != stage
            or data.get("donor_revision") != PINNED_DONOR_REVISION
            or data.get("target_fingerprint") != TARGET_FINGERPRINT
            or set(data.get("sha256", {})) != required
        ):
            raise RuntimeError(f"bad recovered receipt at stage {stage}")
        for name, digest in data["sha256"].items():
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise RuntimeError(f"invalid SHA at stage {stage}: {name}")
            source = f"{RECOVERY}/stage-{stage:02d}/{name}"
            if remote.get(source, 0) <= 0:
                raise RuntimeError(f"missing saved stage file {source}")
            names.add(name)
        recovered[stage] = data["sha256"]
    remaining = set(range(50)) - set(originals) - set(recovered)
    if remaining:
        raise RuntimeError(f"incomplete calibration, stages still missing: {sorted(remaining)}")
    print(f"IQ_ALL_35_RECOVERED_RECEIPTS_PRESENT={sorted(recovered)}", flush=True)

    # Build an authoritative mapping of expected names to preserved source bytes.
    source_map: dict[str, tuple[str, str, str | None]] = {}
    for name in sorted(expected):
        if name == "glm53_calibration.json":
            source_map[name] = (OUTPUT, f"{PREFIX}/{name}", sha(full_manifest_file))
            continue
        m = re.match(r"stage-(\d{2})-", name)
        verified: str | None = None
        candidate: tuple[str, str] | None = None
        if m:
            stage = int(m.group(1))
            if stage in recovered:
                candidate = (OUTPUT, f"{RECOVERY}/stage-{stage:02d}/{name}")
                verified = recovered[stage].get(name)
            elif stage in originals:
                target_roles = stage_artifacts(full["stages"][stage])
                original_roles = stage_artifacts(originals[stage])
                by_role = {role: path for path, role in original_roles.items()}
                original_name = by_role.get(target_roles[name])
                if original_name is None:
                    raise RuntimeError(f"unmapped old stage file: {name}")
                candidate = (ORIGINAL, f"{ORIGINAL_PREFIX}/{original_name}")
                verified = integrity.get("files", {}).get(original_name)
                if not verified:
                    raise RuntimeError(f"protected original missing SHA: {original_name}")
        elif name.startswith("lexical-input.") or name.startswith("final-output."):
            verified = integrity.get("files", {}).get(name)
        canonical = f"{PREFIX}/{name}"
        if canonical in remote:
            if remote[canonical] <= 0:
                raise RuntimeError(f"empty saved file: {canonical}")
            source_map[name] = (OUTPUT, canonical, verified)
            continue
        if candidate is None:
            raise RuntimeError(f"missing required file without trusted source: {name}")
        source_bucket, source_name = candidate
        tree = remote if source_bucket == OUTPUT else original_tree
        if tree.get(source_name, 0) <= 0:
            raise RuntimeError(f"missing restoration source: {source_bucket}/{source_name}")
        source_map[name] = (source_bucket, source_name, verified)

    # Copy only absent canonical files; copies are server-side on Xet where supported.
    # This can be interrupted safely; reruns recheck remote source/dest inventory.
    absent = [name for name in sorted(expected) if f"{PREFIX}/{name}" not in remote]
    print(f"IQ_CALIBRATION_CANONICAL_MISSING={len(absent)}", flush=True)
    for index, name in enumerate(absent, 1):
        source_bucket, source_path, _ = source_map[name]
        dest_path = f"{PREFIX}/{name}"
        if source_bucket == OUTPUT and source_path == dest_path:
            raise RuntimeError(f"unexpected self copy {name}")
        source_uri = f"hf://buckets/{source_bucket}/{source_path}"
        dest_uri = f"hf://buckets/{OUTPUT}/{dest_path}"
        HfApi().copy_files(source_uri, dest_uri)
        if index % 20 == 0 or index == len(absent):
            print(f"IQ_CALIBRATION_CANONICAL_COPIED={index}/{len(absent)}", flush=True)

    # Independently read/verify every canonical artifact (including 19 saved
    # late stages without original24 overlaps). No SHA receipt is published
    # before the full 866-file readback. Only temporary one-file scratch is used.
    refreshed = inventory(OUTPUT, PREFIX)
    incomplete = sorted(name for name in expected if refreshed.get(f"{PREFIX}/{name}", 0) <= 0)
    if incomplete:
        raise RuntimeError(f"incomplete destination files: {incomplete[:8]} ({len(incomplete)})")
    hashes: dict[str, str] = {}
    for index, name in enumerate(sorted(expected), 1):
        source_bucket, source_path, pinned_hash = source_map[name]
        with tempfile.TemporaryDirectory(prefix="iq-verify-", dir=str(work)) as folder:
            local = read(OUTPUT, f"{PREFIX}/{name}", Path(folder) / name)
            got = sha(local)
        if pinned_hash and pinned_hash != got:
            raise RuntimeError(
                f"SHA mismatch: {name} expected={pinned_hash} got={got} "
                f"source={source_bucket}/{source_path}"
            )
        hashes[name] = got
        if index % 25 == 0 or index == len(expected):
            print(f"IQ_CANONICAL_SHA256_READBACK={index}/{len(expected)}", flush=True)
    body = {
        "schema_version": 1, "source_layers": 78, "hash": "sha256",
        "files": hashes,
    }
    raw = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    latest = inventory(OUTPUT, PREFIX)
    if RECEIPT in latest:
        raise RuntimeError("another writer published the receipt; refusing overwrite")
    batch_bucket_files(OUTPUT, add=[(raw, RECEIPT)])
    confirmation = read(OUTPUT, RECEIPT, work / "final-receipt-readback.json")
    if confirmation.read_bytes() != raw:
        raise RuntimeError("final SHA256 receipt upload did not read back correctly")
    print("IQ_FULL78_866_CALIBRATION_ARTIFACTS_SHA256_VERIFIED", flush=True)
    print("IQ_NEXT_STEP=independent_second_readonly_audit_then_full_GPU_weight_compile",
          flush=True)


if __name__ == "__main__":
    main()
