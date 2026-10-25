#!/usr/bin/env python3
"""Resume ONLY the missing GLM-5.3 full78 WARM stages on a local CUDA GPU.

Uses the HF Buckets Python API rather than HF Jobs mounts and uploads each
stage (receipt LAST). Does NOT compile the IQ expert weights. Existing
historical calibration, saved stages and expert shards are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from huggingface_hub import (
    HfApi,
    batch_bucket_files,
    bucket_info,
    download_bucket_files,
    hf_hub_download,
    list_bucket_tree,
)

from iq_transfer.glm53_partial_repair import _files
from scripts.select_glm53_missing_gpu_stages import (
    ORIGINAL_FINGERPRINT,
    TARGET_FINGERPRINT,
)
from iq_transfer.glm53_partial_recovery_receipts import PINNED_DONOR_REVISION

USER = "kofecheckmate"
BUCKET = f"{USER}/iq-glm53-all256"
ORIGINAL_BUCKET = f"{USER}/mocha"
ORIGINAL_PREFIX = "legacy/visionboxx-macha/calibration"
RUN = "glm53-full78-20261024-c"
CALIB_PREFIX = f"calibration78/{RUN}"
RECOVERY_PREFIX = f"{CALIB_PREFIX}/recovery-stage-uploads"
DONOR = "zai-org/GLM-5.3-BF16"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(bucket: str, mappings: list[tuple[str, Path]]) -> None:
    for _remote, local in mappings:
        local.parent.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        bucket,
        files=[(remote, str(local)) for remote, local in mappings],
        raise_on_missing_files=True,
    )
    for _, local in mappings:
        if not local.is_file() or not local.stat().st_size:
            raise RuntimeError(f"missing or empty download: {local}")


def read_metadata(work: Path) -> tuple[dict, set[int]]:
    src = work / "private-metadata"
    manifest = work / "glm53_calibration.json"
    download(ORIGINAL_BUCKET, [
        (f"{ORIGINAL_PREFIX}/glm53_calibration.json", src / "glm53_calibration.json"),
        (f"{ORIGINAL_PREFIX}/.calibration_integrity.json", src / ".calibration_integrity.json"),
    ])
    download(BUCKET, [(f"{CALIB_PREFIX}/glm53_calibration.json", manifest)])

    integrity = json.loads((src / ".calibration_integrity.json").read_text())
    oldpath = src / "glm53_calibration.json"
    old = json.loads(oldpath.read_text())
    full = json.loads(manifest.read_text())
    if (
        integrity.get("donor_revision") != PINNED_DONOR_REVISION
        or integrity.get("files", {}).get("glm53_calibration.json") != sha256(oldpath)
        or old.get("target_config_fingerprint") != ORIGINAL_FINGERPRINT
        or len(old.get("stages", [])) != 24
        or old.get("source_layers") != 78
    ):
        raise RuntimeError("protected original 24-stage calibration metadata mismatch")
    if (
        full.get("schema_version") != 2
        or full.get("source_layers") != 78
        or full.get("target_config_fingerprint") != TARGET_FINGERPRINT
        or full.get("calibration_mode") != "weight_only"
        or len(full.get("stages", [])) != 78
        or [s.get("source_layer") for s in full["stages"]] != list(range(78))
    ):
        raise RuntimeError("saved full78 calibration metadata mismatch")
    old_stages = {int(s["source_layer"]) for s in old["stages"]}
    if len(old_stages) != 24:
        raise RuntimeError("duplicate original source layer")
    return full, old_stages


def inventory(full: dict, work: Path) -> set[int]:
    remote = {
        x.path: x
        for x in list_bucket_tree(BUCKET, prefix=f"{RECOVERY_PREFIX}/", recursive=True)
        if x.type == "file"
    }
    seen_folders = {
        int(m.group(1))
        for name in remote
        if (m := re.fullmatch(
            re.escape(RECOVERY_PREFIX) + r"/stage-(\\d{2})/[^/]+", name
        ))
    }
    if any(i not in range(50) for i in seen_folders):
        raise RuntimeError("unrecognized recovery stage directory")
    recovered = set()
    for i in sorted(seen_folders):
        base = f"{RECOVERY_PREFIX}/stage-{i:02d}"
        required = _files(full["stages"][i])
        receipt_path = f"{base}/receipt.json"
        names = {p.removeprefix(base + "/") for p in remote if p.startswith(base + "/")}
        if names != required | {"receipt.json"}:
            raise RuntimeError(f"remote stage {i} incomplete; refusing overwrite")
        for name in required:
            if not remote[f"{base}/{name}"].size:
                raise RuntimeError(f"zero-byte remote artifact: {base}/{name}")
        receipt_file = work / "receipts" / f"stage-{i:02d}.json"
        download(BUCKET, [(receipt_path, receipt_file)])
        body = json.loads(receipt_file.read_text())
        hashes = body.get("sha256", {})
        if (
            body.get("schema_version") != 1
            or body.get("source_layer") != i
            or body.get("donor_revision") != PINNED_DONOR_REVISION
            or body.get("target_fingerprint") != TARGET_FINGERPRINT
            or set(hashes) != required
            or any(not re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes.values())
        ):
            raise RuntimeError(f"invalid saved recovery receipt: {i}")
        recovered.add(i)
    return recovered


def preflight() -> None:
    if os.getenv("IQ_CONFIRM_NO_OTHER_WRITER") != "1":
        raise RuntimeError("set IQ_CONFIRM_NO_OTHER_WRITER=1 only if no other IQ job writes this bucket")
    account = HfApi().whoami()["name"]
    if account.lower() != USER.lower():
        raise RuntimeError(f"expected HF account {USER}, got {account}")
    if bucket_info(BUCKET).private:
        raise RuntimeError("destination visibility changed; audit private capacity before proceeding")
    import torch
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("CUDA-enabled PyTorch is required; no CPU fallback")
    properties = torch.cuda.get_device_properties(0)
    vram = properties.total_memory / (1024 ** 3)
    if vram < 6.5:
        raise RuntimeError(f"GPU only has {vram:.2f} GiB; expected at least 6.5 GiB")
    print(f"IQ_LOCAL_CUDA_READY={properties.name} vram={vram:.2f}GiB torch={torch.__version__}", flush=True)


def publish(i: int, upload: Path, full: dict, work: Path) -> None:
    stage = upload / f"stage-{i:02d}"
    expected = _files(full["stages"][i])
    if {f.name for f in stage.iterdir() if f.is_file()} != expected | {"receipt.json"}:
        raise RuntimeError(f"locally incomplete stage {i}")
    receipt = json.loads((stage / "receipt.json").read_text())
    if (
        receipt.get("source_layer") != i
        or receipt.get("donor_revision") != PINNED_DONOR_REVISION
        or receipt.get("target_fingerprint") != TARGET_FINGERPRINT
        or set(receipt.get("sha256", {})) != expected
    ):
        raise RuntimeError(f"invalid local receipt for stage {i}")
    for name in expected:
        if sha256(stage / name) != receipt["sha256"][name]:
            raise RuntimeError(f"local stage {i} checksum mismatch: {name}")

    base = f"{RECOVERY_PREFIX}/stage-{i:02d}"
    if list(list_bucket_tree(BUCKET, prefix=base + "/", recursive=True)):
        raise RuntimeError(f"remote stage {i} already exists; refusing overwrite")

    # Publish the data first. A half-uploaded folder has no receipt and will
    # be refused on the next invocation instead of silently reused.
    names = sorted(expected)
    for offset in range(0, len(names), 3):
        subset = names[offset:offset + 3]
        batch_bucket_files(
            BUCKET,
            add=[(str(stage / n), f"{base}/{n}") for n in subset],
        )
    remote = {
        x.path: x
        for x in list_bucket_tree(BUCKET, prefix=base + "/", recursive=True)
        if x.type == "file"
    }
    if set(remote) != {f"{base}/{n}" for n in expected} or any(
        remote[f"{base}/{n}"].size != (stage / n).stat().st_size
        for n in expected
    ):
        raise RuntimeError(f"stage {i} upload sizes incomplete; receipt NOT published")

    # Upload the small commit receipt LAST and read it back.
    batch_bucket_files(
        BUCKET,
        add=[(str(stage / "receipt.json"), f"{base}/receipt.json")],
    )
    copied = work / "readback" / f"stage-{i:02d}.json"
    download(BUCKET, [(f"{base}/receipt.json", copied)])
    if copied.read_bytes() != (stage / "receipt.json").read_bytes():
        raise RuntimeError(f"remote commit receipt mismatch: stage {i}")
    print(f"IQ_LOCAL_DURABLE_STAGE_COMMIT={i}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=Path.home() / "IQ-local-warm-state")
    p.add_argument("--max-stages", type=int, default=11)
    args = p.parse_args()
    if not 1 <= args.max_stages <= 11:
        p.error("--max-stages must be 1..11")
    preflight()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    full, old_stages = read_metadata(work)
    recovered = inventory(full, work)
    missing = [
        i for i in range(50)
        if i not in old_stages and i not in recovered
    ]
    if len(missing) + len(recovered) != 35:
        raise RuntimeError(f"unexpected calibration inventory: {len(missing)} missing, {len(recovered)} saved")
    print(f"IQ_RECOVERY_SAVED={len(recovered)}/35 REMAINING={missing}", flush=True)
    if not missing:
        print("IQ_ALL_35_NEW_WARM_STAGES_SAVED; NEXT: independent 866-file assembly/verification", flush=True)
        return
    if not (work / "glm53_calibration.json").is_file():
        raise RuntimeError("missing public full78 manifest")
    donor = work / "donor"
    donor.mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(
            repo_id=DONOR, filename=name,
            revision=PINNED_DONOR_REVISION, local_dir=str(donor),
        )
    for i in missing[:args.max_stages]:
        with tempfile.TemporaryDirectory(prefix=f"iq-local-warm-{i:02d}-", dir=str(work)) as tmp:
            root = Path(tmp)
            overlay = root / "overlay"
            overlay.mkdir()
            (overlay / "glm53_calibration.json").write_bytes(
                (work / "glm53_calibration.json").read_bytes()
            )
            upload = root / "uploads"
            upload.mkdir()
            cmd = [
                sys.executable, "-m", "iq_transfer.cli",
                "glm53-bootstrap-calibration",
                "--checkpoint", str(donor),
                "--output", str(overlay),
                "--repo-id", DONOR,
                "--checkpoint-revision", PINNED_DONOR_REVISION,
                "--donor-license", "GLM-5.3",
                "--routed-experts", "256", "--full-78",
                "--repair-overlay", "--repair-stage-ids", str(i),
                "--repair-publish-directory", str(upload),
                "--warm-device", "cuda", "--streaming-source",
            ]
            print(f"IQ_LOCAL_WARM_STAGE_START={i}", flush=True)
            subprocess.run(cmd, check=True)
            publish(i, upload, full, work)
    print("IQ_LOCAL_WARM_BATCH_FINISHED; rerun to refresh remote inventory", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"IQ_LOCAL_WARM_BLOCKED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3) from exc
