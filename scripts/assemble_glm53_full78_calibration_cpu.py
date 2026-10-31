#!/usr/bin/env python3
"""Assemble verified 78-stage calibration maps without wasting GPU on recomputation.

Run on a CPU worker AFTER the last H200 calibration job has stopped.
Original 24-stage maps and the existing late 28 stages are never overwritten.
The 35 new CUDA stage groups are independently checked against their SHA256
receipts before being copied into the canonical 866-file calibration root.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from scripts.glm53_full78_gate import expected_calibration_files

EXPECTED_STAGES = 35
EXPECTED_FILES = 866
DEFAULT_RUN = "glm53-full78-20261024-c"
ORIGINAL = Path("/hf-bucket/legacy/visionboxx-macha/calibration")
ROOT = Path("/out/calibration78")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def run(run_id: str) -> None:
    if os.getenv("IQ_CONFIRM_NO_OTHER_WRITER") != "1":
        raise RuntimeError("exclusive writer confirmation required after GPU job exit")
    if os.getenv("IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT") != "1":
        raise RuntimeError("public HF storage policy acknowledgment required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{4,79}", run_id):
        raise RuntimeError("invalid RUN_ID")
    existing = ROOT / run_id
    partial_iq = Path("/out/IQ/runs") / run_id
    upload = existing / "recovery-stage-uploads"
    if not ORIGINAL.is_dir() or not existing.is_dir() or not partial_iq.is_dir():
        raise RuntimeError("missing immutable original or partial calibration/checkpoint mounts")
    receipts = sorted(upload.glob("stage-*/receipt.json"))
    if len(receipts) != EXPECTED_STAGES:
        raise RuntimeError(f"remaining uncomputed WARM stages: {EXPECTED_STAGES - len(receipts)}")
    if (existing / "calibration_sha256.json").exists():
        raise RuntimeError("existing final receipt: independently verify; never overwrite")

    with tempfile.TemporaryDirectory(prefix="iq78-cpu-assemble-") as scratch:
        overlay = Path(scratch) / "calibration"
        overlay.mkdir()
        shutil.copyfile(existing / "glm53_calibration.json", overlay / "glm53_calibration.json")
        # Independently hash the reused 24-stage originals and late-stage overlaps.
        subprocess.run([
            sys.executable, "-m", "scripts.reconcile_glm53_partial",
            "--partial-calibration", str(overlay),
            "--original-calibration", str(ORIGINAL),
            "--existing-calibration", str(existing),
            "--partial-checkpoint", str(partial_iq),
            "--expected-old-fingerprint",
            "9cba3cfd5aded22306a1c3f52f9096b70c96fb18a87929daeafe7e4c47542611",
            "--link-overlay",
        ], check=True)
        # Validate each of the 35 saved stage receipts and every actual tensor hash.
        subprocess.run([
            sys.executable, "-m", "iq_transfer.glm53_partial_recovery_receipts",
            "--overlay", str(overlay),
            "--public-calibration", str(existing),
        ], check=True)

        import json
        manifest = json.loads((overlay / "glm53_calibration.json").read_text())
        names = expected_calibration_files(manifest)
        if len(names) != EXPECTED_FILES or any(
            not (overlay / name).is_file() for name in names
        ):
            raise RuntimeError("not all 866 full78 calibration artifacts are verified")

        missing = sorted(n for n in names if not (existing / n).is_file())
        print(f"IQ_FULL78_CPU_ASSEMBLY_VERIFIED_SOURCE_FILES={len(names)}", flush=True)
        print(f"IQ_FULL78_CPU_ASSEMBLY_MISSING_DESTINATIONS={len(missing)}", flush=True)
        for index, name in enumerate(missing, 1):
            source = (overlay / name).resolve(strict=True)
            dest = existing / name
            if source.is_dir() or not source.is_file() or dest.exists() or dest.is_symlink():
                raise RuntimeError(f"unsafe source or conflicting destination: {name}")
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".partial", dir=str(existing))
            try:
                with os.fdopen(fd, "wb") as output, source.open("rb") as inp:
                    shutil.copyfileobj(inp, output, length=4 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if digest(Path(temporary)) != digest(source):
                    raise RuntimeError(f"published stage copy mismatch: {name}")
                if dest.exists() or dest.is_symlink():
                    raise RuntimeError(f"concurrent output writer: {name}")
                os.replace(temporary, dest)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            if index % 20 == 0:
                print(f"IQ_FULL78_CPU_ASSEMBLY_PUBLISHED={index}/{len(missing)}", flush=True)

    subprocess.run([sys.executable, "-m", "scripts.glm53_full78_gate",
                    "calibration", "--root", str(existing)], check=True)
    subprocess.run([sys.executable, "-m", "scripts.glm53_calibration_sha256",
                    "make", "--root", str(existing)], check=True)
    print("IQ_ALL_866_CALIBRATION_FILES_ASSEMBLED_WITH_SHA256_RECEIPT", flush=True)
    print("IQ_NEXT_STEP=independent_read_only_verification_before_GPU_compilation",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=os.getenv("RUN_ID", DEFAULT_RUN))
    args = p.parse_args()
    try:
        run(args.run_id)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(3, f"BLOCKED_FULL78_CPU_ASSEMBLY: {exc}\n")
