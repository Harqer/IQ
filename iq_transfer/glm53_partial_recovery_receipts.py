"""Validate and locally rehydrate durable per-stage CUDA WARM recoveries.

Never trust a stage directory without its SHA256 receipt. This is a local
read-only linking operation over an HF bucket mount; it never mutates original
checkpoint weights, original calibration, or saved stage-50..77 data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .glm53_calibration import GLM53CalibrationError
from .glm53_partial_repair import _files

PINNED_DONOR_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


def sha256_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rehydrate_committed_stages(
    overlay: str | Path, public_calibration: str | Path
) -> list[int]:
    overlay = Path(overlay)
    public_calibration = Path(public_calibration)
    manifest = json.loads(
        (overlay / "glm53_calibration.json").read_text(encoding="utf-8")
    )
    stages = manifest.get("stages")
    if (manifest.get("source_layers") != 78 or
        not isinstance(stages, list) or len(stages) != 78):
        raise GLM53CalibrationError("invalid full78 recovery manifest")
    root = public_calibration / "recovery-stage-uploads"
    if not root.exists():
        return []
    recovered = []
    for stage in stages:
        source_layer = int(stage["source_layer"])
        folder = root / f"stage-{source_layer:02d}"
        if not folder.exists():
            continue
        if not folder.is_dir() or folder.is_symlink():
            raise GLM53CalibrationError("invalid staged recovery folder")
        receipt_file = folder / "receipt.json"
        if not receipt_file.is_file() or receipt_file.is_symlink():
            raise GLM53CalibrationError(
                f"uncommitted stage {source_layer}; refuse partial recovery"
            )
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        required = _files(stage)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("source_layer") != source_layer
            or receipt.get("donor_revision") != PINNED_DONOR_REVISION
            or receipt.get("target_fingerprint")
            != manifest.get("target_config_fingerprint")
            or not isinstance(receipt.get("sha256"), dict)
            or set(receipt["sha256"]) != required
        ):
            raise GLM53CalibrationError(
                f"invalid or incomplete stage {source_layer} SHA256 receipt"
            )
        for name in sorted(required):
            src = folder / name
            dest = overlay / name
            if (
                not src.is_file() or src.is_symlink() or
                dest.exists() or dest.is_symlink()
            ):
                raise GLM53CalibrationError(
                    f"stage {source_layer} conflicts with existing calibration: {name}"
                )
            if sha256_digest(src) != receipt["sha256"][name]:
                raise GLM53CalibrationError(
                    f"stored stage {source_layer} SHA mismatch: {name}"
                )
        # Link only once the complete group and all hashes are validated.
        for name in sorted(required):
            (overlay / name).symlink_to((folder / name).resolve())
        recovered.append(source_layer)
    return recovered


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--public-calibration", type=Path, required=True)
    args = p.parse_args()
    try:
        recovered = rehydrate_committed_stages(args.overlay, args.public_calibration)
    except (GLM53CalibrationError, OSError, KeyError, ValueError) as exc:
        p.exit(3, f"BLOCKED_UNTRUSTED_SAVED_WARM_STAGE: {exc}\n")
    print(f"GLM53_VERIFIED_RECOVERED_WARM_STAGES={recovered}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
