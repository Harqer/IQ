#!/usr/bin/env python3
"""Select genuine missing full78 WARM source layers without hashing 16 GB on GPU.

The 24-stage source maps and stage-50..77 checkpoint contents were audited
independently on CPU already. This selector does not *promote* those assets:
it only chooses which NEW source layers to compute. Their full original
SHA256 validation remains mandatory before final full78 compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from iq_transfer.glm53_partial_repair import _files
from iq_transfer.glm53_partial_recovery_receipts import PINNED_DONOR_REVISION

TARGET_FINGERPRINT = "6c292b471e85dfd97d2210087af68b247bc46702e75e5277d08f3ee7c49e788d"
ORIGINAL_FINGERPRINT = "9cba3cfd5aded22306a1c3f52f9096b70c96fb18a87929daeafe7e4c47542611"


def choose_missing_stages(original: Path, public: Path, checkpoint: Path,
                          maximum: int) -> list[int]:
    if not 1 <= maximum <= 8:
        raise ValueError("batch must contain 1..8 source stages")
    if not checkpoint.is_dir():
        raise ValueError("original partial IQ checkpoint is missing")
    original_manifest = original / "glm53_calibration.json"
    integrity = json.loads((original / ".calibration_integrity.json").read_text())
    manifest_bytes = original_manifest.read_bytes()
    if integrity.get("donor_revision") != PINNED_DONOR_REVISION or (
        hashlib.sha256(manifest_bytes).hexdigest()
        != integrity.get("files", {}).get("glm53_calibration.json")
    ):
        raise ValueError("original protected WARM manifest fingerprint mismatch")
    old = json.loads(manifest_bytes)
    if old.get("target_config_fingerprint") != ORIGINAL_FINGERPRINT:
        raise ValueError("original 24-stage target fingerprint changed")
    original_sources = {int(s["source_layer"]) for s in old.get("stages", [])}
    if len(original_sources) != 24:
        raise ValueError("incorrect original calibrated source-layer coverage")

    full = json.loads((public / "glm53_calibration.json").read_text())
    stages = full.get("stages", [])
    if (full.get("schema_version") != 2 or
        full.get("source_layers") != 78 or
        full.get("calibration_mode") != "weight_only" or
        full.get("target_config_fingerprint") != TARGET_FINGERPRINT or
        len(stages) != 78 or
        [s.get("source_layer") for s in stages] != list(range(78))):
        raise ValueError("saved full78 manifest or target geometry changed")
    for i in range(50, 78):
        record = stages[i]
        names = _files(record)
        if not names or not all(
            (public / name).is_file() and (public / name).stat().st_size > 0
            for name in names
        ):
            raise ValueError(f"saved late source stage {i} is incomplete")
    upload_root = public / "recovery-stage-uploads"
    committed = set()
    if upload_root.exists():
        for i, stage in enumerate(stages):
            folder = upload_root / f"stage-{i:02d}"
            if not folder.exists():
                continue
            receipt = folder / "receipt.json"
            if not receipt.is_file() or folder.is_symlink():
                raise ValueError(f"incomplete/unsafe previously staged source layer {i}")
            body = json.loads(receipt.read_text())
            expected = _files(stage)
            if (body.get("source_layer") != i or
                body.get("donor_revision") != PINNED_DONOR_REVISION or
                body.get("target_fingerprint") != TARGET_FINGERPRINT or
                body.get("schema_version") != 1 or
                not isinstance(body.get("sha256"), dict) or
                set(body["sha256"]) != expected or
                not all(
                    (folder / name).is_file()
                    and (folder / name).stat().st_size > 0
                    for name in expected
                )):
                raise ValueError(f"untrusted source layer {i} recovery receipt")
            committed.add(i)
    remaining = [i for i in range(78)
                 if i not in original_sources
                 and i < 50
                 and i not in committed]
    selected = remaining[:maximum]
    if not selected:
        raise ValueError("all new WARM stage candidates already have receipts")
    print(json.dumps({
        "selected_new_source_layers": selected,
        "still_missing_after_selection": remaining[len(selected):],
        "reusable_original_source_layers": sorted(original_sources),
        "saved_late_stage_count": 28,
        "previously_staged_receipts": len(committed),
        "note": "Original and late-stage full SHA256 checks remain final compile gates.",
    }, sort_keys=True), file=sys.stderr, flush=True)
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", required=True, type=Path)
    p.add_argument("--public", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--max-stages", required=True, type=int)
    args = p.parse_args()
    try:
        selected = choose_missing_stages(args.original, args.public,
                                        args.checkpoint, args.max_stages)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(3, f"BLOCKED_UNTRUSTED_GPU_STAGE_SELECTION: {exc}\n")
    print(",".join(str(s) for s in selected), flush=True)


if __name__ == "__main__":
    main()
