"""Export a bounded set of IQ-projected experts from one verified GLM stage.

The per-expert projector verifies raw tensor hashes and writes independent
artifacts. This orchestrator verifies the source calibration ONCE per stage,
then reuses the identical basis for every routed expert.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .complete_transplant import canonical_glm53_config
from .glm53_single_expert_projection import (
    GLM53SingleExpertProjectionError,
    _calibration_map,
    _json,
    _sha_file,
    project_one_raw_expert,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def project_raw_stage(
    *,
    raw_dir: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    stage: int = 1,
    experts: int = 4,
    block_rows: int = 64,
) -> dict[str, Any]:
    if type(stage) is not int or not 1 <= stage <= 23:
        raise GLM53SingleExpertProjectionError("routed stage must be 1..23")
    if type(experts) is not int or not 1 <= experts <= 256:
        raise GLM53SingleExpertProjectionError("experts must be 1..256")
    raw_root = Path(raw_dir)
    calib_root = Path(calibration_dir)
    output_root = Path(output_dir)
    bundle_path = raw_root / "raw_expert_manifest.json"
    bundle = _json(bundle_path)
    if (bundle.get("source_repo") != REPO
            or bundle.get("source_revision") != REVISION
            or bundle.get("not_an_iq_checkpoint") is not True):
        raise GLM53SingleExpertProjectionError("incorrect raw GLM donor provenance")
    capacity = bundle.get("target_experts")
    if type(capacity) is not int or capacity not in (64, 128, 256):
        raise GLM53SingleExpertProjectionError("raw export target expert count invalid")
    if bundle.get("target_config_fingerprint") != canonical_glm53_config(
        routed_experts=capacity
    ).fingerprint:
        raise GLM53SingleExpertProjectionError("target model config fingerprint mismatch")
    rows = [v for v in bundle.get("layers", []) if v.get("stage") == stage]
    if len(rows) != 1:
        raise GLM53SingleExpertProjectionError("stage not uniquely identified in raw export")
    row = rows[0]
    ids = row.get("exported_donor_ids", [])
    if not isinstance(ids, list) or len(ids) < experts:
        raise GLM53SingleExpertProjectionError(
            f"stage only contains {len(ids)} previously exported experts; requested {experts}"
        )
    donor_layer = row.get("source_layer")
    if type(donor_layer) is not int or not 3 <= donor_layer < 78:
        raise GLM53SingleExpertProjectionError("invalid donor layer")
    dest = output_root.resolve()
    if any(dest == base or base in dest.parents for base in
           (raw_root.resolve(), calib_root.resolve())):
        raise GLM53SingleExpertProjectionError("projected batch cannot overwrite inputs")
    if output_root.is_symlink() or (
        output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir()))
    ):
        raise GLM53SingleExpertProjectionError("batch output requires a new empty directory")
    # Load and verify the 100 MB canonical WARM map only once.
    checked_map = _calibration_map(calib_root, stage, donor_layer, capacity)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "INCOMPLETE").write_text(
        "Stage projection in progress; not an IQ checkpoint.\n", encoding="utf-8"
    )
    entries = []
    for slot in range(experts):
        slot_dir = output_root / f"slot-{slot:03d}"
        projection = project_one_raw_expert(
            raw_dir=raw_root,
            calibration_dir=calib_root,
            output_dir=slot_dir,
            stage=stage,
            slot=slot,
            block_rows=block_rows,
            _verified_calibration=checked_map,
        )
        entries.append({
            "slot": slot,
            "donor_expert_id": projection["donor_expert_id"],
            "projection_manifest_sha256": _sha_file(
                slot_dir / "projection_manifest.json"
            ),
            "relative_path": f"slot-{slot:03d}",
            "energy": {
                k: v["subspace_retained_energy"]
                for k, v in projection["projection"].items()
            },
        })
    complete_stage = experts == capacity
    summary = {
        "schema_version": 1,
        "artifact_type": "glm53_stage_selected_experts_latent_projection",
        "not_an_iq_checkpoint": True,
        "status": "complete_single_stage" if complete_stage else "partial_stage_experiment",
        "stage": stage,
        "source_layer": donor_layer,
        "target_capacity": capacity,
        "exported_experts": experts,
        "raw_bundle_manifest_sha256": _sha_file(bundle_path),
        "calibration_file_sha256": checked_map[2],
        "entries": entries,
        "note": "Only expert FFNs; no router, shared MLP, attention, or full IQ checkpoint.",
    }
    (output_root / "stage_projection_manifest.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "INCOMPLETE").unlink()
    return summary
