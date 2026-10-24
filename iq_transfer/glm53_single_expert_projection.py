"""Project one verified raw GLM expert into IQ's calibrated Stable LatentMoE basis.

This writes ONLY a three-weight experiment shard, never a complete IQ
checkpoint. Original BF16 weights and their donor bundle remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .complete_transplant import canonical_glm53_config, donor_layer_positions
from .transport import load_coordinate_map
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class GLM53SingleExpertProjectionError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GLM53SingleExpertProjectionError(f"invalid JSON object: {path}")
    return value


def _calibration_map(
    root: Path, stage: int, source_layer: int, capacity: int,
) -> tuple[np.ndarray, int, dict[str, str]]:
    manifest_path = root / "glm53_calibration.json"
    manifest = _json(manifest_path)
    records = manifest.get("stages")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("source_layers") != 78
        or not isinstance(records, list)
        or len(records) != 24
        or not isinstance(records[stage], dict)
    ):
        raise GLM53SingleExpertProjectionError("invalid calibration stage inventory")
    entry = records[stage]
    if entry.get("stage") != stage or entry.get("source_layer") != source_layer:
        raise GLM53SingleExpertProjectionError("raw expert and calibration source layer disagree")

    config = canonical_glm53_config(routed_experts=capacity)
    expected_position = donor_layer_positions(config)[stage][1]
    if entry.get("moe_physical_layer") != expected_position:
        raise GLM53SingleExpertProjectionError("calibration recipient MoE position mismatch")
    fp = manifest.get("target_config_fingerprint")
    old_fp = canonical_glm53_config().fingerprint
    if fp != config.fingerprint and not (
        fp == old_fp and manifest.get("calibration_mode") == "weight_only"
    ):
        raise GLM53SingleExpertProjectionError(
            "calibration fingerprint/mode is incompatible with target configuration"
        )
    stem = f"stage-{stage:02d}-latent"
    if entry.get("latent_map") != stem:
        raise GLM53SingleExpertProjectionError("calibration latent map filename mismatch")
    file_names = ("glm53_calibration.json", f"{stem}.json", f"{stem}.safetensors")
    integrity = _json(root / ".calibration_integrity.json")
    expected_hashes = integrity.get("files")
    if not isinstance(expected_hashes, dict):
        raise GLM53SingleExpertProjectionError("calibration hash inventory is missing")
    verified = {}
    for name in file_names:
        path = root / name
        expected = expected_hashes.get(name)
        digest = _sha_file(path)
        if not isinstance(expected, str) or expected != digest:
            raise GLM53SingleExpertProjectionError(f"calibration integrity mismatch: {name}")
        verified[name] = digest
    coord = load_coordinate_map(root / stem)
    p = np.asarray(coord.matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape != (
        config.model.hidden_size * 3 // 2,
        config.stable_moe.latent_size,
    ):
        # Canonical IQ=4096, GLM donor=6144. Check exact target geometry.
        raise GLM53SingleExpertProjectionError("unexpected latent coordinate geometry")
    if not np.isfinite(p).all():
        raise GLM53SingleExpertProjectionError("nonfinite latent map")
    # Verify every Gram column, with bounded intermediate memory.
    for start in range(0, p.shape[1], 64):
        stop = min(start + 64, p.shape[1])
        gram = p.T @ p[:, start:stop]
        for offset, col in enumerate(range(start, stop)):
            gram[col, offset] -= 1.0
        if np.max(np.abs(gram)) > 1e-8:
            raise GLM53SingleExpertProjectionError(
                "latent map is not column-orthonormal"
            )
    return p, expected_position, verified


def _read_donor(
    root: Path, manifest: dict[str, Any], key: str, shape: tuple[int, int],
) -> torch.Tensor:
    mapping = manifest["weight_map"]
    if key not in mapping:
        raise GLM53SingleExpertProjectionError(f"selected donor tensor absent: {key}")
    filename = mapping[key]
    if Path(filename).name != filename:
        raise GLM53SingleExpertProjectionError("unsafe donor shard path")
    path = root / filename
    if _sha_file(path) != manifest["file_sha256"][filename]:
        raise GLM53SingleExpertProjectionError(f"donor file SHA mismatch: {filename}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise GLM53SingleExpertProjectionError(f"donor tensor missing from file: {key}")
        tensor = handle.get_tensor(key)
    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
        raise GLM53SingleExpertProjectionError(f"unexpected BF16 donor tensor shape: {key}")
    if _sha_tensor(tensor) != manifest["tensor_sha256"][key]:
        raise GLM53SingleExpertProjectionError(f"donor tensor SHA mismatch: {key}")
    if not bool(torch.isfinite(tensor).all()):
        raise GLM53SingleExpertProjectionError(f"nonfinite donor tensor: {key}")
    return tensor


def _project_gate_or_up(
    weight: torch.Tensor, p: np.ndarray, *,
    block_rows: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = torch.empty((weight.shape[0], p.shape[1]), dtype=torch.bfloat16)
    source_sq = projected_sq = 0.0
    for start in range(0, weight.shape[0], block_rows):
        stop = min(start + block_rows, weight.shape[0])
        chunk = weight[start:stop].float().numpy().astype(np.float64)
        result = chunk @ p
        source_sq += float(np.sum(chunk * chunk))
        projected_sq += float(np.sum(result * result))
        output[start:stop] = torch.from_numpy(result).to(torch.bfloat16)
    return output, _projection_metrics(source_sq, projected_sq)


def _project_down(
    weight: torch.Tensor, p: np.ndarray, *,
    block_rows: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    projected = np.zeros((p.shape[1], weight.shape[1]), dtype=np.float64)
    source_sq = 0.0
    for start in range(0, weight.shape[0], block_rows):
        stop = min(start + block_rows, weight.shape[0])
        chunk = weight[start:stop].float().numpy().astype(np.float64)
        source_sq += float(np.sum(chunk * chunk))
        projected += p[start:stop].T @ chunk
    projected_sq = float(np.sum(projected * projected))
    return torch.from_numpy(projected).to(torch.bfloat16), _projection_metrics(
        source_sq, projected_sq
    )


def _projection_metrics(donor_sq: float, projected_sq: float) -> dict[str, float]:
    if donor_sq <= 0 or projected_sq < 0 or not np.isfinite(donor_sq + projected_sq):
        raise GLM53SingleExpertProjectionError("invalid donor/projection energy")
    retention = projected_sq / donor_sq
    if retention > 1.000001:
        raise GLM53SingleExpertProjectionError("orthogonal projection unexpectedly increases norm")
    return {
        "donor_frobenius_sq": donor_sq,
        "projected_frobenius_sq": projected_sq,
        "subspace_retained_energy": min(retention, 1.0),
        "subspace_discarded_energy": max(0.0, 1.0 - retention),
    }


def project_one_raw_expert(
    *,
    raw_dir: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    stage: int = 1,
    slot: int = 0,
    block_rows: int = 64,
) -> dict[str, Any]:
    """Project an indexed expert using one verified WARM basis, without retraining."""
    if type(stage) is not int or not 0 <= stage < 24:
        raise GLM53SingleExpertProjectionError("stage must be 0..23")
    if type(slot) is not int or slot < 0:
        raise GLM53SingleExpertProjectionError("slot must be nonnegative")
    if type(block_rows) is not int or not 1 <= block_rows <= 128:
        raise GLM53SingleExpertProjectionError("block_rows must be 1..128")
    raw = Path(raw_dir)
    calibration = Path(calibration_dir)
    output = Path(output_dir)
    manifest_path = raw / "raw_expert_manifest.json"
    bundle = _json(manifest_path)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("not_an_iq_checkpoint") is not True
        or bundle.get("source_repo") != REPO
        or bundle.get("source_revision") != REVISION
        or bundle.get("status") not in {"partial_smoke_only", "complete_raw_donor_bundle"}
    ):
        raise GLM53SingleExpertProjectionError("raw donor bundle provenance mismatch")
    capacity = bundle.get("target_experts")
    if type(capacity) is not int or capacity not in (64, 128, 256):
        raise GLM53SingleExpertProjectionError("unsupported recipient expert count")
    target_config = canonical_glm53_config(routed_experts=capacity)
    if bundle.get("target_config_fingerprint") != target_config.fingerprint:
        raise GLM53SingleExpertProjectionError("raw bundle recipient config mismatch")
    stages = [v for v in bundle.get("layers", []) if v.get("stage") == stage]
    if len(stages) != 1:
        raise GLM53SingleExpertProjectionError("raw bundle does not contain requested stage")
    record = stages[0]
    ids = record.get("exported_donor_ids", [])
    if slot >= len(ids) or type(ids[slot]) is not int:
        raise GLM53SingleExpertProjectionError("selected expert slot was not exported")
    donor = ids[slot]
    source_layer = record["source_layer"]
    if type(source_layer) is not int or not 0 <= source_layer < 78:
        raise GLM53SingleExpertProjectionError("invalid donor layer")
    p, physical, calibration_hashes = _calibration_map(
        calibration, stage, source_layer, capacity
    )
    hidden, latent = p.shape
    intermediate = target_config.stable_moe.expert_intermediate_size
    input_roots = (raw.resolve(), calibration.resolve())
    destination = output.resolve()
    if any(destination == base or base in destination.parents for base in input_roots):
        raise GLM53SingleExpertProjectionError(
            "projection output cannot overlap donor or calibration inputs"
        )
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise GLM53SingleExpertProjectionError("output must be a new empty directory")
    # Validate all inputs BEFORE creating any output artifacts.
    prefix = f"model.layers.{source_layer}.mlp.experts.{donor}"
    inputs = {}
    for kind in ("gate", "up", "down"):
        shape = (intermediate, hidden) if kind != "down" else (hidden, intermediate)
        key = f"{prefix}.{kind}_proj.weight"
        inputs[kind] = _read_donor(raw, bundle, key, shape)

    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text(
        "Single-expert projection incomplete; not an IQ checkpoint.\n",
        encoding="utf-8",
    )
    state = {}
    diagnostics = {}
    for kind in ("gate", "up", "down"):
        if kind == "down":
            tensor, measures = _project_down(inputs[kind], p, block_rows=block_rows)
        else:
            tensor, measures = _project_gate_or_up(
                inputs[kind], p, block_rows=block_rows
            )
        key = f"layers.{physical}.moe.routed_experts.{slot}.{kind}_proj.weight"
        state[key] = tensor.contiguous()
        diagnostics[kind] = measures
        del tensor
    shard = f"projected-stage-{stage:02d}-slot-{slot:03d}.safetensors"
    save_file(state, str(output / shard))
    reread = load_file(str(output / shard), device="cpu")
    if set(reread) != set(state) or any(
        reread[k].dtype != state[k].dtype or _sha_tensor(reread[k]) != _sha_tensor(state[k])
        for k in state
    ):
        raise GLM53SingleExpertProjectionError("projected shard roundtrip failure")
    report = {
        "schema_version": 1,
        "artifact_type": "glm53_selected_expert_latent_projection",
        "not_an_iq_checkpoint": True,
        "stage": stage,
        "donor_layer": source_layer,
        "donor_expert_id": donor,
        "recipient_moe_physical_layer": physical,
        "recipient_slot": slot,
        "donor_hidden_size": hidden,
        "target_latent_size": latent,
        "target_intermediate_size": intermediate,
        "target_config_fingerprint": target_config.fingerprint,
        "raw_bundle_manifest_sha256": _sha_file(manifest_path),
        "calibration_sha256": calibration_hashes,
        "shard": shard,
        "shard_sha256": _sha_file(output / shard),
        "output_tensor_sha256": {key: _sha_tensor(value) for key, value in state.items()},
        "projection": diagnostics,
        "note": "Weight-subspace energy only; no functional behavior or attention parity validated.",
    }
    (output / "projection_manifest.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output / "INCOMPLETE").unlink()
    return report
