"""Matrix-free, CPU-bounded exploratory correction to a shared GLM WARM latent basis.

Train on selected routed experts; report independent validation expert(s).
No full 6144x6144 covariance or SVD, no network, no checkpoint mutation.
This is a weight-energy-only optimization, NOT activation or task fidelity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from safetensors.numpy import save_file

from .complete_transplant import canonical_glm53_config
from .glm53_single_expert_projection import (
    _calibration_map,
    _json,
    _read_donor,
    _sha_file,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class ExpertBasisProbeError(RuntimeError):
    pass


KINDS = ("gate", "up", "down")
ANGLES = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)


def gram_action(weight: np.ndarray, vectors: np.ndarray, kind: str) -> np.ndarray:
    """Apply W.T W (gate/up) or W W.T (down) without building a Gram matrix."""
    if kind in ("gate", "up"):
        return weight.T @ (weight @ vectors)
    if kind == "down":
        return weight @ (weight.T @ vectors)
    raise ExpertBasisProbeError(f"unexpected expert operator {kind}")


def projection_components(
    weight: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    kind: str,
) -> tuple[float, float, float, float]:
    """Operator energy on paired orthonormal old/novel columns."""
    if kind in ("gate", "up"):
        a, b = weight @ old, weight @ new
    elif kind == "down":
        a, b = weight.T @ old, weight.T @ new
    else:
        raise ExpertBasisProbeError("unsupported MoE operator")
    src = float(np.einsum("ij,ij->", weight, weight, dtype=np.float64))
    aa = float(np.einsum("ij,ij->", a, a, dtype=np.float64))
    bb = float(np.einsum("ij,ij->", b, b, dtype=np.float64))
    ab = float(np.einsum("ij,ij->", a, b, dtype=np.float64))
    if not all(np.isfinite(t) for t in (src, aa, bb, ab)) or src <= 0:
        raise ExpertBasisProbeError("operator has invalid source norm or projection")
    return src, aa, bb, ab


def rotate_energy(
    original_retained: float,
    components: tuple[float, float, float, float],
    angle: float,
) -> float:
    """Exact change in squared projection energy for one rotated column block."""
    source_sq, old_sq, new_sq, cross = components
    c, s = float(np.cos(angle)), float(np.sin(angle))
    delta = (c*c - 1.0)*old_sq + s*s*new_sq + 2.0*c*s*cross
    return float(original_retained + delta / source_sq)


def novel_orthogonal_directions(
    basis: np.ndarray,
    raw_gradient: np.ndarray,
) -> np.ndarray:
    """Orthonormal tangent directions with a positive per-column ascent sign."""
    p = np.asarray(basis, dtype=np.float64)
    g = np.asarray(raw_gradient, dtype=np.float64)
    if g.shape[0] != p.shape[0] or g.shape[1] > p.shape[1]:
        raise ExpertBasisProbeError("incorrect gradient dimensions")
    residual = g - p @ (p.T @ g)
    residual = residual - p @ (p.T @ residual)
    if np.linalg.norm(residual, axis=0).min() <= 1e-12:
        raise ExpertBasisProbeError("cannot identify independent tangent directions")
    r, _ = np.linalg.qr(residual, mode="reduced")
    if np.linalg.norm(r.T @ r - np.eye(r.shape[1]), ord=np.inf) > 1e-7:
        raise ExpertBasisProbeError("candidate tangent vectors are not orthogonal")
    if np.max(np.abs(p.T @ r)) > 1e-7:
        raise ExpertBasisProbeError("candidate tangent vectors overlap the old basis")
    signs = np.sign(np.sum(residual * r, axis=0))
    signs[signs == 0] = 1
    return r * signs


def rotated_basis(
    original: np.ndarray, columns: np.ndarray, tangent: np.ndarray, angle: float,
) -> np.ndarray:
    p = np.asarray(original, dtype=np.float64).copy()
    if tangent.shape != (p.shape[0], len(columns)):
        raise ExpertBasisProbeError("tangent geometry differs from column selection")
    c, s = np.cos(angle), np.sin(angle)
    p[:, columns] = c * p[:, columns] + s * tangent
    # Verify before any candidate map is written.
    if np.max(np.abs(p.T @ p - np.eye(p.shape[1]))) > 2e-7:
        raise ExpertBasisProbeError("rotated basis is not column-orthonormal")
    return p


def _donor_operator(
    raw: Path,
    manifest: Mapping[str, Any],
    source_layer: int,
    donor_id: int,
    kind: str,
    hidden: int,
    intermediate: int,
) -> np.ndarray:
    shape = (intermediate, hidden) if kind != "down" else (hidden, intermediate)
    key = f"model.layers.{source_layer}.mlp.experts.{donor_id}.{kind}_proj.weight"
    tensor = _read_donor(raw, dict(manifest), key, shape)
    return tensor.float().numpy()


def _read_baseline(
    baseline: Path, raw_manifest_digest: str,
    donor_ids: list[int], stage: int, calibration_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    batch = _json(baseline / "stage_projection_manifest.json")
    entries = batch.get("entries")
    if (
        batch.get("schema_version") != 1
        or batch.get("stage") != stage
        or batch.get("not_an_iq_checkpoint") is not True
        or batch.get("raw_bundle_manifest_sha256") != raw_manifest_digest
        or batch.get("calibration_file_sha256") != calibration_hashes
        or not isinstance(entries, list)
        or len(entries) < len(donor_ids)
    ):
        raise ExpertBasisProbeError("baseline stage projection metadata mismatch")
    metrics = []
    for slot, donor_id in enumerate(donor_ids):
        entry = entries[slot]
        if entry.get("slot") != slot or entry.get("donor_expert_id") != donor_id:
            raise ExpertBasisProbeError("baseline expert order/source mismatch")
        location = entry.get("relative_path")
        if location != f"slot-{slot:03d}":
            raise ExpertBasisProbeError("baseline projection path mismatch")
        path = baseline / location / "projection_manifest.json"
        if _sha_file(path) != entry.get("projection_manifest_sha256"):
            raise ExpertBasisProbeError("baseline projection manifest digest mismatch")
        report = _json(path)
        if (report.get("raw_bundle_manifest_sha256") != raw_manifest_digest
                or report.get("calibration_sha256") != calibration_hashes
                or report.get("donor_expert_id") != donor_id
                or report.get("stage") != stage):
            raise ExpertBasisProbeError("baseline donor or calibration mismatch")
        projector = report.get("projection")
        if not isinstance(projector, dict):
            raise ExpertBasisProbeError("baseline energy metrics unavailable")
        ratios = {}
        for kind in KINDS:
            values = projector.get(kind)
            if not isinstance(values, dict):
                raise ExpertBasisProbeError(f"missing baseline operator: {kind}")
            frac = values.get("subspace_retained_energy")
            if type(frac) not in (float, int) or not 0 <= frac <= 1.0000001:
                raise ExpertBasisProbeError("invalid original energy ratio")
            ratios[kind] = float(frac)
        metrics.append(ratios)
    return metrics


def run_expert_informed_probe(
    *,
    raw_dir: str | Path,
    baseline_dir: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    stage: int = 1,
    candidate_columns: int = 16,
    validation_slot: int = 3,
    seed: int = 77,
) -> dict[str, Any]:
    """Optimize train-only normalized weight energy and report unseen expert."""
    if type(stage) is not int or not 1 <= stage <= 23:
        raise ExpertBasisProbeError("stage must be one of 1..23")
    if type(candidate_columns) is not int or not 1 <= candidate_columns <= 32:
        raise ExpertBasisProbeError("candidate_columns must be 1..32")
    if type(validation_slot) is not int or validation_slot < 0:
        raise ExpertBasisProbeError("validation_slot must be a nonnegative integer")
    raw, baseline, calibration, output = map(
        Path, (raw_dir, baseline_dir, calibration_dir, output_dir)
    )
    destination = output.resolve()
    for root in (raw, baseline, calibration):
        source = root.resolve()
        if destination == source or source in destination.parents:
            raise ExpertBasisProbeError("candidate output cannot be inside input trees")
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise ExpertBasisProbeError("candidate output must be new/empty")
    raw_path = raw / "raw_expert_manifest.json"
    manifest = _json(raw_path)
    if (
        manifest.get("source_repo") != REPO
        or manifest.get("source_revision") != REVISION
        or manifest.get("not_an_iq_checkpoint") is not True
    ):
        raise ExpertBasisProbeError("invalid donor checkpoint provenance")
    capacity = manifest.get("target_experts")
    if type(capacity) is not int or capacity not in (64, 128, 256):
        raise ExpertBasisProbeError("unknown IQ configuration capacity")
    config = canonical_glm53_config(routed_experts=capacity)
    if manifest.get("target_config_fingerprint") != config.fingerprint:
        raise ExpertBasisProbeError("donor/target fingerprint mismatch")
    rows = [r for r in manifest.get("layers", []) if r.get("stage") == stage]
    if len(rows) != 1:
        raise ExpertBasisProbeError("expected exactly one exported routed stage")
    row = rows[0]
    source_layer = row.get("source_layer")
    ids = row.get("exported_donor_ids")
    if (type(source_layer) is not int or not 3 <= source_layer < 78
            or not isinstance(ids, list)
            or len(ids) < 3
            or len(ids) > capacity
            or len(set(ids)) != len(ids)
            or any(type(donor_id) is not int or not 0 <= donor_id < 256 for donor_id in ids)):
        raise ExpertBasisProbeError("need at least three valid selected donor experts")
    if validation_slot >= len(ids):
        raise ExpertBasisProbeError("validation_slot is outside available experts")
    p, physical, verified_hashes = _calibration_map(
        calibration, stage, source_layer, capacity
    )
    hidden, latent = p.shape
    if not 1 <= candidate_columns < latent:
        raise ExpertBasisProbeError("too many basis columns requested")
    baseline_metrics = _read_baseline(
        baseline, _sha_file(raw_path), ids, stage, verified_hashes
    )
    intermediate = config.stable_moe.expert_intermediate_size
    if intermediate != 2048:
        raise ExpertBasisProbeError("unexpected IQ intermediate width")
    # One fixed, reproducible column subset; only 16/32 columns are changed.
    # This probes local improvements rather than claiming optimal rank-2048 SVD.
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(latent, candidate_columns, replace=False))
    q0 = np.asarray(p[:, indices], dtype=np.float32)
    total_g = np.zeros((hidden, candidate_columns), dtype=np.float64)
    train = [i for i in range(len(ids)) if i != validation_slot]
    # Gradient of mean projected Frobenius energy, normalized per operator.
    for slot in train:
        for kind in KINDS:
            weight = _donor_operator(
                raw, manifest, source_layer, ids[slot], kind, hidden, intermediate
            )
            sq = float(np.einsum("ij,ij->", weight, weight, dtype=np.float64))
            if sq <= 0 or not np.isfinite(sq):
                raise ExpertBasisProbeError("invalid donor expert norm")
            total_g += (
                gram_action(weight, q0, kind).astype(np.float64) / sq
            ) / (len(train) * len(KINDS))
            del weight
    tangent = novel_orthogonal_directions(p, total_g)
    q_new = np.asarray(tangent, dtype=np.float32)

    # Evaluate each operator ONCE: all angle trials use analytic cross terms.
    parts: list[dict[str, tuple[float, float, float, float]]] = []
    for slot, donor_id in enumerate(ids):
        one = {}
        for kind in KINDS:
            weight = _donor_operator(
                raw, manifest, source_layer, donor_id, kind, hidden, intermediate
            )
            one[kind] = projection_components(weight, q0, q_new, kind)
            del weight
        parts.append(one)
    options = []
    for angle in ANGLES:
        results = []
        for slot, donor_id in enumerate(ids):
            after = {
                kind: rotate_energy(baseline_metrics[slot][kind], parts[slot][kind], angle)
                for kind in KINDS
            }
            if any(not np.isfinite(v) or v < -0.001 or v > 1.001
                   for v in after.values()):
                raise ExpertBasisProbeError("invalid candidate retained energy")
            improvements = {
                kind: after[kind] - baseline_metrics[slot][kind] for kind in KINDS
            }
            results.append({
                "slot": slot, "donor_id": donor_id,
                "role": "validation" if slot == validation_slot else "train",
                "before": baseline_metrics[slot],
                "after": after,
                "delta": improvements,
                "mean_delta": sum(improvements.values()) / len(KINDS),
            })
        mean_train_delta = float(np.mean([
            x["mean_delta"] for x in results if x["role"] == "train"
        ]))
        options.append({
            "angle_radians": angle,
            "train_mean_energy_delta": mean_train_delta,
            "experts": results,
        })
    choice = max(options, key=lambda item: item["train_mean_energy_delta"])
    selected_angle = choice["angle_radians"]
    # No automatic replacement of the original GLM calibration.
    if choice["train_mean_energy_delta"] <= 0:
        raise ExpertBasisProbeError("no positive expert-informed direction found")
    refined = rotated_basis(p, indices, tangent, selected_angle)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "candidate-stage-latent.safetensors"
    save_file({"matrix": refined}, str(state_path))
    report = {
        "schema_version": 1,
        "artifact_type": "glm53_expert_informed_basis_energy_probe",
        "not_an_iq_checkpoint": True,
        "not_canonical_calibration": True,
        "weight_energy_only": True,
        "not_real_activation_validation": True,
        "stage": stage,
        "source_layer": source_layer,
        "recipient_moe_physical_layer": physical,
        "train_slots": train,
        "validation_slot": validation_slot,
        "source_expert_ids": ids,
        "basis_dimensions": [hidden, latent],
        "modified_basis_columns": indices.tolist(),
        "selected_angle_radians": selected_angle,
        "candidate_matrix_sha256": _sha_file(state_path),
        "calibration_sha256": verified_hashes,
        "raw_manifest_sha256": _sha_file(raw_path),
        "baseline_stage_manifest_sha256": _sha_file(
            baseline / "stage_projection_manifest.json"
        ),
        "candidate_results": choice,
        "angle_trials": [
            {"angle_radians": o["angle_radians"],
             "train_mean_energy_delta": o["train_mean_energy_delta"]}
            for o in options
        ],
        "warning": (
            "Exploratory improvement in weight energy does NOT imply better GLM "
            "hidden-state fidelity, routing, or task quality. Holdout is reported "
            "but not used for angle selection. Never replace canonical maps yet."
        ),
    }
    (output / "basis_probe_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
