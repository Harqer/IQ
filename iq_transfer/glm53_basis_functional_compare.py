"""Compare GLM donor SwiGLU fidelity for the original and experimental WARM bases.

Evaluation is in the same 6144-wide donor output space. The candidate is
selected from weight energy, not from these functional measurements.
No calibration, projection, or checkpoint file is modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file as load_numpy
from safetensors.torch import load_file as load_torch

from .complete_transplant import canonical_glm53_config
from .glm53_expert_fidelity import comparison, expert_function, load_capture
from .glm53_single_expert_projection import (
    _calibration_map,
    _json,
    _read_donor,
    _sha_file,
    _sha_tensor,
)
from .glm53_expert_informed_basis import _read_baseline
from .glm53_routed_fidelity import read_verified_router_pairs
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class ExpertBasisFunctionalError(ValueError):
    pass


def check_candidate(
    candidate_dir: Path, baseline_dir: Path, raw_dir: Path,
    original: np.ndarray, stage: int, source_layer: int,
    calibration_hashes: dict[str, str], ids: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    report = _json(candidate_dir / "basis_probe_report.json")
    expected = {
        "schema_version": 1,
        "artifact_type": "glm53_expert_informed_basis_energy_probe",
        "not_an_iq_checkpoint": True,
        "not_canonical_calibration": True,
        "weight_energy_only": True,
        "stage": stage,
        "source_layer": source_layer,
        "source_expert_ids": ids,
        "calibration_sha256": calibration_hashes,
        "raw_manifest_sha256": _sha_file(raw_dir / "raw_expert_manifest.json"),
        "baseline_stage_manifest_sha256": _sha_file(
            baseline_dir / "stage_projection_manifest.json"
        ),
        "basis_dimensions": list(original.shape),
    }
    for k, v in expected.items():
        if report.get(k) != v:
            raise ExpertBasisFunctionalError(f"candidate provenance mismatch: {k}")
    if not isinstance(report.get("train_slots"), list):
        raise ExpertBasisFunctionalError("candidate training slots absent")
    train_slots = report["train_slots"]
    holdout = report.get("validation_slot")
    if (type(holdout) is not int or holdout not in range(len(ids))
            or sorted(train_slots) != [i for i in range(len(ids)) if i != holdout]):
        raise ExpertBasisFunctionalError("invalid candidate training/holdout partition")
    name = candidate_dir / "candidate-stage-latent.safetensors"
    if _sha_file(name) != report.get("candidate_matrix_sha256"):
        raise ExpertBasisFunctionalError("candidate basis SHA256 mismatch")
    contents = load_numpy(str(name))
    if set(contents) != {"matrix"}:
        raise ExpertBasisFunctionalError("candidate basis safetensors must contain only matrix")
    new = contents["matrix"]
    if new.shape != original.shape or not np.isfinite(new).all():
        raise ExpertBasisFunctionalError("candidate basis geometry or finite check failed")
    idx = report.get("modified_basis_columns")
    if (not isinstance(idx, list) or not idx or len(set(idx)) != len(idx)
            or any(type(k) is not int or not 0 <= k < original.shape[1] for k in idx)):
        raise ExpertBasisFunctionalError("invalid candidate modified-column inventory")
    keep = np.ones(original.shape[1], dtype=bool)
    keep[idx] = False
    if not np.array_equal(new[:, keep], original[:, keep]):
        raise ExpertBasisFunctionalError("candidate changed unlisted WARM columns")
    # An intact, hash-verified candidate must also be semi-orthogonal.
    for first in range(0, len(idx), 16):
        block = idx[first:first + 16]
        gram = new.T @ new[:, block]
        for offset, k in enumerate(block):
            gram[k, offset] -= 1.0
        if np.max(np.abs(gram)) > 2e-7:
            raise ExpertBasisFunctionalError("candidate map is not semi-orthogonal")
    return new, report


def _load_baseline_expert(
    baseline_dir: Path, slot: int, source_id: int, physical: int,
    raw_digest: str, calibration_hashes: dict[str, str], stage: int,
    latent: int, intermediate: int,
) -> dict[str, torch.Tensor]:
    root = baseline_dir / f"slot-{slot:03d}"
    report = _json(root / "projection_manifest.json")
    if (report.get("stage") != stage or report.get("donor_expert_id") != source_id
            or report.get("recipient_slot") != slot
            or report.get("recipient_moe_physical_layer") != physical
            or report.get("raw_bundle_manifest_sha256") != raw_digest
            or report.get("calibration_sha256") != calibration_hashes
            or report.get("not_an_iq_checkpoint") is not True):
        raise ExpertBasisFunctionalError("baseline expert provenance mismatch")
    name = report.get("shard")
    if not isinstance(name, str) or Path(name).name != name:
        raise ExpertBasisFunctionalError("unsafe baseline shard filename")
    if _sha_file(root / name) != report.get("shard_sha256"):
        raise ExpertBasisFunctionalError("baseline shard file checksum mismatch")
    tensors = load_torch(str(root / name), device="cpu")
    prefix = f"layers.{physical}.moe.routed_experts.{slot}."
    expected = {f"{prefix}{k}_proj.weight" for k in ("gate", "up", "down")}
    if set(tensors) != expected:
        raise ExpertBasisFunctionalError("baseline expert tensor names mismatch")
    found = {}
    for kind in ("gate", "up", "down"):
        key = f"{prefix}{kind}_proj.weight"
        value = tensors[key]
        shape = (intermediate, latent) if kind != "down" else (latent, intermediate)
        if (tuple(value.shape) != shape or value.dtype != torch.bfloat16
                or _sha_tensor(value) != report["output_tensor_sha256"][key]):
            raise ExpertBasisFunctionalError(f"baseline projected tensor mismatch: {key}")
        found[kind] = value.float()
    return found


def _candidate_expert(
    donor: dict[str, torch.Tensor],
    candidate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # BF16 round-trip reflects IQ's actual expert weight storage format.
    return {
        "gate": (donor["gate"] @ candidate).to(torch.bfloat16).float(),
        "up": (donor["up"] @ candidate).to(torch.bfloat16).float(),
        "down": (candidate.T @ donor["down"]).to(torch.bfloat16).float(),
    }


@torch.no_grad()
def compare_basis_functions(
    *,
    raw_dir: str | Path,
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    calibration_dir: str | Path,
    stage: int = 1,
    samples: int = 16,
    seed: int = 77,
    captured_activations: str | Path | None = None,
    router_topk: str | Path | None = None,
) -> dict[str, Any]:
    if type(stage) is not int or not 1 <= stage <= 23:
        raise ExpertBasisFunctionalError("stage must be 1..23")
    max_count = 4096 if captured_activations is not None else 128
    if type(samples) is not int or not 1 <= samples <= max_count:
        raise ExpertBasisFunctionalError(f"samples must be in [1,{max_count}]")
    if router_topk is not None and (captured_activations is None or stage != 1):
        raise ExpertBasisFunctionalError(
            "router-topk requires stage-1 real captured activations"
        )
    if type(seed) is not int or seed < 0:
        raise ExpertBasisFunctionalError("seed must be nonnegative")
    raw, baseline, candidate, cal = map(
        Path, (raw_dir, baseline_dir, candidate_dir, calibration_dir)
    )
    manifest_path = raw / "raw_expert_manifest.json"
    source = _json(manifest_path)
    capacity = source.get("target_experts")
    if (source.get("source_repo") != REPO
            or source.get("source_revision") != REVISION
            or source.get("not_an_iq_checkpoint") is not True
            or type(capacity) is not int
            or capacity not in (64, 128, 256)):
        raise ExpertBasisFunctionalError("invalid raw donor provenance or capacity")
    config = canonical_glm53_config(routed_experts=capacity)
    if source.get("target_config_fingerprint") != config.fingerprint:
        raise ExpertBasisFunctionalError("raw donor config fingerprint mismatch")
    rows = [row for row in source.get("layers", []) if row.get("stage") == stage]
    if len(rows) != 1:
        raise ExpertBasisFunctionalError("no unique donor layer for stage")
    layer = rows[0]["source_layer"]
    ids = rows[0]["exported_donor_ids"]
    if not isinstance(ids, list) or len(ids) < 3:
        raise ExpertBasisFunctionalError("expected three or more exported routed experts")
    original, physical, verified_hashes = _calibration_map(cal, stage, layer, capacity)
    _read_baseline(baseline, _sha_file(manifest_path), ids, stage, verified_hashes)
    revised, probe = check_candidate(
        candidate, baseline, raw, original, stage, layer, verified_hashes, ids
    )
    p0, p1 = (
        torch.from_numpy(np.asarray(basis, dtype=np.float32).copy())
        for basis in (original, revised)
    )
    width, latent = p0.shape
    intermediate = config.stable_moe.expert_intermediate_size

    if captured_activations is not None:
        activations = load_capture(
            captured_activations, width=width, max_samples=samples
        )
        scenarios = {"captured_glm_pre_expert_hidden_states": activations}
        input_origin = "user_supplied_capture_unverified_origin"
        capture_sha = _sha_file(Path(captured_activations))
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scenarios = {
            "isotropic_full_donor_width": torch.randn(
                samples, width, generator=generator
            ),
            "original_warm_subspace": torch.randn(
                samples, latent, generator=generator
            ) @ p0.T,
            "candidate_warm_subspace": torch.randn(
                samples, latent, generator=generator
            ) @ p1.T,
        }
        input_origin = "synthetic_not_real_activations"
        capture_sha = None

    selected = None
    router_sha = None
    if router_topk is not None:
        selected, router_sha = read_verified_router_pairs(
            capture_path=captured_activations, routing_path=router_topk,
            source_layer=layer,
            samples_used=next(iter(scenarios.values())).shape[0],
        )

    results = []
    for slot, donor_id in enumerate(ids):
        if selected is not None:
            mask = (selected == donor_id).any(axis=1)
            if not bool(mask.any()):
                results.append({
                    "slot": slot, "donor_expert_id": donor_id,
                    "role": "validation" if slot == probe["validation_slot"] else "train",
                    "routed_token_count": 0,
                    "status": "no_routed_tokens_no_fidelity_evidence",
                    "scenarios": {},
                })
                continue
        else:
            mask = None
        prefix = f"model.layers.{layer}.mlp.experts.{donor_id}."
        donor = {}
        for kind in ("gate", "up", "down"):
            shape = (intermediate, width) if kind != "down" else (width, intermediate)
            donor[kind] = _read_donor(raw, source, f"{prefix}{kind}_proj.weight", shape).float()
        old_expert = _load_baseline_expert(
            baseline, slot, donor_id, physical, _sha_file(manifest_path),
            verified_hashes, stage, latent, intermediate
        )
        new_expert = _candidate_expert(donor, p1)
        scenarios_report = {}
        for label, x in scenarios.items():
            if mask is not None:
                x = x[torch.from_numpy(mask)]
            reference = expert_function(x, **donor)
            y_original = expert_function(x @ p0, **old_expert) @ p0.T
            y_candidate = expert_function(x @ p1, **new_expert) @ p1.T
            old_metrics = comparison(reference, y_original)
            new_metrics = comparison(reference, y_candidate)
            scenarios_report[label] = {
                "evaluated_tokens": int(x.shape[0]),
                "original_warm": old_metrics,
                "expert_informed_candidate": new_metrics,
                "relative_rmse_improvement": (
                    old_metrics["relative_rmse"] - new_metrics["relative_rmse"]
                ),
                "cosine_improvement": (
                    new_metrics["mean_token_cosine"] - old_metrics["mean_token_cosine"]
                ),
                "positive_rmse_improvement": (
                    new_metrics["relative_rmse"] < old_metrics["relative_rmse"]
                ),
            }
        results.append({
            "slot": slot, "donor_expert_id": donor_id,
            "role": "validation" if slot == probe["validation_slot"] else "train",
            "routed_token_count": int(mask.sum()) if mask is not None else None,
            "status": "evaluated",
            "scenarios": scenarios_report,
        })
        del donor, old_expert, new_expert
    summary = {}
    for name in scenarios:
        available_train = [
            row["scenarios"][name]["relative_rmse_improvement"]
            for row in results
            if row["role"] == "train" and name in row["scenarios"]
        ]
        heldout = results[probe["validation_slot"]]["scenarios"].get(name)
        summary[name] = {
            "train_mean_relative_rmse_change": (
                float(np.mean(available_train)) if available_train else None
            ),
            "validation_relative_rmse_change": (
                heldout["relative_rmse_improvement"] if heldout is not None else None
            ),
            "training_experts_with_routed_evidence": len(available_train),
            "heldout_has_routed_evidence": heldout is not None,
        }
    return {
        "schema_version": 1,
        "artifact_type": "glm53_expert_informed_basis_functional_comparison",
        "not_an_iq_checkpoint": True,
        "not_canonical_calibration": True,
        "input_origin": input_origin,
        "samples_per_scenario": next(iter(scenarios.values())).shape[0],
        "seed_if_synthetic": seed if captured_activations is None else None,
        "capture_sha256": capture_sha,
        "router_topk_sha256": router_sha,
        "routing_conditioned": router_topk is not None,
        "routing_coverage": (
            {str(row["donor_expert_id"]): row["routed_token_count"] for row in results}
            if selected is not None else None
        ),
        "source_layer": layer,
        "stage": stage,
        "source_expert_ids": ids,
        "validation_slot": probe["validation_slot"],
        "raw_manifest_sha256": _sha_file(manifest_path),
        "candidate_report_sha256": _sha_file(candidate / "basis_probe_report.json"),
        "baseline_stage_manifest_sha256": _sha_file(
            baseline / "stage_projection_manifest.json"
        ),
        "calibration_sha256": verified_hashes,
        "result": summary,
        "experts": results,
        "limitations": (
            "This isolates one SwiGLU FFN per expert. Both projected outputs are "
            "lifted back to the same donor output space before comparison. "
            "Synthetic test inputs do not establish actual GLM input distributions. "
            "Router-conditioned metrics only include tokens assigned to each expert; "
            "missing assignments mean missing evidence, never zero error. "
            "Real GLM states with broader coverage and end-to-end IQ evaluations "
            "are required."
        ),
    }
