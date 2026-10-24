"""Evaluate GLM expert projection fidelity on subspace, off-subspace, or captured activations.

This is an isolated *expert* comparison, not an end-to-end IQ accuracy test.
Synthetic inputs only test numerical projection consistency and sensitivity to
discarded source dimensions. Real GLM hidden states must be supplied explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .complete_transplant import canonical_glm53_config
from .glm53_single_expert_projection import (
    GLM53SingleExpertProjectionError,
    _calibration_map,
    _json,
    _read_donor,
    _sha_file,
    _sha_tensor,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class GLM53FidelityError(ValueError):
    pass


def expert_function(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """One SwiGLU FFN, without routing/norm/residual or full-model context."""
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def comparison(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    if reference.shape != actual.shape or reference.ndim != 2 or not reference.numel():
        raise GLM53FidelityError("expert output geometry mismatch")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise GLM53FidelityError("nonfinite evaluated expert outputs")
    difference = actual - reference
    rmse = difference.float().square().mean().sqrt()
    ref_rms = reference.float().square().mean().sqrt()
    ref_norm = torch.linalg.vector_norm(reference.float(), dim=-1)
    actual_norm = torch.linalg.vector_norm(actual.float(), dim=-1)
    denom = (ref_norm * actual_norm).clamp_min(1e-30)
    cos = ((reference.float() * actual.float()).sum(dim=-1) / denom)
    return {
        "rmse": float(rmse),
        "reference_rms": float(ref_rms),
        "relative_rmse": float(rmse / ref_rms.clamp_min(1e-12)),
        "mean_token_cosine": float(cos.mean()),
        "max_absolute_error": float(difference.abs().max()),
    }


@torch.no_grad()
def evaluate_expert(
    donor: dict[str, torch.Tensor],
    recipient: dict[str, torch.Tensor],
    basis: torch.Tensor,
    source_inputs: torch.Tensor,
) -> dict[str, Any]:
    """Compare the same donor inputs, compressed inputs, and IQ projected expert."""
    if basis.ndim != 2 or source_inputs.ndim != 2:
        raise GLM53FidelityError("expected rank-2 latent basis and token inputs")
    source_width, target_width = basis.shape
    if source_inputs.shape[1] != source_width or source_inputs.shape[0] < 1:
        raise GLM53FidelityError("source activations have incompatible dimensions")
    if not bool(torch.isfinite(source_inputs).all()):
        raise GLM53FidelityError("source activations contain nonfinite values")
    intermediate = donor["gate"].shape[0]
    expected_shapes = {
        "gate": (intermediate, source_width),
        "up": (intermediate, source_width),
        "down": (source_width, intermediate),
    }
    recipient_shapes = {
        "gate": (intermediate, target_width),
        "up": (intermediate, target_width),
        "down": (target_width, intermediate),
    }
    if any(tuple(donor[k].shape) != expected_shapes[k] or
           tuple(recipient[k].shape) != recipient_shapes[k]
           for k in ("gate", "up", "down")):
        raise GLM53FidelityError("expert source/recipient tensor shapes disagree")

    p = basis.float()
    x = source_inputs.float()
    donor_weights = {k: v.float() for k, v in donor.items()}
    recipient_weights = {k: v.float() for k, v in recipient.items()}
    xt = x @ p
    x_reconstruct = xt @ p.T
    removed = x - x_reconstruct
    input_energy = x.square().sum().clamp_min(1e-30)
    excluded_ratio = float(removed.square().sum() / input_energy)
    direct = expert_function(x, **donor_weights) @ p
    compressed = expert_function(x_reconstruct, **donor_weights) @ p
    target = expert_function(xt, **recipient_weights)
    return {
        "samples": int(x.shape[0]),
        "source_width": source_width,
        "target_width": target_width,
        "input_discarded_energy_fraction": excluded_ratio,
        "projector_numerical_error": comparison(compressed, target),
        "discarded_input_effect": comparison(direct, compressed),
        "total_error_against_original": comparison(direct, target),
        "limitation": (
            "Single SwiGLU expert only; excludes router/top-k, normalization, "
            "neighbor layers, attention, and downstream task performance."
        ),
    }


def load_capture(path: str | Path, *, width: int, max_samples: int) -> torch.Tensor:
    path = Path(path)
    if path.suffix != ".npy":
        raise GLM53FidelityError(
            "captured GLM hidden states must be a 2D numeric .npy file"
        )
    activations = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        activations.ndim != 2
        or activations.shape[1] != width
        or not np.issubdtype(activations.dtype, np.floating)
        or activations.shape[0] < 1
    ):
        raise GLM53FidelityError(
            "captured activations must have shape [tokens, donor_hidden_width]"
        )
    count = min(max_samples, activations.shape[0])
    tensor = torch.from_numpy(np.array(activations[:count], dtype=np.float32, copy=True))
    if not bool(torch.isfinite(tensor).all()):
        raise GLM53FidelityError("captured activations contain nonfinite values")
    return tensor


def _load_verified_inputs(
    *,
    raw_dir: Path,
    projected_dir: Path,
    calibration_dir: Path,
    stage: int,
    slot: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    raw_path = raw_dir / "raw_expert_manifest.json"
    projection_path = projected_dir / "projection_manifest.json"
    raw = _json(raw_path)
    report = _json(projection_path)
    if (
        raw.get("source_repo") != REPO
        or raw.get("source_revision") != REVISION
        or raw.get("not_an_iq_checkpoint") is not True
        or report.get("not_an_iq_checkpoint") is not True
        or report.get("artifact_type") != "glm53_selected_expert_latent_projection"
        or report.get("stage") != stage
        or report.get("recipient_slot") != slot
        or report.get("raw_bundle_manifest_sha256") != _sha_file(raw_path)
    ):
        raise GLM53FidelityError("donor/projection bundle provenance mismatch")
    capacity = raw.get("target_experts")
    if type(capacity) is not int or capacity not in (64, 128, 256):
        raise GLM53FidelityError("unrecognized source expert capacity")
    config = canonical_glm53_config(routed_experts=capacity)
    if (raw.get("target_config_fingerprint") != config.fingerprint or
            report.get("target_config_fingerprint") != config.fingerprint):
        raise GLM53FidelityError("recipient configuration fingerprint mismatch")
    records = [row for row in raw.get("layers", []) if row["stage"] == stage]
    if len(records) != 1:
        raise GLM53FidelityError("source stage missing or duplicated")
    row = records[0]
    if slot >= len(row.get("exported_donor_ids", [])):
        raise GLM53FidelityError("donor slot not present in exported bundle")
    donor_id = row["exported_donor_ids"][slot]
    source_layer = row["source_layer"]
    if (report["donor_expert_id"], report["donor_layer"]) != (donor_id, source_layer):
        raise GLM53FidelityError("selected source expert does not match projection")
    p, physical, calib_hashes = _calibration_map(
        calibration_dir, stage, source_layer, capacity
    )
    if (physical != report["recipient_moe_physical_layer"] or
            calib_hashes != report["calibration_sha256"]):
        raise GLM53FidelityError("projection calibration provenance mismatch")
    source_width, latent_width = p.shape
    intermediate = config.stable_moe.expert_intermediate_size
    donor_prefix = f"model.layers.{source_layer}.mlp.experts.{donor_id}."
    target_prefix = f"layers.{physical}.moe.routed_experts.{slot}."
    donor = {}
    for kind in ("gate", "up", "down"):
        shape = ((intermediate, source_width) if kind != "down"
                 else (source_width, intermediate))
        donor[kind] = _read_donor(
            raw_dir, raw, f"{donor_prefix}{kind}_proj.weight", shape
        ).float()
    shard_name = report["shard"]
    if (Path(shard_name).name != shard_name or
            _sha_file(projected_dir / shard_name) != report["shard_sha256"]):
        raise GLM53FidelityError("projected shard integrity mismatch")
    tensors = load_file(str(projected_dir / shard_name), device="cpu")
    expected = {f"{target_prefix}{k}_proj.weight" for k in ("gate", "up", "down")}
    if set(tensors) != expected:
        raise GLM53FidelityError("projected tensor key inventory mismatch")
    recipient = {}
    for kind in ("gate", "up", "down"):
        key = f"{target_prefix}{kind}_proj.weight"
        t = tensors[key]
        shape = ((intermediate, latent_width) if kind != "down"
                 else (latent_width, intermediate))
        if (t.dtype != torch.bfloat16 or tuple(t.shape) != shape or
                _sha_tensor(t) != report["output_tensor_sha256"][key]):
            raise GLM53FidelityError(f"corrupt or unexpected projection: {key}")
        recipient[kind] = t.float()
    info = {
        "donor_layer": source_layer, "donor_expert_id": donor_id,
        "stage": stage, "slot": slot, "physical_layer": physical,
        "raw_manifest_sha256": _sha_file(raw_path),
        "projection_manifest_sha256": _sha_file(projection_path),
        "calibration_hashes": calib_hashes,
    }
    return donor, recipient, torch.tensor(p, dtype=torch.float32), info


@torch.no_grad()
def audit_expert(
    *,
    raw_dir: str | Path,
    projected_dir: str | Path,
    calibration_dir: str | Path,
    stage: int = 1,
    slot: int = 0,
    samples: int = 32,
    seed: int = 77,
    captured_activations: str | Path | None = None,
) -> dict[str, Any]:
    if type(samples) is not int or samples < 1 or samples > 512:
        raise GLM53FidelityError("samples must be in [1, 512]")
    if type(stage) is not int or not 0 <= stage < 24:
        raise GLM53FidelityError("stage must be 0..23")
    if type(slot) is not int or slot < 0:
        raise GLM53FidelityError("slot must be a nonnegative integer")
    donor, recipient, p, info = _load_verified_inputs(
        raw_dir=Path(raw_dir), projected_dir=Path(projected_dir),
        calibration_dir=Path(calibration_dir), stage=stage, slot=slot,
    )
    if captured_activations is not None:
        path = Path(captured_activations)
        x = load_capture(path, width=p.shape[0], max_samples=samples)
        analyses = {"captured_source_hidden_states": evaluate_expert(
            donor, recipient, p, x
        )}
        origin = "user_supplied_capture"
        capture_sha = _sha_file(path)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn(samples, p.shape[1], generator=generator)
        projected_inputs = latent @ p.T
        noise = torch.randn(samples, p.shape[0], generator=generator)
        orthogonal = noise - (noise @ p) @ p.T
        analyses = {}
        for alpha in (0.0, 0.5, 1.0):
            analyses[f"off_subspace_scale_{alpha:g}"] = evaluate_expert(
                donor, recipient, p, projected_inputs + alpha * orthogonal
            )
        origin = "synthetic_stress_test_not_real_activations"
        capture_sha = None
    return {
        "schema_version": 1,
        "artifact_type": "glm53_single_expert_fidelity_audit",
        "not_an_iq_checkpoint": True,
        "input_origin": origin,
        "captured_activation_file_sha256": capture_sha,
        "samples_requested_max": samples,
        "random_seed_if_synthetic": seed if captured_activations is None else None,
        "source": info,
        "analyses": analyses,
        "interpretation": (
            "Projector numerical error tests algebra/rounding; discarded-input "
            "effect tests representational loss. Neither measures full model quality."
        ),
    }
