"""Incremental GLM MoE compiler: persist one selected expert at a time.

The original GLM transform is algebraically:
  gate/up = P_mid.T @ W @ pinv(P_input).T
  down    = P_output.T @ W @ pinv(P_mid).T
Sparse GLM's P_mid is an identity matrix, so we can reuse the two inverse
coordinate maps and never build 64 transformed experts simultaneously.
This is a peak *expert count* improvement, not a strict process-RAM bound:
NumPy pseudoinverses, dense donor matrices and output serialization may
still exceed small-memory limits.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import SafetensorsSource
from .complete_transplant import _save_shard
from .donor import DonorError
from .glm53 import GLM53Inspector
from .glm53_calibration import GLM53StageCalibration
from .glm53_moe import select_experts_by_usage
from .transport import CoordinateMap
from iq_model import IQHybridConfig


class GLM53IncrementalMoEError(RuntimeError):
    pass


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Validate and convert a single tensor; no collection of donor experts."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise GLM53IncrementalMoEError(
            f"{label}: expected finite tensor {shape}, got {out.shape}"
        )
    return out


def _convert(arr: np.ndarray, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if not np.isfinite(arr).all():
        raise GLM53IncrementalMoEError("transformed expert contains nonfinite values")
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype)


def _inverse(map_: CoordinateMap) -> np.ndarray:
    # Keep NumPy's original float64 pinv for mathematical parity with
    # transform_glm53_moe_selected; compute it only once per map per stage.
    return np.linalg.pinv(map_.matrix).T


def _save_expert(
    *,
    source: SafetensorsSource,
    donor_prefix: str,
    output_prefix: str,
    save_filename: str,
    output_dir: Path,
    weight_map: dict[str, str],
    source_width: int,
    intermediate_width: int,
    target_output_map: np.ndarray,
    input_inverse: np.ndarray,
) -> None:
    if target_output_map.shape[0] != source_width:
        raise GLM53IncrementalMoEError("expert output-map source width mismatch")
    if input_inverse.shape[0] != source_width:
        raise GLM53IncrementalMoEError("expert input-map source width mismatch")
    state: dict[str, torch.Tensor] = {}
    for kind in ("gate", "up"):
        w = _finite(
            source.get(f"{donor_prefix}.{kind}_proj.weight"),
            (intermediate_width, source_width),
            f"{donor_prefix}.{kind}",
        )
        # P_mid=I, so P_mid.T @ W = W.
        state[f"{output_prefix}.{kind}_proj.weight"] = _convert(w @ input_inverse)
        del w
    down = _finite(
        source.get(f"{donor_prefix}.down_proj.weight"),
        (source_width, intermediate_width),
        f"{donor_prefix}.down",
    )
    state[f"{output_prefix}.down_proj.weight"] = _convert(target_output_map.T @ down)
    del down
    _save_shard(output_dir, save_filename, state, weight_map)
    del state


def _common(
    *,
    source: SafetensorsSource,
    stage: GLM53StageCalibration,
    stable: Any,
    physical: int,
    sparse: bool,
    selected: tuple[int, ...],
    source_width: int,
    input_inverse: np.ndarray,
) -> dict[str, torch.Tensor]:
    target = f"layers.{physical}.moe"
    state: dict[str, torch.Tensor] = {
        f"layers.{physical}.norm.weight": torch.ones(
            stable.hidden_size, dtype=torch.bfloat16
        ),
        f"{target}.latent_norm.weight": torch.ones(
            stable.latent_size, dtype=torch.bfloat16
        ),
    }
    pin = stage.moe_input_map.matrix
    plat = stage.latent_map.matrix
    pout = stage.residual_map.matrix
    if any(map_.shape[0] != source_width for map_ in (pin, plat, pout)):
        raise GLM53IncrementalMoEError("coordinate-map donor widths differ")
    latent_down = plat.T @ input_inverse
    latent_up = pout.T @ _inverse(stage.latent_map)
    state[f"{target}.latent_down.weight"] = _convert(latent_down)
    state[f"{target}.latent_up.weight"] = _convert(latent_up)
    if sparse:
        source_layer = stage.source_layer
        root = f"model.layers.{source_layer}.mlp"
        router = _finite(
            source.get(f"{root}.gate.weight"),
            (256, source_width),
            f"{root}.gate.weight",
        )
        bias = _finite(
            source.get(f"{root}.gate.e_score_correction_bias"),
            (256,),
            f"{root}.gate.e_score_correction_bias",
        )
        selection = np.asarray(selected)
        target_router = router[selection] @ input_inverse
        target_bias = bias[selection].copy()
        target_bias -= target_bias.mean()
        state[f"{target}.router.weight"] = _convert(target_router)
        state[f"{target}.routing_bias"] = _convert(target_bias, dtype=torch.float32)
    else:
        state[f"{target}.router.weight"] = torch.zeros(
            (stable.num_experts, pin.shape[1]), dtype=torch.bfloat16
        )
        state[f"{target}.routing_bias"] = torch.zeros(
            stable.num_experts, dtype=torch.float32
        )
    return state


def save_incremental_moe_stage(
    *,
    source: SafetensorsSource,
    inspector: GLM53Inspector,
    stage: GLM53StageCalibration,
    config: IQHybridConfig,
    output_dir: Path,
    weight_map: dict[str, str],
) -> None:
    """Write common, shared, and each routed expert to independently indexed shards.

    Dense stages have one real shared donor expert plus function-neutral routed
    experts. Sparse stages preserve selected-ID ordering and exact linear
    coordinate transport, computing inverses once per stage.
    """
    stable = config.stable_moe
    if stable is None:
        raise GLM53IncrementalMoEError("Stable LatentMoE required")
    physical = stage.moe_physical_layer
    source_layer = stage.source_layer
    source_width = inspector.config.hidden_size
    intermediate_width = inspector.layout.moe_intermediate_size
    sparse = source_layer >= inspector.layout.first_k_dense_replace
    if sparse:
        if stage.expert_usage is None:
            raise GLM53IncrementalMoEError("sparse stage lacks expert usage")
        selected = select_experts_by_usage(
            stage.expert_usage, target_experts=stable.num_experts
        )
        if len(stage.expert_usage) != 256:
            raise GLM53IncrementalMoEError("donor expert usage width is not 256")
    else:
        if stage.dense_intermediate_map is None:
            raise GLM53IncrementalMoEError("dense stage lacks intermediate map")
        selected = ()
        # Dense expert intermediate maps need not be identity; those matrices
        # need a separate streamed transform. Fail before loading donor data.
        if stable.num_experts > 32:
            raise GLM53IncrementalMoEError(
                "incremental dense GLM transforms are not implemented; "
                "do not run a full expanded compile yet"
            )

    # Existing transform uses the same pinv for every expert. Reusing it
    # avoids repeated multi-GB decompositions.
    input_inverse = _inverse(stage.moe_input_map)
    latent_inverse = _inverse(stage.latent_map) if sparse else None
    if not np.isfinite(input_inverse).all() or (
        latent_inverse is not None and not np.isfinite(latent_inverse).all()
    ):
        raise GLM53IncrementalMoEError("nonfinite coordinate pseudoinverse")

    common = _common(
        source=source,
        stage=stage,
        stable=stable,
        physical=physical,
        sparse=sparse,
        selected=selected,
        source_width=source_width,
        input_inverse=input_inverse,
    )
    _save_shard(
        output_dir,
        f"model-moe-{stage.stage:03d}-common.safetensors",
        common,
        weight_map,
    )
    del common
    if not sparse:
        # This branch is not enabled for expanded checkpoints.
        raise GLM53IncrementalMoEError("dense incremental compile unavailable")

    root = f"model.layers.{source_layer}.mlp"
    target = f"layers.{physical}.moe"
    _save_expert(
        source=source,
        donor_prefix=f"{root}.shared_experts",
        output_prefix=f"{target}.shared_experts.0",
        save_filename=f"model-moe-{stage.stage:03d}-shared.safetensors",
        output_dir=output_dir,
        weight_map=weight_map,
        source_width=source_width,
        intermediate_width=intermediate_width,
        target_output_map=stage.residual_map.matrix,
        input_inverse=input_inverse,
    )

    assert latent_inverse is not None
    for slot, donor_id in enumerate(selected):
        _save_expert(
            source=source,
            donor_prefix=f"{root}.experts.{donor_id}",
            output_prefix=f"{target}.routed_experts.{slot}",
            save_filename=f"model-moe-{stage.stage:03d}-expert-{slot:03d}.safetensors",
            output_dir=output_dir,
            weight_map=weight_map,
            source_width=source_width,
            intermediate_width=intermediate_width,
            target_output_map=stage.latent_map.matrix,
            input_inverse=latent_inverse,
        )
        gc.collect()
