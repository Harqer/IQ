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
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import SafetensorsSource
from .complete_transplant import _save_shard
from .glm53 import GLM53Inspector
from .glm53_calibration import GLM53StageCalibration
from .glm53_moe import select_experts_by_usage
from .glm53_resume import register_saved_shard, expected_expert_keys, expected_expert_shapes
from .glm53_gpu_projection import ProjectionOps
from .transport import CoordinateMap
from iq_model import IQHybridConfig


_INVERSE_DENSE_THRESHOLD = 8_000_000


class GLM53IncrementalMoEError(RuntimeError):
    pass


def _check_runtime_budget(stage: int, slot: int = -1) -> None:
    """Stop BEFORE starting another expensive expert; saved shards remain reusable."""
    deadline = os.environ.get("IQ_TRANSFER_DEADLINE_EPOCH")
    if not deadline:
        return
    if not deadline.isdigit():
        raise GLM53IncrementalMoEError("invalid IQ_TRANSFER_DEADLINE_EPOCH")
    # Leave 120 seconds for ordinary Python exception/cleanup before job timeout.
    if time.time() + 120 >= int(deadline):
        print(f"GLM53_CHECKPOINTED_BUDGET_STOP stage={stage} next_slot={slot}",
              flush=True)
        raise GLM53IncrementalMoEError(
            "transfer time budget reached; persisted complete shards can resume"
        )


def _energy(value: np.ndarray) -> float:
    return float(np.sum(np.square(value), dtype=np.float64))


def _energy_ratio(a: float, b: float) -> dict[str, float | None]:
    if not np.isfinite(a) or not np.isfinite(b) or min(a, b) < 0:
        raise GLM53IncrementalMoEError("nonfinite projection energy")
    return {
        "donor_frobenius_sq": a,
        "projected_frobenius_sq": b,
        "norm_energy_ratio": b / a if a else None,
    }


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Validate and convert a single tensor; no collection of donor experts."""
    if isinstance(value, torch.Tensor):
        # NumPy cannot directly view torch.bfloat16; BF16 -> FP32 -> FP64
        # is exact because every BF16 value is representable in FP32.
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
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


def _inverse(map_: CoordinateMap, projection_ops: ProjectionOps | None = None) -> np.ndarray:
    # The canonical weight-only subcloning maps have orthonormal one-hot
    # columns. Their Moore-Penrose transpose equals the original matrix.
    # Recognize that exact case in linear memory/time and skip a huge SVD.
    matrix = np.asarray(map_.matrix, dtype=np.float64)
    if (
        matrix.shape[0] >= matrix.shape[1]
        and np.all(np.count_nonzero(matrix, axis=1) <= 1)
        and np.all(np.sum(matrix * matrix, axis=0) == 1.0)
    ):
        return matrix
    # WARM's canonical maps are dense semi-orthogonal, not one-hot.
    # Verify the full Gram matrix by small column blocks before exploiting
    # pinv(P).T == P. This uses O(k * block_cols) scratch, not O(k*k).
    if matrix.size > _INVERSE_DENSE_THRESHOLD:
        if projection_ops is not None and projection_ops.cuda:
            if not projection_ops.check_orthonormal(matrix):
                raise GLM53IncrementalMoEError("GPU WARM basis not orthonormal")
            return matrix
        if matrix.shape[0] < matrix.shape[1]:
            raise GLM53IncrementalMoEError(
                "large wide coordinate map requires a bounded inverse solver"
            )
        for start in range(0, matrix.shape[1], 128):
            stop = min(start + 128, matrix.shape[1])
            gram = matrix.T @ matrix[:, start:stop]
            for offset, col in enumerate(range(start, stop)):
                gram[col, offset] -= 1.0
            if not np.isfinite(gram).all() or float(np.max(np.abs(gram))) > 1e-8:
                raise GLM53IncrementalMoEError(
                    "large non-orthogonal coordinate map needs a bounded inverse solver"
                )
        return matrix
    # Small general maps preserve the reference float64 pseudoinverse.
    return np.linalg.pinv(matrix).T


def _resume_expert_shard(
    *, output_dir: Path, filename: str, weight_map: dict[str, str],
    physical: int, slot: int, stable: Any,
) -> bool:
    """Skip completed experts only with the exact expected tensor contract.

    safetensors shape/dtype checks use headers without loading full tensors.
    This applies equally to CPU, CUDA and function-neutral dense stages.
    """
    expected = expected_expert_shapes(
        physical, slot, latent_width=stable.latent_size,
        intermediate_width=stable.expert_intermediate_size,
    )
    return register_saved_shard(
        output_dir, filename, weight_map, layer=physical,
        expected_keys=expected_expert_keys(physical, slot),
        expected_shapes=expected,
        expected_dtypes={key: "BF16" for key in expected},
    )


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
    middle_map: np.ndarray | None = None,
    middle_inverse: np.ndarray | None = None,
    projection_ops: ProjectionOps | None = None,
) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    if (middle_map is None) != (middle_inverse is None):
        raise GLM53IncrementalMoEError("intermediate map and inverse must be paired")
    if middle_map is not None and (
        middle_map.shape[0] != intermediate_width
        or middle_inverse is None
        or middle_inverse.shape[0] != intermediate_width
    ):
        raise GLM53IncrementalMoEError("expert intermediate-map source width mismatch")
    if target_output_map.shape[0] != source_width:
        raise GLM53IncrementalMoEError("expert output-map source width mismatch")
    if input_inverse.shape[0] != source_width:
        raise GLM53IncrementalMoEError("expert input-map source width mismatch")

    state: dict[str, torch.Tensor] = {}
    dense = middle_map is not None
    for kind in ("gate", "up"):
        key = f"{donor_prefix}.{kind}_proj.weight"
        shape = (intermediate_width, source_width)
        source_energy = 0.0
        if dense:
            if tuple(source.shape(key)) != shape:
                raise GLM53IncrementalMoEError(f"{key}: unexpected donor shape")
            assert middle_map is not None
            # Read only donor rows needed by each block. Accumulate the
            # calibrated intermediate projection before applying input map.
            projected = np.zeros(
                (middle_map.shape[1], source_width), dtype=np.float64
            )
            for row in range(0, intermediate_width, 128):
                stop = min(row + 128, intermediate_width)
                chunk = _finite(
                    source.get_rows(key, range(row, stop)),
                    (stop - row, source_width),
                    key,
                )
                source_energy += _energy(chunk)
                projected += middle_map[row:stop].T @ chunk
                del chunk
            result = projection_ops.matmul(projected, input_inverse, cache_right=True) if projection_ops else projected @ input_inverse
            del projected
        else:
            w = _finite(source.get(key), shape, key)
            source_energy = _energy(w)
            result = projection_ops.matmul(w, input_inverse, cache_right=True) if projection_ops else w @ input_inverse
            del w
        metrics[kind] = _energy_ratio(source_energy, _energy(result))
        state[f"{output_prefix}.{kind}_proj.weight"] = _convert(result)
        del result

    key = f"{donor_prefix}.down_proj.weight"
    shape = (source_width, intermediate_width)
    source_energy = 0.0
    if dense:
        if tuple(source.shape(key)) != shape:
            raise GLM53IncrementalMoEError(f"{key}: unexpected donor shape")
        assert middle_inverse is not None
        result = np.zeros(
            (target_output_map.shape[1], middle_inverse.shape[1]),
            dtype=np.float64,
        )
        for row in range(0, source_width, 128):
            stop = min(row + 128, source_width)
            chunk = _finite(
                source.get_rows(key, range(row, stop)),
                (stop - row, intermediate_width),
                key,
            )
            # (P_out.T @ W) @ pinv(P_mid).T can be accumulated in
            # row-blocks: P_out[rows].T @ (W[rows] @ pinv(P_mid).T).
            source_energy += _energy(chunk)
            result += target_output_map[row:stop].T @ (
                chunk @ middle_inverse
            )
            del chunk
    else:
        down = _finite(source.get(key), shape, key)
        source_energy = _energy(down)
        result = projection_ops.matmul(target_output_map.T, down, cache_left=True) if projection_ops else target_output_map.T @ down
        del down
    metrics["down"] = _energy_ratio(source_energy, _energy(result))
    state[f"{output_prefix}.down_proj.weight"] = _convert(result)
    del result
    _save_shard(output_dir, save_filename, state, weight_map)
    del state
    return metrics

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
    latent_inverse: np.ndarray,
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
    latent_up = pout.T @ latent_inverse
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
    resume_existing: bool = False,
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
    # Fail before any weight reads if calibration geometry cannot produce
    # the exact IQ model parameter shapes.
    if (
        stage.moe_input_map.matrix.shape != (source_width, stable.hidden_size)
        or stage.residual_map.matrix.shape != (source_width, stable.hidden_size)
        or stage.latent_map.matrix.shape != (source_width, stable.latent_size)
    ):
        raise GLM53IncrementalMoEError("calibration geometry differs from target IQ widths")
    sparse = source_layer >= inspector.layout.first_k_dense_replace
    if sparse and intermediate_width != stable.expert_intermediate_size:
        raise GLM53IncrementalMoEError("sparse donor expert width differs from IQ expert width")
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
        if stage.dense_intermediate_map.matrix.shape[1] != stable.expert_intermediate_size:
            raise GLM53IncrementalMoEError("dense calibration intermediate target width differs")
        selected = ()

    # Existing transform uses the same pinv for every expert. Reusing it
    # avoids repeated multi-GB decompositions.
    _check_runtime_budget(stage.stage)
    projection_ops = ProjectionOps()
    input_inverse = _inverse(stage.moe_input_map, projection_ops)
    latent_inverse = _inverse(stage.latent_map, projection_ops)
    if projection_ops.cuda:
        print(f"GLM53_GPU_PROJECTION stage={stage.stage} device={projection_ops.device}", flush=True)
    if not np.isfinite(input_inverse).all() or not np.isfinite(latent_inverse).all():
        raise GLM53IncrementalMoEError("nonfinite coordinate pseudoinverse")

    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage.stage,
        "source_layer": source_layer,
        "target_layer": physical,
        "projection_type": "shared_WARM_subspaces",
        "caution": (
            "Frobenius energy ratio is subspace-retained energy only when "
            "coordinate bases are orthonormal; it is not functional fidelity."
        ),
        "routed": {},
    }
    common_name = f"model-moe-{stage.stage:03d}-common.safetensors"
    if not (
        resume_existing and register_saved_shard(
            output_dir, common_name, weight_map, layer=physical
        )
    ):
        common = _common(
            source=source,
            stage=stage,
            stable=stable,
            physical=physical,
            sparse=sparse,
            selected=selected,
            source_width=source_width,
            input_inverse=input_inverse,
            latent_inverse=latent_inverse,
        )
        _save_shard(output_dir, common_name, common, weight_map)
        del common
    root = f"model.layers.{source_layer}.mlp"
    if not sparse:
        # The actual dense donor function is retained in the shared expert;
        # all routed experts are deliberately zero, matching the old compiler.
        middle = stage.dense_intermediate_map
        assert middle is not None
        middle_inverse = _inverse(middle, projection_ops)
        shared_name = f"model-moe-{stage.stage:03d}-shared.safetensors"
        if resume_existing and register_saved_shard(
            output_dir, shared_name, weight_map, layer=physical
        ):
            report["shared"] = {"resumed_existing_shard": True}
        else:
            report["shared"] = _save_expert(
                source=source,
                donor_prefix=root,
                output_prefix=f"layers.{physical}.moe.shared_experts.0",
                save_filename=f"model-moe-{stage.stage:03d}-shared.safetensors",
                output_dir=output_dir,
                weight_map=weight_map,
                source_width=source_width,
                intermediate_width=middle.matrix.shape[0],
                target_output_map=stage.residual_map.matrix,
                input_inverse=input_inverse,
                middle_map=middle.matrix,
                middle_inverse=middle_inverse,
                projection_ops=projection_ops,
            )
        target = f"layers.{physical}.moe"
        for slot in range(stable.num_experts):
            _check_runtime_budget(stage.stage, slot)
            expert_name = f"model-moe-{stage.stage:03d}-expert-{slot:03d}.safetensors"
            if resume_existing and _resume_expert_shard(
                output_dir=output_dir, filename=expert_name,
                weight_map=weight_map, physical=physical, slot=slot,
                stable=stable,
            ):
                continue
            zeros: dict[str, torch.Tensor] = {
                f"{target}.routed_experts.{slot}.gate_proj.weight": torch.zeros(
                    (stable.expert_intermediate_size, stable.latent_size),
                    dtype=torch.bfloat16,
                ),
                f"{target}.routed_experts.{slot}.up_proj.weight": torch.zeros(
                    (stable.expert_intermediate_size, stable.latent_size),
                    dtype=torch.bfloat16,
                ),
                f"{target}.routed_experts.{slot}.down_proj.weight": torch.zeros(
                    (stable.latent_size, stable.expert_intermediate_size),
                    dtype=torch.bfloat16,
                ),
            }
            _save_shard(
                output_dir,
                f"model-moe-{stage.stage:03d}-expert-{slot:03d}.safetensors",
                zeros,
                weight_map,
            )
            del zeros
        (output_dir / f"model-moe-{stage.stage:03d}-projection.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        projection_ops.close()
        return

    target = f"layers.{physical}.moe"
    shared_name = f"model-moe-{stage.stage:03d}-shared.safetensors"
    if resume_existing and register_saved_shard(
        output_dir, shared_name, weight_map, layer=physical
    ):
        report["shared"] = {"resumed_existing_shard": True}
    else:
        report["shared"] = _save_expert(
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
            projection_ops=projection_ops,
        )


    if projection_ops.cuda:
        # Batch real routed experts in the SAME full checkpoint compilation.
        # Every completed expert remains an independently resumable shard.
        from .glm53_gpu_batch import (
            ExpertSpec, batch_capacity, device_bases, project_routed_batch,
        )

        pending: list[ExpertSpec] = []
        for slot, donor_id in enumerate(selected):
            expert_name = f"model-moe-{stage.stage:03d}-expert-{slot:03d}.safetensors"
            if resume_existing and _resume_expert_shard(
                output_dir=output_dir, filename=expert_name,
                weight_map=weight_map, physical=physical, slot=slot,
                stable=stable,
            ):
                report["routed"][str(slot)] = {
                    "donor_expert_id": int(donor_id),
                    "resumed_existing_shard": True,
                }
            else:
                pending.append(ExpertSpec(slot=slot, donor_id=int(donor_id)))
        if pending:
            batch_size = batch_capacity(
                source_width=source_width,
                intermediate_width=intermediate_width,
                expert_count=len(pending),
            )
            input_basis, output_basis_t = device_bases(
                projection_ops, input_inverse=latent_inverse,
                output_map=stage.latent_map.matrix, source_width=source_width,
            )
            from .glm53_batch_prefetch import RoutedBatchPrefetch

            device_batch_limit = batch_size
            stage_compute_s = 0.0
            stage_save_s = 0.0
            stage_prefetch_wait_s = 0.0

            def commit_batch(
                specs: list[ExpertSpec], planned_source: Any,
            ) -> None:
                nonlocal device_batch_limit, stage_compute_s, stage_save_s
                # Preserve already staged HTTP tensors across CUDA OOM retries.
                if len(specs) > device_batch_limit:
                    cursor = 0
                    while cursor < len(specs):
                        part = specs[cursor:cursor + device_batch_limit]
                        commit_batch(part, planned_source)
                        cursor += len(part)
                    return
                compute_started = time.perf_counter()
                try:
                    results = project_routed_batch(
                        source=planned_source,
                        donor_root=root,
                        specs=specs,
                        source_width=source_width,
                        intermediate_width=intermediate_width,
                        input_basis=input_basis,
                        output_basis_t=output_basis_t,
                        ops=projection_ops,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    # No new writes have happened within the failed GEMM.
                    # Split the *already downloaded* batch; never redownload.
                    if len(specs) == 1:
                        raise GLM53IncrementalMoEError(
                            "CUDA OOM for one expert; resume on larger GPU"
                        ) from exc
                    device_batch_limit = max(1, len(specs) // 2)
                    torch.cuda.empty_cache()
                    print(
                        f"GLM53_GPU_BATCH_OOM_REDUCED stage={stage.stage} "
                        f"experts_per_batch={device_batch_limit}", flush=True,
                    )
                    left_count = device_batch_limit
                    commit_batch(specs[:left_count], planned_source)
                    commit_batch(specs[left_count:], planned_source)
                    return
                stage_compute_s += time.perf_counter() - compute_started
                # Individual atomic shards are the recovery boundary even
                # when several experts share one GPU batched operation.
                save_started = time.perf_counter()
                for spec, (weights, metrics) in zip(specs, results):
                    prefix = f"{target}.routed_experts.{spec.slot}"
                    state = {
                        f"{prefix}.{name}": tensor
                        for name, tensor in weights.items()
                    }
                    _save_shard(
                        output_dir,
                        f"model-moe-{stage.stage:03d}-expert-{spec.slot:03d}.safetensors",
                        state, weight_map,
                    )
                    report["routed"][str(spec.slot)] = {
                        "donor_expert_id": spec.donor_id,
                        "operators": metrics,
                    }
                stage_save_s += time.perf_counter() - save_started
                del results

            # Read the NEXT bounded remote batch while CUDA computes/saves the
            # current one. A separate HF Session prevents unsafe shared state.
            # Local/test sources deliberately retain the direct read path.
            with RoutedBatchPrefetch(source, donor_root=root) as prefetch:
                started_at = time.perf_counter()
                completed = 0
                # Check the paid-worker deadline *before* scheduling the
                # first prefetch, avoiding a needless outstanding network
                # read if this job must checkpoint/stop immediately.
                _check_runtime_budget(stage.stage, pending[0].slot)
                future = prefetch.submit(pending[:batch_size])
                while completed < len(pending):
                    specs = pending[completed:completed + batch_size]
                    _check_runtime_budget(stage.stage, specs[0].slot)
                    next_offset = completed + len(specs)
                    upcoming = (
                        pending[next_offset:next_offset + batch_size]
                        if next_offset < len(pending) else []
                    )
                    next_future = prefetch.submit(upcoming)
                    wait_started = time.perf_counter()
                    staged_source = prefetch.source_for(future, source)
                    stage_prefetch_wait_s += time.perf_counter() - wait_started
                    commit_batch(specs, staged_source)
                    del staged_source
                    completed += len(specs)
                    future = next_future
                    elapsed_s = max(1e-9, time.perf_counter() - started_at)
                    # Source tensor payload bytes are exact for BF16 GLM routed
                    # experts; this rate includes reads, CUDA and atomic writes.
                    processed_mib = (
                        completed * 3 * source_width * intermediate_width * 2
                        / (1024 * 1024)
                    )
                    print(
                        f"GLM53_GPU_BATCH_PROGRESS layer={stage.stage} "
                        f"saved={completed}/{len(pending)} "
                        f"planned_batch={len(specs)} gpu_batch={device_batch_limit} "
                        f"elapsed_s={elapsed_s:.1f} "
                        f"source_mib_per_s={processed_mib/elapsed_s:.1f} "
                        f"prefetch_wait_s={stage_prefetch_wait_s:.2f} "
                        f"batch_compute_s={stage_compute_s:.2f} "
                        f"shard_save_s={stage_save_s:.2f} "
                        f"cuda_peak_allocated_mib="
                        f"{torch.cuda.max_memory_allocated()/(1024*1024):.0f}",
                        flush=True,
                    )
            del input_basis, output_basis_t
    else:
        # Reference implementation kept for offline CPU parity and fixtures.
        for slot, donor_id in enumerate(selected):
            _check_runtime_budget(stage.stage, slot)
            expert_name = f"model-moe-{stage.stage:03d}-expert-{slot:03d}.safetensors"
            if resume_existing and _resume_expert_shard(
                output_dir=output_dir, filename=expert_name,
                weight_map=weight_map, physical=physical, slot=slot,
                stable=stable,
            ):
                report["routed"][str(slot)] = {
                    "donor_expert_id": int(donor_id),
                    "resumed_existing_shard": True,
                }
                continue
            report["routed"][str(slot)] = {
                "donor_expert_id": int(donor_id),
                "operators": _save_expert(
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
                    projection_ops=projection_ops,
                ),
            }
    projection_ops.close()
    (output_dir / f"model-moe-{stage.stage:03d}-projection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
