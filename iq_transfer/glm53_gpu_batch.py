"""CUDA-only bounded batched projection of GLM sparse MoE tensors.

One call processes several *real* routed experts using shared GPU-resident
WARM bases and batched GEMMs. It does not launch a separate benchmark and
never saves partial output: the caller atomically persists each completed
expert after the entire batch has passed validation.

The independent CPU/NumPy projection remains the numerical reference.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .glm53_gpu_projection import ProjectionComputeError, ProjectionOps


@dataclass(frozen=True)
class ExpertSpec:
    slot: int
    donor_id: int


# A conservative bound on the incoming BF16 donor tensors staged on the CPU.
_DEFAULT_BATCH_BYTES = 512 * 1024 * 1024


def batch_capacity(*, source_width: int, intermediate_width: int,
                   expert_count: int, maximum: int | None = None,
                   bytes_limit: int = _DEFAULT_BATCH_BYTES) -> int:
    """Select a bounded number of routed experts, never drop donor coverage."""
    if maximum is None:
        raw = os.environ.get("IQ_MOE_GPU_BATCH_EXPERTS", "4")
        try:
            maximum = int(raw)
        except ValueError as exc:
            raise ProjectionComputeError("invalid IQ_MOE_GPU_BATCH_EXPERTS") from exc
    if not (1 <= maximum <= 16) or source_width <= 0 or intermediate_width <= 0:
        raise ProjectionComputeError("invalid expert batch geometry/capacity")
    if bytes_limit <= 0 or expert_count <= 0:
        raise ProjectionComputeError("invalid expert byte budget/count")
    source_bytes = 3 * source_width * intermediate_width * 2
    if source_bytes > bytes_limit:
        raise ProjectionComputeError(
            "one BF16 donor expert exceeds GPU batch host-memory limit"
        )
    return min(maximum, expert_count, max(1, bytes_limit // source_bytes))


def device_bases(ops: ProjectionOps, *, input_inverse: np.ndarray,
                 output_map: np.ndarray, source_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Upload verified fixed bases once per stage, not once per expert."""
    if not ops.cuda:
        raise ProjectionComputeError("batched expert projection requires CUDA")
    inp, out = np.asarray(input_inverse), np.asarray(output_map)
    if (inp.ndim != 2 or out.ndim != 2 or
        inp.shape[0] != source_width or out.shape[0] != source_width or
        not np.isfinite(inp).all() or not np.isfinite(out).all()):
        raise ProjectionComputeError("invalid sparse expert projection bases")
    # Contiguous copies isolate calibration arrays from later accidental edits.
    with torch.inference_mode():
        a = torch.from_numpy(np.array(inp, dtype=np.float32, order="C", copy=True)).to(ops.device)
        b = torch.from_numpy(np.array(out.T, dtype=np.float32, order="C", copy=True)).to(ops.device)
    return a, b


def _validated_weight(source: Any, key: str, shape: tuple[int, int]) -> torch.Tensor:
    tensor = source.get(key)
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ProjectionComputeError(f"{key}: unexpected donor tensor shape")
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise ProjectionComputeError(f"{key}: unsupported donor dtype {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ProjectionComputeError(f"{key}: nonfinite source weight")
    return tensor.detach().to(device="cpu").contiguous()


@torch.inference_mode()
def project_routed_batch(
    *,
    source: Any,
    donor_root: str,
    specs: list[ExpertSpec],
    source_width: int,
    intermediate_width: int,
    input_basis: torch.Tensor,
    output_basis_t: torch.Tensor,
    ops: ProjectionOps,
) -> list[tuple[dict[str, torch.Tensor], dict[str, dict[str, float | None]]]]:
    """Return per-expert BF16 CPU weights and provenance-neutral energy metrics.

    Computation uses FP32 CUDA and one batched GEMM per gate/up/down kind.
    No writes occur here; caller commits shards in slot order, atomically.
    """
    if not ops.cuda or not specs or len(specs) > 16:
        raise ProjectionComputeError("invalid CUDA expert batch")
    if (tuple(input_basis.shape)[0] != source_width or
        tuple(output_basis_t.shape)[1] != source_width or
        input_basis.device.type != "cuda" or output_basis_t.device.type != "cuda"):
        raise ProjectionComputeError("GPU basis shape/device mismatch")
    if len({s.slot for s in specs}) != len(specs):
        raise ProjectionComputeError("duplicate expert slot in batch")

    entries: list[tuple[dict[str, torch.Tensor], dict[str, dict[str, float | None]]]] = [
        ({}, {}) for _ in specs
    ]
    for kind in ("gate", "up", "down"):
        shape = ((intermediate_width, source_width) if kind != "down"
                 else (source_width, intermediate_width))
        raw = [
            _validated_weight(
                source, f"{donor_root}.experts.{spec.donor_id}.{kind}_proj.weight", shape
            ) for spec in specs
        ]
        # Direct BF16/FP32 tensor -> CUDA FP32; no intermediate NumPy FP64
        # matrices, unnecessary pseudoinverses, or repeated basis uploads.
        stacked = torch.stack([value.to(ops.device, dtype=torch.float32) for value in raw])
        if kind == "down":
            transformed = torch.matmul(output_basis_t, stacked)
        else:
            transformed = torch.matmul(stacked, input_basis)
        if not bool(torch.isfinite(transformed).all()):
            raise ProjectionComputeError(f"nonfinite CUDA output in {kind}")
        # One batched GPU energy reduction and one BF16 device->host transfer
        # instead of B extra CPU FP64 scans and B independent D2H copies.
        original_energy = (
            stacked.square().sum(dim=(-2, -1), dtype=torch.float64).cpu().tolist()
        )
        projected_energy_batch = (
            transformed.square().sum(dim=(-2, -1), dtype=torch.float64).cpu().tolist()
        )
        packed_cpu = transformed.to(dtype=torch.bfloat16).cpu()
        for index in range(len(raw)):
            source_energy = float(original_energy[index])
            projected_energy = float(projected_energy_batch[index])
            if not np.isfinite(source_energy) or not np.isfinite(projected_energy):
                raise ProjectionComputeError("nonfinite expert projection energy")
            entries[index][0][f"{kind}_proj.weight"] = (
                packed_cpu[index].contiguous()
            )
            entries[index][1][kind] = {
                "donor_frobenius_sq": source_energy,
                "projected_frobenius_sq": projected_energy,
                "norm_energy_ratio": (
                    projected_energy / source_energy if source_energy else None
                ),
            }
        del stacked, transformed, raw, packed_cpu
    return entries
