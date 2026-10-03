from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import torch

from .transport import CoordinateMap


class WarmRemapError(RuntimeError):
    pass


WeightOrientation = Literal["input", "output"]


@dataclass(frozen=True)
class WarmWeightOperator:
    weight: Any
    orientation: WeightOrientation
    name: str = ""
    coefficient: float = 1.0


@dataclass(frozen=True)
class WarmRemapDiagnostics:
    source_features: int
    target_features: int
    operator_count: int
    orthogonality_error: float
    retained_weight_energy: float


@dataclass(frozen=True)
class WarmRemap:
    coordinate_map: CoordinateMap
    diagnostics: WarmRemapDiagnostics


def _tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device=device, dtype=torch.float32)
    else:
        result = torch.as_tensor(np.asarray(value), device=device, dtype=torch.float32)
    if result.ndim != 2:
        raise WarmRemapError("WARM operators must be rank-2 weight matrices")
    if not bool(torch.isfinite(result).all()):
        raise WarmRemapError("WARM operator contains non-finite values")
    return result


def orthogonality_error(matrix: Any) -> float:
    q = np.asarray(matrix, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < q.shape[1]:
        raise WarmRemapError("semi-orthogonal map must have rows >= columns")
    gram = q.T @ q
    return float(np.linalg.norm(gram - np.eye(q.shape[1]), ord="fro"))


def fit_weight_orthogonal_remap(
    operators: Iterable[WarmWeightOperator],
    *,
    source_features: int,
    target_features: int,
    source_space: str,
    target_space: str,
    device: str | torch.device = "cpu",
) -> WarmRemap:
    """Build a WARM-style dense semi-orthogonal source->target feature map.

    Each source operator contributes a normalized feature-space Gram matrix:
      input-facing  W[out, source] -> W^T W
      output-facing W[source, in]  -> W W^T

    The dominant target_features eigenspace is retained. Contributions are
    normalized per operator so a large matrix cannot dominate only because it
    contains more parameters. The resulting Q satisfies Q^T Q ~= I and can be
    used by transport_linear() as a well-conditioned rectangular basis map.

    This borrows WARM's orthogonality-preserving remapping principle while
    keeping IQ's architecture-specific MLA/DSA/MoE transformations separate.
    """
    if source_features <= 0 or target_features <= 0:
        raise WarmRemapError("feature dimensions must be positive")
    if target_features > source_features:
        raise WarmRemapError("target_features cannot exceed source_features")

    dev = torch.device(device)
    gram = torch.zeros(
        (source_features, source_features),
        dtype=torch.float32,
        device=dev,
    )
    count = 0
    for operator in operators:
        if operator.coefficient <= 0:
            raise WarmRemapError("operator coefficient must be positive")
        weight = _tensor(operator.weight, device=dev)
        if operator.orientation == "input":
            if weight.shape[1] != source_features:
                raise WarmRemapError(
                    f"{operator.name or 'operator'} input width {weight.shape[1]} "
                    f"!= source_features {source_features}"
                )
            contribution = weight.T @ weight
        elif operator.orientation == "output":
            if weight.shape[0] != source_features:
                raise WarmRemapError(
                    f"{operator.name or 'operator'} output width {weight.shape[0]} "
                    f"!= source_features {source_features}"
                )
            contribution = weight @ weight.T
        else:
            raise WarmRemapError(
                f"unsupported WARM operator orientation: {operator.orientation!r}"
            )
        trace = torch.trace(contribution).clamp_min(torch.finfo(torch.float32).tiny)
        gram.add_(contribution, alpha=float(operator.coefficient) / float(trace))
        count += 1
        del contribution, weight

    if count == 0:
        raise WarmRemapError("at least one WARM operator is required")
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    selected_values = eigenvalues[-target_features:].clamp_min(0)
    basis = eigenvectors[:, -target_features:]

    # Fix eigenvector sign ambiguity deterministically: the largest-magnitude
    # coordinate in each column is always positive.
    peaks = basis.abs().argmax(dim=0)
    columns = torch.arange(target_features, device=dev)
    signs = torch.sign(basis[peaks, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs.unsqueeze(0)

    q = basis.detach().cpu().double().numpy()
    # Float32 eigensolvers are sufficient for the expensive decomposition;
    # QR in float64 tightens the final Stiefel constraint before persistence.
    q, _ = np.linalg.qr(q, mode="reduced")
    total = float(eigenvalues.clamp_min(0).sum().item())
    retained = (
        float(selected_values.sum().item()) / total
        if total > 0
        else 0.0
    )
    error = orthogonality_error(q)
    if not np.isfinite(error) or error > 1e-8:
        raise WarmRemapError(
            f"WARM remap failed orthogonality check: {error}"
        )

    coordinate_map = CoordinateMap(
        matrix=q,
        ridge=1e-12,
        source_space=source_space,
        target_space=target_space,
        diagnostics=None,
    )
    return WarmRemap(
        coordinate_map=coordinate_map,
        diagnostics=WarmRemapDiagnostics(
            source_features=source_features,
            target_features=target_features,
            operator_count=count,
            orthogonality_error=error,
            retained_weight_energy=retained,
        ),
    )
