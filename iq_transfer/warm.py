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



def align_to_mamba_replication_frame(
    coordinate_map: CoordinateMap,
    *,
    foundation_features: int,
    source_space: str | None = None,
    target_space: str | None = None,
) -> CoordinateMap:
    """Fix WARM's target-space rotational freedom to widened Mamba coordinates.

    IQ widens the pretrained Mamba residual x in R^d to [x,x] in R^(2d).
    WARM identifies a 2d-dimensional GLM source subspace but its target basis is
    otherwise arbitrary up to right multiplication by an orthogonal matrix.

    We remove that ambiguity by mapping the strongest d retained WARM
    coordinates into Mamba's pretrained symmetric subspace
        s_i = (e_i + e_{i+d}) / sqrt(2)
    and the remaining d coordinates into the orthogonal anti-symmetric
    expansion subspace
        a_i = (e_i - e_{i+d}) / sqrt(2).

    This preserves Q^TQ=I while making the Mamba foundation's exact replicated
    subspace the canonical recipient frame.
    """
    q = np.asarray(coordinate_map.matrix, dtype=np.float64)
    if foundation_features <= 0:
        raise WarmRemapError("foundation_features must be positive")
    target_features = q.shape[1]
    if target_features != 2 * foundation_features:
        raise WarmRemapError(
            "Mamba replication alignment requires target width == 2 * foundation width"
        )
    scale = 1.0 / np.sqrt(2.0)
    frame = np.zeros((target_features, target_features), dtype=np.float64)
    d = foundation_features
    idx = np.arange(d)
    # Rows are target-basis vectors because row-vector coordinates use y=c@R.
    frame[idx, idx] = scale
    frame[idx, idx + d] = scale
    frame[idx + d, idx] = scale
    frame[idx + d, idx + d] = -scale
    aligned = q @ frame
    error = orthogonality_error(aligned)
    if error > 1e-8:
        raise WarmRemapError(
            f"Mamba-aligned WARM map failed orthogonality check: {error}"
        )
    return CoordinateMap(
        matrix=aligned,
        ridge=coordinate_map.ridge,
        source_space=source_space or coordinate_map.source_space,
        target_space=target_space or coordinate_map.target_space,
        diagnostics=coordinate_map.diagnostics,
    )

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
    selected_values = eigenvalues[-target_features:].flip(0).clamp_min(0)
    basis = eigenvectors[:, -target_features:].flip(1)

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
