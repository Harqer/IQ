from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np


class TransportError(RuntimeError):
    pass


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class MapDiagnostics:
    sample_count: int
    source_features: int
    target_features: int
    effective_rank: int
    condition_number: float
    fit_rmse: float
    validation_rmse: float | None = None


@dataclass(frozen=True)
class CoordinateMap:
    matrix: np.ndarray
    ridge: float
    source_space: str = "generic_source"
    target_space: str = "generic_target"
    diagnostics: MapDiagnostics | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise TransportError("coordinate map matrix must be rank-2")
        if not np.isfinite(matrix).all():
            raise TransportError("coordinate map contains non-finite values")
        if self.ridge <= 0:
            raise TransportError("coordinate map ridge must be positive")
        if not self.source_space.strip() or not self.target_space.strip():
            raise TransportError("coordinate map spaces must be non-empty")


def _rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    diff = actual - expected
    return float(np.sqrt(np.mean(diff * diff)))



def _validate_activation_pair(
    source_activations: Any,
    target_activations: Any,
) -> tuple[np.ndarray, np.ndarray]:
    xs = _numpy(source_activations)
    xt = _numpy(target_activations)
    if xs.ndim != 2 or xt.ndim != 2:
        raise TransportError("coordinate-map activations must be rank-2 [samples, features]")
    if xs.shape[0] != xt.shape[0]:
        raise TransportError("source and target activations must use paired samples")
    if xs.shape[0] < 2 or xs.shape[1] == 0 or xt.shape[1] == 0:
        raise TransportError("coordinate-map activations must contain samples and features")
    if not np.isfinite(xs).all() or not np.isfinite(xt).all():
        raise TransportError("coordinate-map activations contain non-finite values")
    return xs, xt


def channel_correlation_cost(
    source_activations: Any,
    target_activations: Any,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Transport-and-Merge style feature cost: 1 - Pearson correlation."""
    xs, xt = _validate_activation_pair(source_activations, target_activations)
    if eps <= 0:
        raise TransportError("eps must be positive")
    xs = xs - xs.mean(axis=0, keepdims=True)
    xt = xt - xt.mean(axis=0, keepdims=True)
    xs = xs / np.sqrt(np.sum(xs * xs, axis=0, keepdims=True) + eps)
    xt = xt / np.sqrt(np.sum(xt * xt, axis=0, keepdims=True) + eps)
    correlation = np.clip(xs.T @ xt, -1.0, 1.0)
    cost = 1.0 - correlation
    if not np.isfinite(cost).all():
        raise TransportError("correlation transport cost contains non-finite values")
    return cost


def sinkhorn_transport(
    cost: Any,
    *,
    regularization: float = 0.05,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Stable log-domain Sinkhorn coupling with uniform feature marginals."""
    c = _numpy(cost)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] == 0:
        raise TransportError("transport cost must be a non-empty matrix")
    if not np.isfinite(c).all():
        raise TransportError("transport cost contains non-finite values")
    if regularization <= 0 or max_iterations <= 0 or tolerance <= 0:
        raise TransportError("Sinkhorn controls must be positive")

    rows, cols = c.shape
    log_a = np.full(rows, -np.log(rows), dtype=np.float64)
    log_b = np.full(cols, -np.log(cols), dtype=np.float64)
    kernel = -c / regularization
    log_u = np.zeros(rows, dtype=np.float64)
    log_v = np.zeros(cols, dtype=np.float64)

    def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(x, axis=axis, keepdims=True)
        stable = maximum + np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=True))
        return np.squeeze(stable, axis=axis)

    for _ in range(max_iterations):
        next_u = log_a - logsumexp(kernel + log_v[None, :], axis=1)
        next_v = log_b - logsumexp(kernel.T + next_u[None, :], axis=1)
        delta = max(
            float(np.max(np.abs(next_u - log_u))),
            float(np.max(np.abs(next_v - log_v))),
        )
        log_u, log_v = next_u, next_v
        if delta <= tolerance:
            break

    coupling = np.exp(kernel + log_u[:, None] + log_v[None, :])
    total = coupling.sum()
    if not np.isfinite(total) or total <= 0:
        raise TransportError("Sinkhorn produced an invalid coupling")
    coupling /= total
    if not np.isfinite(coupling).all():
        raise TransportError("Sinkhorn coupling contains non-finite values")
    return coupling


def fit_ot_coordinate_map(
    source_activations: Any,
    target_activations: Any,
    *,
    ridge: float = 1e-3,
    regularization: float = 0.05,
    top_k_source: int | None = 64,
    source_space: str = "generic_source",
    target_space: str = "generic_target",
    validation_source: Any | None = None,
    validation_target: Any | None = None,
) -> CoordinateMap:
    """Fit a direct cross-architecture basis map with OT-supported sparse ridge.

    Optimal transport discovers source/target channel correspondence from
    activation correlation. For each target channel, ridge regression is then
    solved only on its strongest transported source support. This preserves
    signed scale information that a probability coupling alone cannot encode.
    """
    xs, xt = _validate_activation_pair(source_activations, target_activations)
    if ridge <= 0:
        raise TransportError("ridge must be positive")
    if top_k_source is not None and top_k_source <= 0:
        raise TransportError("top_k_source must be positive when configured")

    coupling = sinkhorn_transport(
        channel_correlation_cost(xs, xt),
        regularization=regularization,
    )
    source_features, target_features = coupling.shape
    support_size = (
        source_features
        if top_k_source is None
        else min(int(top_k_source), source_features)
    )
    matrix = np.zeros((source_features, target_features), dtype=np.float64)
    for target_index in range(target_features):
        scores = coupling[:, target_index]
        # OT narrows the candidate set; residual-aware greedy selection then
        # recovers weaker directions that marginal correlation can hide behind
        # a dominant source channel.
        pool_size = min(
            source_features,
            max(support_size, support_size * 4),
        )
        pool = np.argpartition(scores, -pool_size)[-pool_size:]
        selected: list[int] = []
        residual = xt[:, target_index].copy()
        coefficients = np.empty(0, dtype=np.float64)
        for _ in range(support_size):
            available = np.asarray(
                [index for index in pool if int(index) not in selected],
                dtype=np.int64,
            )
            if available.size == 0:
                break
            correlations = np.abs(xs[:, available].T @ residual)
            chosen = int(available[int(np.argmax(correlations))])
            selected.append(chosen)
            design = xs[:, selected]
            gram = design.T @ design
            gram.flat[:: gram.shape[0] + 1] += ridge
            rhs = design.T @ xt[:, target_index]
            try:
                coefficients = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError as exc:
                raise TransportError(
                    f"failed OT-supported ridge solve for target feature {target_index}"
                ) from exc
            residual = xt[:, target_index] - design @ coefficients
        if not selected:
            raise TransportError(
                f"OT support selection failed for target feature {target_index}"
            )
        matrix[np.asarray(selected), target_index] = coefficients

    fit_rmse = _rmse(xs @ matrix, xt)
    validation_rmse: float | None = None
    if (validation_source is None) != (validation_target is None):
        raise TransportError("validation_source and validation_target must be provided together")
    if validation_source is not None:
        vs, vt = _validate_activation_pair(validation_source, validation_target)
        if vs.shape[1] != xs.shape[1] or vt.shape[1] != xt.shape[1]:
            raise TransportError("validation feature dimensions must match fit activations")
        validation_rmse = _rmse(vs @ matrix, vt)

    singular = np.linalg.svd(xs, compute_uv=False)
    tolerance_rank = np.finfo(np.float64).eps * max(xs.shape) * singular[0]
    effective_rank = int(np.sum(singular > tolerance_rank))
    smallest = singular[-1]
    condition_number = (
        float("inf")
        if smallest <= tolerance_rank
        else float(singular[0] / smallest)
    )
    diagnostics = MapDiagnostics(
        sample_count=xs.shape[0],
        source_features=xs.shape[1],
        target_features=xt.shape[1],
        effective_rank=effective_rank,
        condition_number=condition_number,
        fit_rmse=fit_rmse,
        validation_rmse=validation_rmse,
    )
    return CoordinateMap(
        matrix=matrix,
        ridge=ridge,
        source_space=source_space,
        target_space=target_space,
        diagnostics=diagnostics,
    )

def fit_ridge_coordinate_map(
    source_activations: Any,
    target_activations: Any,
    *,
    ridge: float = 1e-3,
    source_space: str = "generic_source",
    target_space: str = "generic_target",
    validation_source: Any | None = None,
    validation_target: Any | None = None,
) -> CoordinateMap:
    """Fit x_target ~= x_source @ P using a sample-space ridge solution."""
    xs = _numpy(source_activations)
    xt = _numpy(target_activations)
    if xs.ndim != 2 or xt.ndim != 2:
        raise TransportError("coordinate-map activations must be rank-2 [samples, features]")
    if xs.shape[0] != xt.shape[0]:
        raise TransportError("source and target activations must use paired samples")
    if xs.shape[0] == 0 or xs.shape[1] == 0 or xt.shape[1] == 0:
        raise TransportError("coordinate-map activations must be non-empty")
    if ridge <= 0:
        raise TransportError("ridge must be positive")
    if not np.isfinite(xs).all() or not np.isfinite(xt).all():
        raise TransportError("coordinate-map activations contain non-finite values")

    gram = xs @ xs.T
    gram.flat[:: gram.shape[0] + 1] += ridge
    try:
        dual = np.linalg.solve(gram, xt)
    except np.linalg.LinAlgError as exc:
        raise TransportError("failed to solve ridge coordinate map") from exc
    matrix = xs.T @ dual
    fit_rmse = _rmse(xs @ matrix, xt)

    validation_rmse: float | None = None
    if (validation_source is None) != (validation_target is None):
        raise TransportError("validation_source and validation_target must be provided together")
    if validation_source is not None:
        vs = _numpy(validation_source)
        vt = _numpy(validation_target)
        if vs.ndim != 2 or vt.ndim != 2 or vs.shape[0] != vt.shape[0]:
            raise TransportError("validation activations must be paired rank-2 matrices")
        if vs.shape[1] != xs.shape[1] or vt.shape[1] != xt.shape[1]:
            raise TransportError("validation feature dimensions must match fit activations")
        if not np.isfinite(vs).all() or not np.isfinite(vt).all():
            raise TransportError("validation activations contain non-finite values")
        validation_rmse = _rmse(vs @ matrix, vt)

    singular = np.linalg.svd(xs, compute_uv=False)
    if singular.size == 0:
        raise TransportError("cannot diagnose empty source activation matrix")
    tolerance = np.finfo(np.float64).eps * max(xs.shape) * singular[0]
    effective_rank = int(np.sum(singular > tolerance))
    smallest = singular[-1]
    condition_number = float("inf") if smallest <= tolerance else float(singular[0] / smallest)

    diagnostics = MapDiagnostics(
        sample_count=xs.shape[0],
        source_features=xs.shape[1],
        target_features=xt.shape[1],
        effective_rank=effective_rank,
        condition_number=condition_number,
        fit_rmse=fit_rmse,
        validation_rmse=validation_rmse,
    )
    return CoordinateMap(matrix, ridge, source_space, target_space, diagnostics)


def transport_linear(weight_source: Any, input_map: CoordinateMap, output_map: CoordinateMap) -> np.ndarray:
    """Transport a source linear operator into target coordinates.

    Row-vector convention:
      x_t ~= x_s @ P_in
      y_t ~= y_s @ P_out
      y_s = x_s @ W_s.T

    Therefore W_t = P_out.T @ W_s @ pinv(P_in).T.
    """
    ws = _numpy(weight_source)
    pin = input_map.matrix
    pout = output_map.matrix
    if ws.ndim != 2:
        raise TransportError("source weight must be rank-2")
    if not np.isfinite(ws).all():
        raise TransportError("source weight contains non-finite values")
    if ws.shape != (pout.shape[0], pin.shape[0]):
        raise TransportError(
            f"shape mismatch: weight={ws.shape}, source_out={pout.shape[0]}, source_in={pin.shape[0]}"
        )
    transported = pout.T @ ws @ np.linalg.pinv(pin).T
    if not np.isfinite(transported).all():
        raise TransportError("transported weight contains non-finite values")
    return transported


def save_coordinate_map(coordinate_map: CoordinateMap, path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    tensor_path = path.with_suffix(".safetensors")
    metadata_path = path.with_suffix(".json")
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise TransportError("safetensors is required to save coordinate maps") from exc

    save_file({"matrix": np.asarray(coordinate_map.matrix, dtype=np.float64)}, str(tensor_path))
    metadata = {
        "schema_version": 1,
        "ridge": coordinate_map.ridge,
        "source_space": coordinate_map.source_space,
        "target_space": coordinate_map.target_space,
        "diagnostics": asdict(coordinate_map.diagnostics) if coordinate_map.diagnostics is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    return tensor_path, metadata_path


def load_coordinate_map(path: str | Path) -> CoordinateMap:
    path = Path(path)
    tensor_path = path.with_suffix(".safetensors")
    metadata_path = path.with_suffix(".json")
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise TransportError("safetensors is required to load coordinate maps") from exc
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise TransportError(f"coordinate-map artifact is incomplete: {path}")
    tensors = load_file(str(tensor_path))
    if set(tensors) != {"matrix"}:
        raise TransportError("coordinate-map safetensors must contain exactly the 'matrix' tensor")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise TransportError(f"unsupported coordinate-map schema: {metadata.get('schema_version')}")
    diagnostics_data = metadata.get("diagnostics")
    diagnostics = MapDiagnostics(**diagnostics_data) if diagnostics_data is not None else None
    return CoordinateMap(
        matrix=tensors["matrix"],
        ridge=float(metadata["ridge"]),
        source_space=str(metadata["source_space"]),
        target_space=str(metadata["target_space"]),
        diagnostics=diagnostics,
    )
