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
