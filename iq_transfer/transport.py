from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
class CoordinateMap:
    matrix: np.ndarray
    ridge: float


def fit_ridge_coordinate_map(source_activations: Any, target_activations: Any, *, ridge: float = 1e-3) -> CoordinateMap:
    """Fit x_target ~= x_source @ P using a sample-space ridge solution."""
    xs = _numpy(source_activations)
    xt = _numpy(target_activations)
    if xs.ndim != 2 or xt.ndim != 2:
        raise TransportError("coordinate-map activations must be rank-2 [samples, features]")
    if xs.shape[0] != xt.shape[0]:
        raise TransportError("source and target activations must use paired samples")
    if ridge <= 0:
        raise TransportError("ridge must be positive")
    gram = xs @ xs.T
    gram.flat[:: gram.shape[0] + 1] += ridge
    dual = np.linalg.solve(gram, xt)
    return CoordinateMap(xs.T @ dual, ridge)


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
    if ws.shape != (pout.shape[0], pin.shape[0]):
        raise TransportError(
            f"shape mismatch: weight={ws.shape}, source_out={pout.shape[0]}, source_in={pin.shape[0]}"
        )
    return pout.T @ ws @ np.linalg.pinv(pin).T
