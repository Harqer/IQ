from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np


class ShadowError(RuntimeError):
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
class MeasurementPlan:
    sample_count: int
    measurements: int = 256
    seed: int = 0

    def matrix(self) -> np.ndarray:
        if self.sample_count <= 0 or self.measurements <= 0:
            raise ShadowError("sample_count and measurements must be positive")
        rng = np.random.default_rng(self.seed)
        signs = rng.integers(0, 2, size=(self.measurements, self.sample_count), dtype=np.int8)
        return (2.0 * signs.astype(np.float64) - 1.0) / np.sqrt(self.sample_count)


@dataclass(frozen=True)
class FunctionalShadow:
    layer: int
    observables: np.ndarray
    sample_count: int
    feature_count: int


def extract_shadow(activations: Any, plan: MeasurementPlan, *, layer: int) -> FunctionalShadow:
    x = _numpy(activations)
    if x.ndim < 2:
        raise ShadowError("activations must have shape [samples, ..., features]")
    x = x.reshape(-1, x.shape[-1])
    if x.shape[0] != plan.sample_count:
        raise ShadowError(f"expected {plan.sample_count} samples, got {x.shape[0]}")
    centered = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered)
    if not np.isfinite(norm) or norm <= 0:
        raise ShadowError("activation matrix has zero or invalid centered norm")
    normalized = centered / norm
    projected = plan.matrix() @ normalized
    observables = np.einsum("ij,ij->i", projected, projected)
    return FunctionalShadow(layer, observables, x.shape[0], x.shape[1])


def shadow_distance(a: FunctionalShadow, b: FunctionalShadow) -> float:
    if a.observables.shape != b.observables.shape:
        raise ShadowError("shadows must use the same measurement plan")
    diff = a.observables - b.observables
    scale = np.linalg.norm(a.observables) * np.linalg.norm(b.observables)
    cosine_penalty = 1.0 if scale == 0 else 1.0 - float(np.dot(a.observables, b.observables) / scale)
    mse = float(np.mean(diff * diff))
    return mse + cosine_penalty


def match_layers_monotonic(
    source: Mapping[int, FunctionalShadow],
    target: Mapping[int, FunctionalShadow],
    *,
    depth_prior: float = 0.05,
) -> dict[int, int]:
    """Unique order-preserving target->source layer assignment.

    Intended for compression experiments where donor layers >= target layers.
    """
    s_layers = sorted(source)
    t_layers = sorted(target)
    if len(s_layers) < len(t_layers):
        raise ShadowError("monotonic matcher requires at least as many source layers as target layers")
    ns, nt = len(s_layers), len(t_layers)
    costs = np.empty((nt, ns), dtype=np.float64)
    for ti, t in enumerate(t_layers):
        t_depth = ti / max(1, nt - 1)
        for si, s in enumerate(s_layers):
            s_depth = si / max(1, ns - 1)
            costs[ti, si] = shadow_distance(target[t], source[s]) + depth_prior * abs(t_depth - s_depth)

    inf = float("inf")
    dp = np.full((nt, ns), inf, dtype=np.float64)
    parent = np.full((nt, ns), -1, dtype=np.int64)
    dp[0, : ns - nt + 1] = costs[0, : ns - nt + 1]

    for ti in range(1, nt):
        min_si = ti
        max_si = ns - (nt - ti)
        for si in range(min_si, max_si + 1):
            previous = dp[ti - 1, :si]
            best_prev = int(np.argmin(previous))
            best = previous[best_prev]
            if np.isfinite(best):
                dp[ti, si] = best + costs[ti, si]
                parent[ti, si] = best_prev

    end = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, end]):
        raise ShadowError("no valid monotonic layer assignment")
    chosen = [end]
    for ti in range(nt - 1, 0, -1):
        chosen.append(int(parent[ti, chosen[-1]]))
    chosen.reverse()
    return {t_layers[i]: s_layers[chosen[i]] for i in range(nt)}
