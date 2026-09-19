from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .transport import CoordinateMap


class GQATransportError(RuntimeError):
    pass


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise GQATransportError("GQA activation tensor contains non-finite values")
    return array


def _fix_svd_signs(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, copy=True)
    for col in range(result.shape[1]):
        column = result[:, col]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            result[:, col] *= -1.0
    return result


@dataclass(frozen=True)
class GQAProjection:
    group_map: np.ndarray
    q_map: CoordinateMap
    kv_map: CoordinateMap
    source_q_heads: int
    source_kv_heads: int
    target_q_heads: int
    target_kv_heads: int
    head_dim: int
    explained_variance_ratio: float

    def __post_init__(self) -> None:
        expected_group = (self.source_kv_heads, self.target_kv_heads)
        if self.group_map.shape != expected_group:
            raise GQATransportError(
                f"group map shape mismatch: got {self.group_map.shape}, expected {expected_group}"
            )
        gram = self.group_map.T @ self.group_map
        if not np.allclose(gram, np.eye(self.target_kv_heads), atol=1e-8, rtol=1e-8):
            raise GQATransportError("GQA group map must have orthonormal columns")
        if not (0.0 <= self.explained_variance_ratio <= 1.0 + 1e-12):
            raise GQATransportError("explained_variance_ratio must be in [0, 1]")


def fit_gqa_group_projection(
    source_key_activations: Any,
    *,
    source_q_heads: int,
    source_kv_heads: int,
    target_q_heads: int,
    target_kv_heads: int,
    head_dim: int,
    source_space_prefix: str = "donor",
    target_space_prefix: str = "iq",
) -> GQAProjection:
    """Fit a structure-preserving GQA group projection from donor K activations.

    The reduction operates across KV groups only and preserves the within-group query
    offset and head feature basis. This avoids arbitrary full-width Q/K coordinate maps
    that can destroy QK dot-product geometry.

    Source K activations are expected as [samples, source_kv_heads * head_dim]. The
    highest-variance orthonormal KV-group subspace is retained using SVD. The same
    group projection is expanded across query heads, so donor and recipient preserve
    the same number of query heads per KV group.
    """
    ints = {
        "source_q_heads": source_q_heads,
        "source_kv_heads": source_kv_heads,
        "target_q_heads": target_q_heads,
        "target_kv_heads": target_kv_heads,
        "head_dim": head_dim,
    }
    bad = [name for name, value in ints.items() if int(value) <= 0]
    if bad:
        raise GQATransportError(f"positive GQA dimensions required: {', '.join(bad)}")
    if source_q_heads % source_kv_heads != 0 or target_q_heads % target_kv_heads != 0:
        raise GQATransportError("Q heads must be divisible by KV heads")
    source_repeat = source_q_heads // source_kv_heads
    target_repeat = target_q_heads // target_kv_heads
    if source_repeat != target_repeat:
        raise GQATransportError(
            "structure-preserving GQA transport requires equal query-heads-per-KV-group; "
            f"got source={source_repeat}, target={target_repeat}"
        )
    if target_kv_heads > source_kv_heads:
        raise GQATransportError("PCA group projection currently supports KV-head compression only")

    x = _numpy(source_key_activations)
    if x.ndim != 2:
        raise GQATransportError("source_key_activations must be rank-2 [samples, kv_heads * head_dim]")
    expected_features = source_kv_heads * head_dim
    if x.shape[1] != expected_features:
        raise GQATransportError(
            f"source K activation width mismatch: got {x.shape[1]}, expected {expected_features}"
        )
    if x.shape[0] < 2:
        raise GQATransportError("at least two activation samples are required")

    grouped = x.reshape(x.shape[0], source_kv_heads, head_dim)
    group_observations = grouped.transpose(0, 2, 1).reshape(-1, source_kv_heads)
    group_observations = group_observations - group_observations.mean(axis=0, keepdims=True)
    if np.linalg.norm(group_observations) <= 0:
        raise GQATransportError("source K activations have zero centered group variance")

    _, singular, vt = np.linalg.svd(group_observations, full_matrices=False)
    if vt.shape[0] < target_kv_heads:
        raise GQATransportError("not enough rank to construct requested KV-group projection")
    group_map = _fix_svd_signs(vt[:target_kv_heads].T)

    total_variance = float(np.sum(singular * singular))
    kept_variance = float(np.sum(singular[:target_kv_heads] ** 2))
    explained = kept_variance / total_variance if total_variance > 0 else 0.0

    kv_matrix = np.kron(group_map, np.eye(head_dim, dtype=np.float64))
    query_group_map = np.kron(group_map, np.eye(source_repeat, dtype=np.float64))
    q_matrix = np.kron(query_group_map, np.eye(head_dim, dtype=np.float64))

    ridge_tag = float(np.finfo(np.float64).tiny)
    q_map = CoordinateMap(
        q_matrix,
        ridge_tag,
        f"{source_space_prefix}.q",
        f"{target_space_prefix}.q",
    )
    kv_map = CoordinateMap(
        kv_matrix,
        ridge_tag,
        f"{source_space_prefix}.kv",
        f"{target_space_prefix}.kv",
    )
    return GQAProjection(
        group_map=group_map,
        q_map=q_map,
        kv_map=kv_map,
        source_q_heads=source_q_heads,
        source_kv_heads=source_kv_heads,
        target_q_heads=target_q_heads,
        target_kv_heads=target_kv_heads,
        head_dim=head_dim,
        explained_variance_ratio=explained,
    )
