from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .transport import CoordinateMap, TransportError, transport_linear


class GLM53DSATransformError(RuntimeError):
    pass


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise GLM53DSATransformError("DSA tensor contains non-finite values")
    return result


def _rotary_last_permutation(head_dim: int, rope_dim: int) -> np.ndarray:
    if head_dim <= 0 or rope_dim <= 0 or rope_dim > head_dim:
        raise GLM53DSATransformError("invalid DSA head/rope dimensions")
    pass_dim = head_dim - rope_dim
    # GLM indexer storage: [rotary, pass-through].
    # IQ partial-RoPE storage: [pass-through, rotary].
    return np.concatenate(
        (
            np.arange(rope_dim, head_dim, dtype=np.int64),
            np.arange(0, rope_dim, dtype=np.int64),
        )
    )


@dataclass(frozen=True)
class GLM53DSATransform:
    q_weight: np.ndarray
    k_weight: np.ndarray
    head_weight: np.ndarray
    k_norm_weight: np.ndarray
    k_norm_bias: np.ndarray


def resolve_indexer_source_layer(
    indexer_types: Sequence[str],
    source_layer: int,
) -> int:
    """Resolve GLM's shared-indexer layer to the preceding full indexer."""
    if source_layer < 0 or source_layer >= len(indexer_types):
        raise GLM53DSATransformError("source_layer is outside indexer_types")
    kind = str(indexer_types[source_layer])
    if kind == "full":
        return source_layer
    if kind != "shared":
        raise GLM53DSATransformError(
            f"unsupported GLM indexer type {kind!r}"
        )
    for layer in range(source_layer - 1, -1, -1):
        if str(indexer_types[layer]) == "full":
            return layer
    raise GLM53DSATransformError(
        "shared GLM indexer has no preceding full indexer"
    )


def transform_glm53_dsa_indexer(
    *,
    q_weight: Any,
    k_weight: Any,
    head_weight: Any,
    k_norm_weight: Any,
    k_norm_bias: Any,
    residual_map: CoordinateMap,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    q_lora_rank: int,
) -> GLM53DSATransform:
    """Refactor GLM direct-token DSA indexer into IQ's RoPE-at-end layout."""
    q = _array(q_weight)
    k = _array(k_weight)
    head = _array(head_weight)
    norm_w = _array(k_norm_weight).reshape(-1)
    norm_b = _array(k_norm_bias).reshape(-1)
    source_hidden, target_hidden = residual_map.matrix.shape
    expected = {
        "q": (num_heads * head_dim, q_lora_rank),
        "k": (head_dim, source_hidden),
        "head": (num_heads, source_hidden),
        "norm_w": (head_dim,),
        "norm_b": (head_dim,),
    }
    actual = {
        "q": q.shape,
        "k": k.shape,
        "head": head.shape,
        "norm_w": norm_w.shape,
        "norm_b": norm_b.shape,
    }
    mismatches = [
        f"{name}: got {actual[name]}, expected {shape}"
        for name, shape in expected.items()
        if actual[name] != shape
    ]
    if mismatches:
        raise GLM53DSATransformError(
            "GLM DSA source shape mismatch: " + "; ".join(mismatches)
        )

    permutation = _rotary_last_permutation(head_dim, rope_dim)
    q_heads = q.reshape(num_heads, head_dim, q_lora_rank)
    target_q = q_heads[:, permutation, :].reshape(
        num_heads * head_dim,
        q_lora_rank,
    )
    reordered_k = k[permutation]
    reordered_norm_w = norm_w[permutation]
    reordered_norm_b = norm_b[permutation]

    identity_out = CoordinateMap(
        np.eye(head_dim, dtype=np.float64),
        ridge=residual_map.ridge,
        source_space="glm53.index_key",
        target_space="iq.index_key",
    )
    head_out = CoordinateMap(
        np.eye(num_heads, dtype=np.float64),
        ridge=residual_map.ridge,
        source_space="glm53.index_heads",
        target_space="iq.index_heads",
    )
    try:
        target_k = transport_linear(
            reordered_k,
            residual_map,
            identity_out,
        )
        target_head = transport_linear(
            head,
            residual_map,
            head_out,
        )
    except TransportError as exc:
        raise GLM53DSATransformError(str(exc)) from exc

    if target_k.shape != (head_dim, target_hidden):
        raise GLM53DSATransformError("transported DSA key shape mismatch")
    if target_head.shape != (num_heads, target_hidden):
        raise GLM53DSATransformError("transported DSA head-weight shape mismatch")
    return GLM53DSATransform(
        q_weight=target_q,
        k_weight=target_k,
        head_weight=target_head,
        k_norm_weight=reordered_norm_w,
        k_norm_bias=reordered_norm_b,
    )
