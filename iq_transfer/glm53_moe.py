from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .transport import CoordinateMap, TransportError, transport_linear


class GLM53MoETransformError(RuntimeError):
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
        raise GLM53MoETransformError("MoE tensor contains non-finite values")
    return result


@dataclass(frozen=True)
class ExpertWeights:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


@dataclass(frozen=True)
class GLM53MoETransform:
    source_expert_indices: tuple[int, ...]
    router_weight: np.ndarray
    routing_bias: np.ndarray
    latent_down_weight: np.ndarray
    latent_up_weight: np.ndarray
    routed_experts: tuple[ExpertWeights, ...]
    shared_expert: ExpertWeights


def router_usage_from_topk(
    topk_batches: Iterable[Any],
    *,
    num_experts: int,
) -> np.ndarray:
    if num_experts <= 0:
        raise GLM53MoETransformError("num_experts must be positive")
    counts = np.zeros(num_experts, dtype=np.int64)
    observed = 0
    for batch in topk_batches:
        indices = np.asarray(batch)
        if indices.size == 0:
            continue
        if not np.issubdtype(indices.dtype, np.integer):
            raise GLM53MoETransformError("router top-k indices must be integer typed")
        flat = indices.reshape(-1)
        if flat.min() < 0 or flat.max() >= num_experts:
            raise GLM53MoETransformError("router top-k index is outside expert range")
        counts += np.bincount(flat, minlength=num_experts)
        observed += flat.size
    if observed == 0:
        raise GLM53MoETransformError("router usage capture is empty")
    return counts.astype(np.float64) / float(observed)


def select_experts_by_usage(
    usage: Any,
    *,
    target_experts: int,
) -> tuple[int, ...]:
    scores = _array(usage).reshape(-1)
    if target_experts <= 0 or target_experts > scores.shape[0]:
        raise GLM53MoETransformError(
            "target_experts must be in [1, source_expert_count]"
        )
    if np.any(scores < 0):
        raise GLM53MoETransformError("router usage cannot be negative")
    if float(scores.sum()) <= 0:
        raise GLM53MoETransformError("router usage must contain routed tokens")
    # Primary key: descending usage. Secondary key: stable source expert id.
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    return tuple(int(x) for x in order[:target_experts])


def latent_codec_weights(
    residual_map: CoordinateMap,
    latent_map: CoordinateMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct IQ latent_down/latent_up from two donor-coordinate maps."""
    pres = np.asarray(residual_map.matrix, dtype=np.float64)
    plat = np.asarray(latent_map.matrix, dtype=np.float64)
    if pres.shape[0] != plat.shape[0]:
        raise GLM53MoETransformError(
            "residual and latent maps must share the donor residual width"
        )
    latent_down = plat.T @ np.linalg.pinv(pres).T
    latent_up = pres.T @ np.linalg.pinv(plat).T
    if not np.isfinite(latent_down).all() or not np.isfinite(latent_up).all():
        raise GLM53MoETransformError("latent codec contains non-finite values")
    return latent_down, latent_up


def _transport_expert(
    expert: ExpertWeights,
    *,
    input_map: CoordinateMap,
    intermediate_map: CoordinateMap,
    output_map: CoordinateMap,
) -> ExpertWeights:
    try:
        gate = transport_linear(
            expert.gate,
            input_map,
            intermediate_map,
        )
        up = transport_linear(
            expert.up,
            input_map,
            intermediate_map,
        )
        down = transport_linear(
            expert.down,
            intermediate_map,
            output_map,
        )
    except TransportError as exc:
        raise GLM53MoETransformError(str(exc)) from exc
    return ExpertWeights(gate=gate, up=up, down=down)


def transform_glm53_moe(
    *,
    router_weight: Any,
    routing_bias: Any,
    routed_experts: Iterable[ExpertWeights],
    shared_expert: ExpertWeights,
    expert_usage: Any,
    target_experts: int,
    residual_map: CoordinateMap,
    latent_map: CoordinateMap,
    intermediate_map: CoordinateMap,
) -> GLM53MoETransform:
    """Directly compress GLM routed MoE into IQ Stable LatentMoE weights.

    Expert identity is selected from real donor routing usage. Selected routed
    experts are transported into the donor-derived IQ latent basis; the shared
    expert remains in the full IQ residual basis. This function performs no
    gradient update or teacher/student optimization.
    """
    router = _array(router_weight)
    source_bias = _array(routing_bias).reshape(-1)
    experts = tuple(routed_experts)
    if router.ndim != 2:
        raise GLM53MoETransformError("router weight must be rank-2")
    if len(experts) != router.shape[0]:
        raise GLM53MoETransformError(
            "router row count must equal routed expert count"
        )
    if source_bias.shape != (router.shape[0],):
        raise GLM53MoETransformError(
            "routing_bias must have one entry per donor expert"
        )
    if residual_map.matrix.shape[0] != router.shape[1]:
        raise GLM53MoETransformError(
            "residual map source width must equal router input width"
        )
    selected = select_experts_by_usage(
        expert_usage,
        target_experts=target_experts,
    )
    if max(selected) >= len(experts):
        raise GLM53MoETransformError("expert usage does not match routed experts")

    selection = np.asarray(selected)
    selected_router = router[selection]
    target_router = selected_router @ np.linalg.pinv(
        residual_map.matrix
    ).T
    target_bias = source_bias[selection].copy()
    target_bias -= target_bias.mean()
    latent_down, latent_up = latent_codec_weights(
        residual_map,
        latent_map,
    )

    target_routed = tuple(
        _transport_expert(
            experts[index],
            input_map=latent_map,
            intermediate_map=intermediate_map,
            output_map=latent_map,
        )
        for index in selected
    )
    target_shared = _transport_expert(
        shared_expert,
        input_map=residual_map,
        intermediate_map=intermediate_map,
        output_map=residual_map,
    )
    return GLM53MoETransform(
        source_expert_indices=selected,
        router_weight=target_router,
        routing_bias=target_bias,
        latent_down_weight=latent_down,
        latent_up_weight=latent_up,
        routed_experts=target_routed,
        shared_expert=target_shared,
    )
