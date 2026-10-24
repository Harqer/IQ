"""Replay the official GLM-5.3 sparse router on saved pre-MoE hidden states.

This performs router *selection* only, never runs any expert FFN.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .glm53_stream_capture import _route_sparse_moe
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class GLM53RoutingReplayError(ValueError):
    pass


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_verified_capture(
    path: Path, *, config_sha256: str, index_sha256: str,
) -> np.ndarray:
    if path.suffix != ".npy" or not path.is_file():
        raise GLM53RoutingReplayError("capture must be an existing .npy file")
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise GLM53RoutingReplayError("real teacher capture provenance sidecar absent")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GLM53RoutingReplayError("invalid capture provenance sidecar") from exc
    required = {
        "artifact_type": "glm53_real_pre_moe_hidden_capture",
        "model_repo": REPO,
        "model_revision": REVISION,
        "donor_layer": 3,
        "tap": "layer.3.moe_input",
        "tap_position": "after post_attention_layernorm, before sparse MoE",
        "official_config_sha256": config_sha256,
        "official_index_sha256": index_sha256,
        "hidden_width": 6144,
        "dtype": "float32",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise GLM53RoutingReplayError(f"teacher capture provenance mismatch: {key}")
    if manifest.get("sha256") != sha_file(path):
        raise GLM53RoutingReplayError("capture content checksum mismatch")
    try:
        values = np.load(path, allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise GLM53RoutingReplayError("unable to parse captured teacher vectors") from exc
    if (
        values.ndim != 2 or values.shape[1] != 6144
        or values.dtype != np.float32 or values.shape[0] < 1
        or manifest.get("num_valid_tokens") != values.shape[0]
        or not np.isfinite(values).all()
    ):
        raise GLM53RoutingReplayError("invalid saved teacher activation geometry")
    return values


@torch.no_grad()
def select_routed_experts(
    hidden: np.ndarray,
    weight: torch.Tensor,
    correction: torch.Tensor,
    config: dict[str, Any],
) -> np.ndarray:
    n_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    n_group = int(config["n_group"])
    topk_group = int(config["topk_group"])
    if (
        hidden.ndim != 2 or hidden.shape[0] < 1
        or tuple(weight.shape) != (n_experts, hidden.shape[1])
        or tuple(correction.shape) != (n_experts,)
        or not 1 <= top_k <= n_experts
        or not 1 <= topk_group <= n_group
        or n_experts < 2 or n_experts % n_group
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(correction).all())
        or not np.isfinite(hidden).all()
    ):
        raise GLM53RoutingReplayError("invalid official router inputs and geometry")
    _, topk = _route_sparse_moe(
        torch.from_numpy(np.ascontiguousarray(hidden)),
        router_weight=weight,
        correction_bias=correction,
        num_group=n_group,
        topk_group=topk_group,
        top_k=top_k,
        normalize=bool(config.get("norm_topk_prob", True)),
        scaling=float(config["routed_scaling_factor"]),
    )
    selected = topk.detach().cpu().numpy().astype(np.int64)
    if (
        selected.shape != (hidden.shape[0], top_k)
        or np.any((selected < 0) | (selected >= n_experts))
        or any(len(set(row)) != top_k for row in selected.tolist())
    ):
        raise GLM53RoutingReplayError("router produced invalid expert IDs")
    return selected
