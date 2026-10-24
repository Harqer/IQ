"""Fail-closed binding of official router-topk IDs to a specific GLM teacher capture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .glm53_routing_replay import GLM53RoutingReplayError, sha_file
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def read_verified_router_pairs(
    *, capture_path: str | Path, routing_path: str | Path,
    source_layer: int, samples_used: int,
) -> tuple[np.ndarray, str]:
    capture_path = Path(capture_path)
    routing_path = Path(routing_path)
    if capture_path.suffix != ".npy" or routing_path.suffix != ".npy":
        raise GLM53RoutingReplayError("capture and routing artifacts must be .npy")

    def parse_json(path: Path) -> dict:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GLM53RoutingReplayError(f"missing or invalid provenance: {path}") from exc
        if not isinstance(record, dict):
            raise GLM53RoutingReplayError("provenance manifest must be an object")
        return record

    capture = parse_json(capture_path.with_suffix(".json"))
    routing = parse_json(routing_path.with_suffix(".json"))
    for field, value in {
        "artifact_type": "glm53_real_pre_moe_hidden_capture",
        "model_repo": REPO,
        "model_revision": REVISION,
        "donor_layer": source_layer,
        "tap": f"layer.{source_layer}.moe_input",
        "tap_position": "after post_attention_layernorm, before sparse MoE",
        "dtype": "float32",
    }.items():
        if capture.get(field) != value:
            raise GLM53RoutingReplayError(f"capture provenance mismatch: {field}")
    if capture.get("sha256") != sha_file(capture_path):
        raise GLM53RoutingReplayError("capture SHA256 mismatch")
    for field, value in {
        "artifact_type": "glm53_real_layer3_router_topk",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_layer": source_layer,
        "router_weight_key": f"model.layers.{source_layer}.mlp.gate.weight",
        "router_bias_key": f"model.layers.{source_layer}.mlp.gate.e_score_correction_bias",
        "capture_sha256": sha_file(capture_path),
        "capture_manifest_sha256": sha_file(capture_path.with_suffix(".json")),
        "official_config_sha256": capture.get("official_config_sha256"),
        "official_index_sha256": capture.get("official_index_sha256"),
        "selection_only": True,
    }.items():
        if routing.get(field) != value:
            raise GLM53RoutingReplayError(f"router provenance mismatch: {field}")
    if routing.get("output_sha256") != sha_file(routing_path):
        raise GLM53RoutingReplayError("router-topk SHA256 mismatch")
    try:
        selected = np.load(routing_path, allow_pickle=False, mmap_mode="r")
    except (ValueError, OSError) as exc:
        raise GLM53RoutingReplayError("router topk tensor unreadable") from exc
    rows = capture.get("num_valid_tokens")
    top_k = routing.get("top_k")
    if (
        type(rows) is not int or rows < 1
        or type(top_k) is not int or not 1 <= top_k <= 256
        or selected.dtype != np.int64
        or selected.shape != (rows, top_k)
        or routing.get("num_tokens") != rows
        or not 1 <= samples_used <= rows
    ):
        raise GLM53RoutingReplayError("router-topk tensor geometry differs from capture")
    values = np.array(selected[:samples_used], dtype=np.int64, copy=True)
    if (
        np.any((values < 0) | (values >= 256))
        or any(np.unique(row).size != top_k for row in values)
    ):
        raise GLM53RoutingReplayError("router-topk contains invalid or duplicate IDs")
    return values, sha_file(routing_path)
