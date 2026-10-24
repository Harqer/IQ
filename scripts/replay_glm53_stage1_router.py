#!/usr/bin/env python3
"""Replay pinned GLM-5.3 layer-3 router decisions for saved real activations.

Fetches the official router gate and correction bias only (~3.2 MB), never
runs routed experts. New .npy IDs and a hashed .json manifest are produced.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from iq_transfer.glm53_routing_replay import (
    GLM53RoutingReplayError, read_verified_capture,
    select_routed_experts, sha_file,
)
from iq_transfer.http_range_source import HubSafetensorsSource
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="new route IDs .npy with shape [tokens,8]")
    p.add_argument("--experts", type=int, nargs="+", default=[77, 87, 246, 131])
    args = p.parse_args()
    sidecar = args.output.with_suffix(".json")
    if args.output.suffix != ".npy" or args.output.exists() or sidecar.exists():
        p.error("output must be a new .npy path without an existing sidecar")
    if not args.experts or len(set(args.experts)) != len(args.experts):
        p.error("expert IDs must be a nonempty unique sequence")
    try:
        config_file = Path(hf_hub_download(REPO, "config.json", revision=REVISION))
        index_file = Path(hf_hub_download(
            REPO, "model.safetensors.index.json", revision=REVISION
        ))
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if (
            config.get("model_type") != "glm_moe_dsa"
            or config.get("hidden_size") != 6144
            or config.get("n_routed_experts") != 256
            or config.get("num_experts_per_tok") != 8
            or config.get("first_k_dense_replace") != 3
        ):
            raise GLM53RoutingReplayError("official model configuration mismatch")
        if any(not 0 <= i < 256 for i in args.experts):
            raise GLM53RoutingReplayError("expert ID outside official routing range")
        capture = read_verified_capture(
            args.activations, config_sha256=sha_file(config_file),
            index_sha256=sha_file(index_file),
        )
        with tempfile.TemporaryDirectory(prefix="glm53-router-index-") as temp:
            root = Path(temp)
            (root / "model.safetensors.index.json").write_bytes(index_file.read_bytes())
            source = HubSafetensorsSource(
                root, repo_id=REPO, revision=REVISION,
                max_tensor_bytes=16 * 1024 * 1024,
                max_request_bytes=4 * 1024 * 1024,
            )
            try:
                prefix = "model.layers.3.mlp.gate."
                weight = source.get(prefix + "weight")
                bias = source.get(prefix + "e_score_correction_bias")
            finally:
                source.close()
        routed = select_routed_experts(capture, weight, bias, config)
        counts = {
            str(i): int((routed == i).any(axis=1).sum())
            for i in args.experts
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as out:
            np.save(out, routed, allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "artifact_type": "glm53_real_layer3_router_topk",
            "not_an_iq_checkpoint": True,
            "source_repo": REPO,
            "source_revision": REVISION,
            "source_layer": 3,
            "routing": "topk sigmoid + official correction bias and group selection",
            "router_weight_key": "model.layers.3.mlp.gate.weight",
            "router_bias_key": "model.layers.3.mlp.gate.e_score_correction_bias",
            "capture_sha256": sha_file(args.activations),
            "capture_manifest_sha256": sha_file(args.activations.with_suffix(".json")),
            "official_config_sha256": sha_file(config_file),
            "official_index_sha256": sha_file(index_file),
            "output_sha256": sha_file(args.output),
            "num_tokens": int(routed.shape[0]),
            "top_k": int(routed.shape[1]),
            "selected_counts": counts,
            "selection_only": True,
        }
        with sidecar.open("x", encoding="utf-8") as out:
            json.dump(metadata, out, sort_keys=True, indent=2)
            out.write("\n")
        print(json.dumps({
            "status": "official_router_replayed_on_real_pre_moe_states",
            "tokens": len(routed),
            "top_k": int(routed.shape[1]),
            "selected_counts": counts,
            "missing_experts": [int(i) for i in args.experts if counts[str(i)] == 0],
            "output": str(args.output.resolve()),
            "selection_only": True,
        }, indent=2))
    except (GLM53RoutingReplayError, OSError, KeyError, ValueError, RuntimeError) as exc:
        p.exit(2, f"GLM-5.3 routing replay failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
