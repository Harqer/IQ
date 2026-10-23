#!/usr/bin/env python3
"""Logical CSA/HCA BF16 KV-cache estimator from an IQ checkpoint config.

Does not include model weights, Mamba states, activation/scratch tensors,
allocator capacity slack, or CUDA overhead. No GPUs or torch required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(config: dict, tokens: int, bytes_per_element: int = 2) -> dict:
    c = config["compressed_context"]
    schedule = config["schedule"]["layers"]
    width = c["head_dim"]  # Shared KV: NOT num_attention_heads * head_dim
    iw = c["index_head_dim"]
    window = c["sliding_window"]
    csa_rate = c["csa_compress_rate"]
    hca_rate = c["hca_compress_rate"]
    direct = c["direct_token_indexer"]

    # Local window is already included in full token KV when direct=True.
    local_bytes = min(tokens, window) * width * bytes_per_element
    if direct:
        # Direct indexer scores every past token, and returns raw token KVs.
        csa_bytes = tokens * (iw + width) * bytes_per_element
    else:
        csa_bytes = (
            (tokens // csa_rate) * (iw + width) * bytes_per_element
            + local_bytes
        )
    hca_bytes = (tokens // hca_rate) * width * bytes_per_element + local_bytes
    csa_layers = schedule.count("csa")
    hca_layers = schedule.count("hca")
    return {
        "tokens": tokens,
        "bytes_per_element": bytes_per_element,
        "csa_layers": csa_layers,
        "hca_layers": hca_layers,
        "csa_per_layer_mib": csa_bytes / 2**20,
        "hca_per_layer_mib": hca_bytes / 2**20,
        "total_mib": (csa_bytes * csa_layers + hca_bytes * hca_layers) / 2**20,
        "direct_token_indexer": direct,
        "note": "logical cache payload only; allocated capacity can be larger",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="path to checkpoint iq_config.json")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    args = parser.parse_args()
    conf = json.loads(args.config.read_text(encoding="utf-8"))
    tokens = args.tokens if args.tokens is not None else conf["model"]["max_position_embeddings"]
    if tokens < 1 or args.bytes_per_element < 1:
        parser.error("tokens and bytes-per-element must be positive")
    print(json.dumps(estimate(conf, tokens, args.bytes_per_element), indent=2))


if __name__ == "__main__":
    main()
