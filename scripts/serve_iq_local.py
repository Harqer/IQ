#!/usr/bin/env python3
"""OpenAI-compatible local streaming server for the IQ checkpoint.

Wires the full local stack (Tasks 3-11): BucketClient + RunIndex manifest
download, LocalCache, StreamingWeightProvider, CPUHybridForCausalLM, and the
HybridRunner, then serves /health and /v1/chat/completions via the stdlib
ThreadingHTTPServer (no new dependencies).

Non-expert weights (embeddings, norms, attention, mamba, lm_head) are loaded
through the provider at startup; routed experts stream lazily through the
runner's expert_fn at first use (int4 on NVMe, bf16 in RAM LRU). HF_TOKEN is
read from the environment by BucketClient and never printed or persisted.

Requires HF_TOKEN exported (or the bucket already cached on NVMe). Requires
an HF_TOKEN-visible bucket; downloads the manifests at startup.

Example:
    HF_TOKEN=... python scripts/serve_iq_local.py --port 8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from iq_local.bucket import BUCKET_ID, RUN_PREFIX, BucketClient
from iq_local.cache import LocalCache
from iq_local.cpu_hybrid import CPUHybridForCausalLM
from iq_local.index import RunIndex
from iq_local.provider import StreamingWeightProvider
from iq_local.runner import HybridRunner
from iq_local.server import build_handler

from http.server import ThreadingHTTPServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".iq-local-cache")
    parser.add_argument("--cache-budget-gb", type=float, default=512.0)
    parser.add_argument("--max-ram-experts", type=int, default=64)
    parser.add_argument("--bucket-id", default=BUCKET_ID)
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--tokenizer", default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--cpu-int4-base", action="store_true", help="EXPERIMENTAL: meta + CPU INT4 context/shared-MoE loading")
    parser.add_argument("--cpu-int4-group-size", type=int, default=32)
    return parser.parse_args()


def build_stack(args: argparse.Namespace):
    print(
        f"IQ: bucket={args.bucket_id} run={args.run_prefix} cache={args.cache_dir}",
        flush=True,
    )
    client = BucketClient(args.bucket_id, args.run_prefix)
    cache = LocalCache(args.cache_dir, budget_bytes=int(args.cache_budget_gb * (1 << 30)))
    index = RunIndex(client, args.cache_dir / "run")
    provider = StreamingWeightProvider(
        index, cache, client, max_ram_experts=args.max_ram_experts
    )
    config = index.config
    if args.cpu_int4_base:
        from iq_local.int4_loading import build_cpu_int4_model
        model, stats = build_cpu_int4_model(
            config, index, provider, group_size=args.cpu_int4_group_size
        )
        print(f"IQ: CPU INT4 experimental base loaded: {stats}", flush=True)
    else:
        model = CPUHybridForCausalLM(config, dtype=torch.bfloat16, device="cpu")
        global_keys = [key for key in index.weight_map if not key.startswith("layers.")]
        state: dict[str, torch.Tensor] = {}
        for key in global_keys:
            state[key] = provider.load_global(key)
        for layer_idx in range(len(config.schedule.layers)):
            state.update(provider.load_layer_base(layer_idx))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state mismatch: missing={sorted(missing)[:4]}... "
                f"unexpected={sorted(unexpected)[:4]}..."
            )
        print(f"IQ: base weights loaded ({len(state)} tensors)", flush=True)
        model.eval()
    runner = HybridRunner(model, provider)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    expected_special_ids = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007,
        "<|eot_id|>": 128009,
    }
    for token, expected_id in expected_special_ids.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected_id:
            raise RuntimeError(
                f"Tokenizer special ID mismatch: {token}={actual}, "
                f"expected {expected_id}"
            )
    if len(tokenizer) != config.model.vocab_size:
        raise RuntimeError(
            f"IQ checkpoint vocab={config.model.vocab_size}, "
            f"tokenizer length={len(tokenizer)}"
        )
    return runner, tokenizer


def main() -> None:
    args = parse_args()
    runner, tokenizer = build_stack(args)
    handler = build_handler(runner, tokenizer, args.run_prefix)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"IQ: listening on http://{args.host}:{args.port} "
        f"({args.run_prefix}); serving /health and /v1/chat/completions",
        flush=True,
    )
    started = time.time()
    try:
        server.serve_forever()
    finally:
        print(f"IQ: server stopped after {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
