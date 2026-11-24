#!/usr/bin/env python3
"""Opt-in, CPU-only Phase 5e Mamba-3 CSA/HCA inference server.

This does NOT replace the 235-layer IQ local runtime. CPU recomputes the full
prefix for every generated token. The pretrained MIMO recurrence remains an
uncertified reference implementation; do not expose this server publicly.
"""
from __future__ import annotations

import argparse
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iq_local.mamba3_csa_hca import Mamba3CSAHCARunner, PHASE5E_SHA256
from iq_local.server import build_handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attention-checkpoint", type=Path, default=ROOT / "checkpoints" /
                   "mamba3-csa-hca" / "iq_phase5e_attention_candidate.pt")
    p.add_argument("--foundation-checkpoint", type=Path, default=None,
                   help="Local original pytorch_model.bin; otherwise fetch via HF Hub")
    p.add_argument("--attention-sha256", default=PHASE5E_SHA256,
                   help="Expected Phase 5e checkpoint SHA-256 (verified before loading)")
    p.add_argument("--mode", choices=("hybrid", "baseline"), default="hybrid")
    p.add_argument("--tokenizer", default="unsloth/Meta-Llama-3.1-8B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--probe", action="store_true", help="Run baseline/active logit check, then exit")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    runner = Mamba3CSAHCARunner.from_checkpoints(
        args.attention_checkpoint, foundation_path=args.foundation_checkpoint,
        expected_sha256=args.attention_sha256, mode=args.mode)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    expected = {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
                "<|eot_id|>": 128009}
    for token, token_id in expected.items():
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            raise RuntimeError(f"Tokenizer mismatch for {token}: expected {token_id}")
    if len(tokenizer) != runner.vocab_size:
        raise RuntimeError("Tokenizer vocabulary differs from original Mamba-3")
    print(f"IQ Mamba-3 experiment loaded: mode={args.mode}, "
          f"layers=24, vocab={runner.vocab_size}", flush=True)
    if args.probe:
        ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].reshape(-1).tolist()
        base = runner.next_logits(ids, baseline=True)
        hybrid = runner.next_logits(ids, baseline=False)
        print({"baseline_next": int(base.argmax()), "hybrid_next": int(hybrid.argmax()),
               "max_abs_logit_delta": float((base - hybrid).abs().max())}, flush=True)
        if not 1 <= args.max_new_tokens <= 16:
            p.error("--max-new-tokens must be 1..16")
        generated = runner.generate(ids, args.max_new_tokens)
        print({"mode": args.mode, "text": tokenizer.decode(generated, skip_special_tokens=True)}, flush=True)
        return
    handler = build_handler(runner, tokenizer, f"iq-mamba3-csa-hca-phase5e-{args.mode}")
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"IQ experimental inference listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
