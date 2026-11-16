#!/usr/bin/env python3
"""Run the exact pretrained Mamba-3 MIMO language model inside IQ.

No GLM weight blending, untrained adapters, fake outputs, or paid GPU launch.
Requires an already available BF16-capable CUDA device and HF access to the
Llama-3.1 tokenizer. Baseline is a causal LM, not an instruction-tuned chat model.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from iq_model.mamba3_pretrained import (
    FoundationInferenceError,
    IQMamba3Pretrained,
)

BENCHMARK_PROMPTS = {
    "arithmetic": "2 + 2 =",
    "reasoning": "All robins are birds. All birds are animals. Therefore all robins are",
    "coding": "def square(x):\n    return",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Mamba-3 is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--parity", action="store_true",
                        help="Compare native cached scores against full recomputation for two tokens")
    parser.add_argument("--eval", action="store_true",
                        help="Generate raw arithmetic, reasoning, coding completions")
    parser.add_argument("--preflight", action="store_true",
                        help="Print hardware readiness without downloading or loading weights")
    parser.add_argument("--skip-sha256", action="store_true",
                        help="Explicitly bypass full pinned checkpoint SHA-256 verification")
    parser.add_argument("--parity-atol", type=float, default=0.25)
    parser.add_argument("--parity-rtol", type=float, default=0.025)
    args = parser.parse_args()

    if args.max_new_tokens < 1 or args.max_new_tokens > 128:
        parser.error("--max-new-tokens must be 1..128")
    if args.parity_atol < 0 or args.parity_rtol < 0:
        parser.error("parity tolerances must be nonnegative")

    hardware = {
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    if args.preflight:
        print(json.dumps({"hardware": hardware,
                          "native_pretrained_inference_executed": False}, indent=2))
        return 0

    if not hardware["bf16_supported"]:
        raise FoundationInferenceError(
            "a BF16-capable CUDA GPU is required for the pinned 1.5B native path"
        )

    model = IQMamba3Pretrained.load_official(verify_hash=not args.skip_sha256)
    report = {
        "foundation": "state-spaces/mamba3-mimo-1.5b",
        "source": "original pretrained 24-layer rank-4 MIMO",
        "weights": "strict upstream model load",
        "native_pretrained_inference_executed": True,
        "hardware": hardware,
        "generation": model.generate_text(args.prompt, max_new_tokens=args.max_new_tokens),
    }
    if args.parity:
        report["recurrent_parity"] = model.check_cached_logits(
            args.prompt, atol=args.parity_atol, rtol=args.parity_rtol
        )
    if args.eval:
        report["capability_probes"] = {
            name: model.generate_text(prompt, max_new_tokens=min(args.max_new_tokens, 32))
            for name, prompt in BENCHMARK_PROMPTS.items()
        }
        report["capability_probes_scored"] = False
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FoundationInferenceError, RuntimeError, OSError, ImportError) as exc:
        print(f"IQ_NATIVE_PRETRAINED_INFERENCE_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
