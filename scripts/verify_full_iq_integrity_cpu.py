#!/usr/bin/env python3
"""Verify the entire published IQ/mocha checkpoint using CPU I/O only.

No torch, CUDA, model setup or network calls. Works in system Python before
dependency installation, given a local read-only mount of the full 82 shards.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Independent source pins: recorded from the published HF bucket.
PINNED_INDEX_SHA256 = "0154a99903ea86daea357f855ee24ae10e430e2adda022f2a6843476eaaa6128"
PINNED_SHA256SUMS_SHA256 = "705b9127a8575acc5aba9e48dafa58ddfbb6d7616917711bffcd4f52478e25e2"


def _verifier():
    path = Path(__file__).resolve().parents[1] / "iq_transfer" / "checkpoint_integrity.py"
    spec = importlib.util.spec_from_file_location("_iq_cpu_integrity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("CPU integrity module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    os.environ["IQ_EXPECTED_INDEX_SHA256"] = PINNED_INDEX_SHA256
    os.environ["IQ_EXPECTED_SHA256SUMS_SHA256"] = PINNED_SHA256SUMS_SHA256
    try:
        verified = _verifier().verify_iq_checkpoint_integrity(args.checkpoint)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}), file=sys.stderr)
        return 2
    if verified.shard_count != 82 or verified.shard_bytes != 33846255884:
        print(json.dumps({"status": "FAILED", "reason": "Expected full mocha checkpoint, not a subset"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "VERIFIED", "checkpoint": str(args.checkpoint),
                      "shards": verified.shard_count, "tensors": verified.tensor_count,
                      "verified_sha256_count": verified.verified_sha256_count,
                      "verified_header_count": verified.verified_header_count,
                      "bytes": verified.shard_bytes, "index_sha256": verified.index_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
