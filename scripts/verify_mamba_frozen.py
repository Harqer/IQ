#!/usr/bin/env python3
"""CPU-only bitwise Mamba-3 foundation preservation check across IQ checkpoints.

Hashes every canonical model-mamba-*.safetensors shard and model-global.
This compares the serialized pretrained recurrent/lexical weights, not
the runtime behavior of the hybrid residual stream.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


class MambaPreservationError(ValueError):
    pass


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_mamba_preservation(original: Path, candidate: Path) -> dict:
    initial = json.loads((original / "iq_config.json").read_text(encoding="utf-8"))
    target = json.loads((candidate / "iq_config.json").read_text(encoding="utf-8"))
    if initial.get("mamba3") != target.get("mamba3"):
        raise MambaPreservationError("Mamba runtime/config changed")
    if initial.get("schedule") != target.get("schedule"):
        raise MambaPreservationError("Mamba physical placement changed")
    if initial.get("model", {}).get("vocab_size") != target.get("model", {}).get("vocab_size"):
        raise MambaPreservationError("lexical vocabulary changed")
    if initial.get("model", {}).get("hidden_size") != target.get("model", {}).get("hidden_size"):
        raise MambaPreservationError("residual/embedding width changed")
    expected = {"model-global.safetensors"}
    expected |= {f"model-mamba-{i:03d}.safetensors" for i in range(32)}
    all_results = []
    for base in (original, candidate):
        indexed = json.loads(
            (base / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        files = set(indexed.get("weight_map", {}).values())
        if not expected <= files:
            raise MambaPreservationError(
                f"{base}: checkpoint is missing global/Mamba foundation shards"
            )
        globals_keys = {
            key for key, file in indexed["weight_map"].items()
            if file == "model-global.safetensors"
        }
        if globals_keys != {"embed_tokens.weight", "norm.weight", "lm_head.weight"}:
            raise MambaPreservationError(f"{base}: lexical/global tensor keys differ")
        for name in expected:
            if not (base / name).is_file():
                raise MambaPreservationError(f"missing foundation shard: {base / name}")
    for name in sorted(expected):
        left = digest(original / name)
        right = digest(candidate / name)
        if left != right:
            raise MambaPreservationError(f"foundation weights changed: {name}")
        all_results.append({"shard": name, "sha256": left})
    return {
        "bitwise_mamba_and_lexical_weights_identical": True,
        "foundation_shards": len(all_results),
        "shard_sha256": all_results,
        "runtime_behavior_preserved": False,
        "note": "Identical weights do not prove equal activations when GLM layers change.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    args = p.parse_args()
    try:
        report = verify_mamba_preservation(args.original, args.candidate)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Mamba preservation FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
