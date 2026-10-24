#!/usr/bin/env python3
"""CPU-only audit of IQ checkpoint metadata; no model weights or CUDA required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def audit(config: dict, manifest: dict) -> dict:
    findings: list[dict[str, str]] = []

    def record(level: str, key: str, description: str) -> None:
        findings.append({"level": level, "key": key, "detail": description})

    model = config.get("model", {})
    if model.get("vocab_size") != 128256:
        record("error", "vocabulary", "IQ vocab must match the Mamba-3 donor's 128256 IDs")
    else:
        record("pass", "vocabulary", "Vocab size matches; ID-to-token equivalence remains unverified")

    if model.get("hidden_size") != 4096:
        record("error", "hidden_width", "IQ donor replication expects width 4096")

    layers = config.get("schedule", {}).get("layers", [])
    if not isinstance(layers, list):
        record("error", "schedule", "Schedule must contain a list of layers")
        layers = []
    counts = {n: layers.count(n) for n in ("mamba3", "csa", "hca", "moe")}
    if len(layers) != 80 or counts != {"mamba3": 32, "csa": 12, "hca": 12, "moe": 24}:
        record("error", "schedule", f"Unexpected physical layer count/types: {counts}")
    else:
        record("warning", "function_parity", "80-layer hybrid is not functionally identical to 24-layer Mamba donor; donor MLP blocks are not reproduced intact")

    if manifest.get("method") != "mamba3_foundation_plus_glm53_capability_weight_transfer":
        record("error", "checkpoint_method", "Unexpected transplant metadata method")
    mode = manifest.get("calibration_mode")
    if mode == "weight_only":
        record("blocker", "activation_calibration", "No activation-refined student/teacher alignment recorded")
    elif mode == "activation_refined":
        record("warning", "activation_calibration", "Activation refinement recorded but held-out logits/generation still unverified")
    else:
        record("blocker", "activation_calibration", f"Calibration mode missing/unknown: {mode!r}")

    foundation = manifest.get("foundation", {})
    if foundation.get("repo") != "state-spaces/mamba3-mimo-1.5b":
        record("error", "foundation", "Missing expected Mamba-3 donor identity")
    if manifest.get("recipient_config_fingerprint") is None:
        record("warning", "fingerprint", "Missing target config fingerprint")
    if config.get("attnres") or config.get("reasoning"):
        record("warning", "native_controller", "Recipient-native controller modules require output-activation validation")
    return {
        "calibration_mode": mode,
        "layers": counts,
        "findings": findings,
        "language_quality_confirmed": False,
        "http_200_proves_model_quality": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", help="HF bucket owner/name, such as kofecheckmate/mocha")
    p.add_argument("--config", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.bucket:
        if args.config or args.manifest:
            p.error("use --bucket OR local --config and --manifest")
        try:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
            base = f"hf://buckets/{args.bucket.strip('/')}"
            with fs.open(f"{base}/iq_config.json", "r") as f:
                config = json.load(f)
            with fs.open(f"{base}/transplant_manifest.json", "r") as f:
                manifest = json.load(f)
        except (ImportError, OSError, ValueError) as exc:
            print(f"Could not inspect bucket metadata: {exc}", file=sys.stderr)
            return 2
    else:
        if not (args.config and args.manifest):
            p.error("supply --bucket or both local --config and --manifest")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = audit(config, manifest)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in result["findings"]:
            print(f"{item['level'].upper():7s} {item['key']}: {item['detail']}")
        print("Coherent generation is NOT verified by this metadata audit.")
    return int(any(f["level"] in ("blocker", "error") for f in result["findings"]))


if __name__ == "__main__":
    sys.exit(main())
