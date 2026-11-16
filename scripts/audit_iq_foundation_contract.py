#!/usr/bin/env python3
"""Offline IQ foundation/lexical provenance check. No GPU, network, or weight reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GLM = "zai-org/GLM-5.3-BF16"
MAMBA = "state-spaces/mamba3-mimo-1.5b"
VOCABS = {"glm53": 154880, "mamba3": 128256}


def read_json(root: Path, filename: str) -> dict | None:
    path = root / filename
    if not path.is_file():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{filename} must be a JSON object")
    return obj


def audit(root: Path, intent: str = "glm53") -> dict:
    if intent not in VOCABS:
        raise ValueError(f"unsupported foundation intent: {intent}")
    findings: list[dict] = []

    def flag(level: str, code: str, detail: str) -> None:
        findings.append({"level": level, "code": code, "detail": detail})

    try:
        config = read_json(root, "iq_config.json")
        manifest = read_json(root, "transplant_manifest.json")
        index = read_json(root, "model.safetensors.index.json")
        compile_status = read_json(root, "COMPILE_COMPLETE.json")
    except (OSError, ValueError) as exc:
        return {"status": "blocked", "intent": intent, "findings": [{
            "level": "blocker", "code": "unreadable_metadata", "detail": str(exc)
        }]}

    if not config or not manifest or not index:
        flag("blocker", "metadata_missing", "Expected iq_config.json, transplant_manifest.json and model.safetensors.index.json")
        return {"status": "blocked", "intent": intent, "findings": findings}

    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}
    vocab = model_config.get("vocab_size")
    width = model_config.get("hidden_size")
    method = manifest.get("method")
    foundation = manifest.get("foundation", {})
    if not isinstance(foundation, dict):
        foundation = {}
    source = foundation.get("repo")

    # For the old compiler, provenance unambiguously assigns lexical weights
    # to Mamba; never infer GLM lexical provenance from capability_source.
    lexical = manifest.get("lexical_source")
    if lexical is None and method == "mamba3_foundation_plus_glm53_capability_weight_transfer":
        lexical = MAMBA

    if vocab != VOCABS[intent]:
        flag("blocker", "vocab_mismatch", f"IQ vocab={vocab}; {intent} expects {VOCABS[intent]}")
    expected_repo = GLM if intent == "glm53" else MAMBA
    if lexical != expected_repo:
        flag("blocker", "wrong_lexical_foundation", f"Lexical source={lexical!r}, expected {expected_repo}; tokenizer-only swap is unsafe")
    if intent == "glm53" and method == "mamba3_foundation_plus_glm53_capability_weight_transfer":
        flag("blocker", "wrong_transfer_direction", "Compiler explicitly writes Mamba-derived embeddings, final norm and LM head")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        flag("blocker", "missing_weight_map", "No safetensors weight_map in index")
        weight_map = {}
    for key in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
        if key not in weight_map:
            flag("blocker", "lexical_tensor_missing", f"Missing index entry: {key}")
    if (root / "tokenizer.json").is_file():
        flag("info", "tokenizer_present", "Local tokenizer exists, but full ID and special-token equivalence is unverified")
    else:
        flag("warning", "tokenizer_unbundled", "No tokenizer.json; separately mounted tokenizer requires exact ID-map verification")

    if manifest.get("calibration_mode") != "activation_refined":
        flag("blocker", "unvalidated_calibration", f"Calibration is {manifest.get('calibration_mode')!r}, not activation_refined")
    if compile_status is None:
        flag("blocker", "compile_status_missing", "Compilation status absent; inference-readiness not established")
    else:
        if compile_status.get("inference_ready") is not True:
            flag("blocker", "inference_not_ready", "Compile marker does not mark inference_ready=true")
        if compile_status.get("end_to_end_knowledge_retention_proven") is not True:
            flag("blocker", "unproven_capability_retention", "Compile marker does not prove end-to-end capability retention")

    return {
        "status": "blocked" if any(f["level"] == "blocker" for f in findings) else "metadata_consistent_not_quality_verified",
        "intent": intent,
        "checkpoint": str(root),
        "observed": {
            "vocab_size": vocab,
            "hidden_size": width,
            "transfer_method": method,
            "foundation_repo": source,
            "lexical_source": lexical,
            "indexed_weight_bytes": index.get("metadata", {}).get("total_size") if isinstance(index.get("metadata"), dict) else None,
            "indexed_tensors": len(weight_map),
            "indexed_shards": len(set(weight_map.values())) if all(isinstance(x, str) for x in weight_map.values()) else None,
        },
        "findings": findings,
        "note": "Metadata-only provenance check. Passing does NOT prove tensor correctness, functional equivalence, coherent language or inference readiness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local or read-only-mounted checkpoint directory")
    parser.add_argument("--intent", choices=tuple(VOCABS), default="glm53")
    args = parser.parse_args()
    result = audit(args.checkpoint, args.intent)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
