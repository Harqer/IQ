#!/usr/bin/env python3
"""Offline GLM-5.3 / Mamba-3 MIMO source-reference contract.

Reads JSON metadata ONLY; no model downloads, torch, CUDA, HF Jobs, or inference.
Passing proves source-config consistency, NOT pretrained tensor/content parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GLM_SOURCE = "zai-org/GLM-5.3-BF16"
MAMBA_SOURCE = "state-spaces/mamba3-mimo-1.5b"
PINNED_GLM_REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
PINNED_MAMBA_REVISION = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"


def audit_sources(glm: dict, mamba: dict, *, manifest: dict | None = None,
                  compile_status: dict | None = None,
                  lexical_intent: str = "glm53") -> dict:
    if lexical_intent not in {"glm53", "mamba3"}:
        raise ValueError("lexical_intent must be glm53 or mamba3")
    findings: list[dict] = []

    def require(name: str, actual, expected) -> None:
        if actual != expected:
            findings.append({
                "severity": "blocker", "key": name, "actual": actual,
                "expected": expected,
            })

    # Official GLM-5.3 config invariants. These are source checkpoints,
    # not inferred from the differently shaped compiled IQ student.
    for key, value in (
        ("model_type", "glm_moe_dsa"),
        ("vocab_size", 154880),
        ("hidden_size", 6144),
        ("num_hidden_layers", 78),
        ("num_attention_heads", 64),
        ("q_lora_rank", 2048),
        ("kv_lora_rank", 512),
        ("qk_nope_head_dim", 192),
        ("qk_rope_head_dim", 64),
        ("qk_head_dim", 256),
        ("v_head_dim", 256),
        ("moe_intermediate_size", 2048),
        ("intermediate_size", 12288),
        ("first_k_dense_replace", 3),
        ("n_routed_experts", 256),
        ("n_shared_experts", 1),
        ("num_experts_per_tok", 8),
        ("hidden_act", "silu"),
        ("scoring_func", "sigmoid"),
        ("norm_topk_prob", True),
        ("routed_scaling_factor", 2.5),
        ("rms_norm_eps", 1e-5),
        ("rope_interleave", True),
        ("tie_word_embeddings", False),
    ):
        require("glm." + key, glm.get(key), value)

    require("glm.qk_decomposition",
            glm.get("qk_nope_head_dim", -100) + glm.get("qk_rope_head_dim", -100),
            glm.get("qk_head_dim"))
    require("glm.eos_token_ids", glm.get("eos_token_id"),
            [154820, 154827, 154829])
    kinds = glm.get("mlp_layer_types")
    if kinds is not None:
        require("glm.78_layer_mlp_types", len(kinds), 78)
        require("glm.first_three_dense", kinds[:3], ["dense"] * 3)
        require("glm.last_75_sparse", kinds[3:], ["sparse"] * 75)

    # Mamba donor's 24 BLOCKS all contain two distinct sublayers.
    # Calling a recurrent mixer alone a "full Mamba-3 MIMO block" is false.
    for key, value in (
        ("vocab_size", 128256),
        ("d_model", 2048),
        ("n_layer", 24),
        ("d_intermediate", 3824),
        ("attn_layer_idx", []),
        ("tie_embeddings", True),
    ):
        require("mamba." + key, mamba.get(key), value)
    ssm = mamba.get("ssm_cfg")
    if not isinstance(ssm, dict):
        findings.append({"severity": "blocker", "key": "mamba.ssm_cfg",
                         "actual": ssm, "expected": "dict"})
        ssm = {}
    for key, value in (
        ("layer", "Mamba3"),
        ("d_state", 128),
        ("expand", 2),
        ("headdim", 64),
        ("is_mimo", True),
        ("mimo_rank", 4),
        ("chunk_size", 16),
        ("rope_fraction", 0.5),
    ):
        require("mamba.ssm." + key, ssm.get(key), value)

    if manifest is not None:
        foundation = manifest.get("foundation") or {}
        capability = manifest.get("capability_source") or {}
        require("checkpoint.foundation_repo", foundation.get("repo"), MAMBA_SOURCE)
        require("checkpoint.foundation_revision", foundation.get("revision"),
                PINNED_MAMBA_REVISION)
        require("checkpoint.glm_capability_repo", capability.get("repo"), GLM_SOURCE)
        require("checkpoint.glm_capability_revision", capability.get("revision"),
                PINNED_GLM_REVISION)
        if lexical_intent == "glm53":
            # Source provenance is Mamba lexical: a mismatch, not a new
            # GLM-based checkpoint created by editing the manifest.
            findings.append({
                "severity": "blocker", "key": "checkpoint.lexical_owner",
                "actual": "mamba3", "expected": "glm53",
            })
        if manifest.get("calibration_mode") != "activation_refined":
            findings.append({
                "severity": "blocker", "key": "checkpoint.calibration",
                "actual": manifest.get("calibration_mode"),
                "expected": "activation_refined",
            })

    if compile_status is not None:
        for key in ("inference_ready", "end_to_end_knowledge_retention_proven"):
            if compile_status.get(key) is not True:
                findings.append({
                    "severity": "blocker", "key": "checkpoint." + key,
                    "actual": compile_status.get(key), "expected": True,
                })

    return {
        "status": "blocked" if findings else "reference_config_consistent",
        "checks": {"glm": "GLM-5.3-BF16", "mamba": "Mamba-3-MIMO-1.5B",
                   "mamba_complete_blocks": 24,
                   "per_block": ["Mamba-3 MIMO mixer", "gated MLP"]},
        "findings": findings,
        "note": "Passing this metadata audit DOES NOT prove tokenizer ID mapping, "
                "safetensors header integrity, Mamba block weight coverage, "
                "MLP numerical behavior, teacher parity or inference readiness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-config", type=Path, required=True)
    parser.add_argument("--mamba-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional read-only legacy all256 directory")
    parser.add_argument("--intent", choices=("glm53", "mamba3"), default="glm53")
    args = parser.parse_args()
    try:
        glm = json.loads(args.glm_config.read_text(encoding="utf-8"))
        mamba = json.loads(args.mamba_config.read_text(encoding="utf-8"))
        manifest = status = None
        if args.checkpoint is not None:
            manifest = json.loads((args.checkpoint / "transplant_manifest.json").read_text(encoding="utf-8"))
            status = json.loads((args.checkpoint / "COMPILE_COMPLETE.json").read_text(encoding="utf-8"))
        report = audit_sources(glm, mamba, manifest=manifest,
                               compile_status=status, lexical_intent=args.intent)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"status": "blocked", "findings": [{"severity": "blocker",
                  "key": "unreadable_inputs", "detail": str(exc)}]}
    print(json.dumps(report, sort_keys=True, indent=2))
    return int(report["status"] == "blocked")


if __name__ == "__main__":
    raise SystemExit(main())
