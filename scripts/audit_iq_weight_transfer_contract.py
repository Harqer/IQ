#!/usr/bin/env python3
"""Read-only GLM-5.3 -> IQ weight-transfer invariant audit (JSON metadata only).

This checks source/recipient architecture and checkpoint provenance. It never
loads tensors, imports CUDA, calls an inference endpoint, or allocates GPUs.
It cannot prove tensor-layout or numerical parity without paired computations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(donor: dict, recipient: dict, manifest: dict) -> dict:
    findings: list[dict] = []

    def record(category: str, status: str, name: str, actual, expected, reason: str) -> None:
        findings.append({
            "category": category, "status": status, "name": name,
            "actual": actual, "expected": expected, "reason": reason,
        })

    model = recipient.get("model") or {}
    context = recipient.get("compressed_context") or {}
    moe = recipient.get("moe") or {}
    stable = recipient.get("stable_moe") or {}
    schedule = (recipient.get("schedule") or {}).get("layers") or []
    method = manifest.get("method")

    def compare(category: str, name: str, actual, expected, reason: str) -> None:
        record(category, "pass" if actual == expected else "blocker",
               name, actual, expected, reason)

    # Tensor dimension contract: exact GLM functional reuse cannot be claimed
    # when source dimensions are reduced without a proven invertible mapping.
    compare("dimensions", "token_vocabulary",
            model.get("vocab_size"), donor.get("vocab_size"),
            "Tokenizer IDs, embeddings and output-logit rows must match GLM.")
    compare("dimensions", "residual_hidden_size",
            model.get("hidden_size"), donor.get("hidden_size"),
            "Reducing 6144 to 4096 is non-invertible for arbitrary GLM states.")
    compare("dimensions", "attention_head_count",
            context.get("num_attention_heads"), donor.get("num_attention_heads"),
            "The number of attention heads must be compatible.")
    compare("dimensions", "attention_qk_head_width",
            context.get("head_dim"),
            donor.get("qk_head_dim",
                      donor.get("qk_nope_head_dim", 0) + donor.get("qk_rope_head_dim", 0)),
            "Head-width compression changes Q/K geometry and attention logits.")
    compare("dimensions", "routed_expert_count",
            stable.get("num_experts", moe.get("num_experts")), donor.get("n_routed_experts"),
            "Do not discard routed expert identities without a measured transfer.")
    compare("dimensions", "routed_top_k",
            stable.get("top_k", moe.get("top_k")), donor.get("num_experts_per_tok"),
            "Changing the active expert set changes the model function.")
    compare("dimensions", "expert_intermediate_size",
            stable.get("expert_intermediate_size", moe.get("expert_intermediate_size")),
            donor.get("moe_intermediate_size"),
            "Expert intermediate dimensions must map exactly.")

    # Computational semantics cannot be inferred from parameter shape alone.
    compare("semantics", "normalization_epsilon",
            model.get("rms_norm_eps"), donor.get("rms_norm_eps"),
            "RMSNorm epsilon affects numerical behavior.")
    if donor.get("hidden_act") == "silu" and recipient.get("moe_variant") == "stable_latent":
        record("semantics", "blocker", "expert_activation",
               "SiTUAndMul", "SiLU/SwiGLU",
               "The current StableLatentMoE uses SiTU, not the GLM donor's activation.")
    if method == "mamba3_foundation_plus_glm53_capability_weight_transfer":
        record("semantics", "blocker", "donor_layer_normalization",
               "new unit RMSNorm weights", "GLM pretrained normalization weights",
               "glm53_compile.py initializes attention/MoE outer layer norms to ones.")
        record("semantics", "unverified", "attention_and_routing_parity",
               "CSA/HCA compression + StableLatentMoE routing",
               "original GLM MLA/DSA + GLM MoE",
               "Mathematical and numerical parity of attention, RoPE, routing, and normalization is not established.")

    # Verify that mapping metadata is present; that is necessary, not sufficient.
    layer_map = manifest.get("source_layer_map") or {}
    if not isinstance(layer_map, dict):
        layer_map = {}
    expected_layers = donor.get("num_hidden_layers")
    covered = len(layer_map)
    compare("mapping", "mapped_donor_layers", covered, expected_layers,
            "Each donor stage needs explicit source->target provenance.")
    try:
        mapped_values = [int(v) for v in layer_map.values()]
        valid_values = (len(set(mapped_values)) == len(mapped_values)
                        and all(0 <= v < expected_layers for v in mapped_values))
    except (TypeError, ValueError):
        valid_values = False
    record("mapping", "pass" if valid_values else "blocker",
           "source_layer_indices", valid_values, True,
           "Source indices must be unique and within GLM layer bounds.")
    record("mapping", "unverified", "tensor_layout_and_permutation_parity",
           "shape/index metadata only", "reference tensor-order parity tests",
           "Weight-map completeness does not prove transposes, QKV permutations, RoPE order or per-head layouts.")

    # Residual connectivity is part of the learned GLM computation.
    compare("connectivity", "physical_layer_count", len(schedule),
            expected_layers,
            "Inserting Mamba and controller layers changes the computation graph.")
    mamba_count = sum(layer == "mamba3" for layer in schedule)
    if mamba_count:
        record("connectivity", "unverified", "mamba_residual_identity_at_initialization",
               f"{mamba_count} recurrent layers", "new branches start with zero contribution",
               "Added recurrent branches must be function-neutral until measured/trained.")
    if recipient.get("attnres") is not None:
        record("connectivity", "unverified", "attnres_residual_equivalence",
               "block attention residual controller", "original GLM residual pathway",
               "A different residual aggregation path needs explicit equivalence evidence.")

    blockers = [f for f in findings if f["status"] != "pass"]
    return {
        "status": "blocked" if blockers else "metadata_consistent_unproven",
        "note": "Metadata-only audit: even zero blockers would NOT establish forward, logit or language-quality parity.",
        "counts": {status: sum(f["status"] == status for f in findings)
                   for status in ("pass", "blocker", "unverified")},
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-config", type=Path, required=True,
                        help="Local official GLM-5.3 config.json (metadata only)")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local or read-only-mounted all256 checkpoint directory")
    args = parser.parse_args()
    required = {
        "GLM config": args.glm_config,
        "IQ config": args.checkpoint / "iq_config.json",
        "transplant manifest": args.checkpoint / "transplant_manifest.json",
    }
    try:
        records = {name: json.loads(path.read_text(encoding="utf-8"))
                   for name, path in required.items()}
        result = audit(records["GLM config"], records["IQ config"],
                       records["transplant manifest"])
    except (OSError, ValueError, TypeError) as exc:
        result = {"status": "blocked", "error": f"unreadable source/target metadata: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
