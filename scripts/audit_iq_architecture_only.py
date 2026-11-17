"""Validate IQ's canonical GLM-tokenized hybrid architecture without donor weights.

This path does not load GLM/Mamba checkpoints, build an inference engine,
run language generation, or claim transfer of any pretrained behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from iq_model import HybridLayerType, validate_canonical_hybrid_backbone
from iq_model.glm_tokenizer import (
    GLM53_EOS_TOKEN_IDS,
    GLM53_PAD_TOKEN_ID,
    GLM53_VOCAB_SIZE,
    with_glm_tokenizer_config,
)
from iq_transfer.complete_transplant import (
    canonical_glm53_config,
    expected_complete_state_keys,
)


class ArchitectureOnlyError(ValueError):
    """IQ's expected architecture-only contract changed."""


@dataclass(frozen=True)
class ArchitectureOnlyResult:
    report: dict[str, object]
    config: object


def inspect_architecture_only() -> ArchitectureOnlyResult:
    """Use the actual canonical IQ hybrid with a GLM-native ID space.

    Unlike the compiled GLM transplant, this intentionally never invokes
    glm53_compile.py, loading donor neural weights or calibration maps.
    """
    original = canonical_glm53_config()
    config = validate_canonical_hybrid_backbone(
        with_glm_tokenizer_config(original)
    )
    if config.model.vocab_size != GLM53_VOCAB_SIZE:
        raise ArchitectureOnlyError("wrong GLM vocabulary geometry")
    if config.model.hidden_size != 4096:
        raise ArchitectureOnlyError("canonical IQ residual width changed")
    if config.mamba3.mimo_rank != 4 or config.mamba3.d_model != 4096:
        raise ArchitectureOnlyError("native Mamba-3 MIMO architecture changed")
    if config.schedule != original.schedule:
        raise ArchitectureOnlyError("tokenizer swap changed physical layers")
    if config.model.num_hidden_layers != len(config.schedule.layers):
        raise ArchitectureOnlyError("physical-layer depth mismatch")
    if config.moe != original.moe or config.stable_moe != original.stable_moe:
        raise ArchitectureOnlyError("tokenizer swap changed expert configuration")
    if (
        config.compressed_context != original.compressed_context
        or config.attnres != original.attnres
        or config.reasoning != original.reasoning
        or config.multimodal != original.multimodal
    ):
        raise ArchitectureOnlyError("tokenizer swap changed IQ context/controller modules")
    counter = Counter(config.schedule.layers)
    expected_counts = {
        HybridLayerType.MAMBA3: 32,
        HybridLayerType.MOE: 24,
        HybridLayerType.CSA: 12,
        HybridLayerType.HCA: 12,
    }
    if any(counter.get(kind, 0) != number for kind, number in expected_counts.items()):
        raise ArchitectureOnlyError("unexpected IQ physical layer composition")
    keys = expected_complete_state_keys(config)
    for name in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
        if name not in keys:
            raise ArchitectureOnlyError(f"missing IQ lexical/global state slot: {name}")
    for section in ("attnres.", "reasoning.", "reasoning_injector."):
        if not any(key.startswith(section) for key in keys):
            raise ArchitectureOnlyError(f"missing IQ component in state schema: {section}")
    if not any(".mamba.core.mimo_" in key for key in keys):
        raise ArchitectureOnlyError("rank-4 MIMO state tensors absent from checkpoint schema")
    if not any(".attention." in key for key in keys):
        raise ArchitectureOnlyError("CSA/HCA state tensors absent from checkpoint schema")
    if not any(".moe." in key for key in keys):
        raise ArchitectureOnlyError("Stable LatentMoE state tensors absent from schema")

    return ArchitectureOnlyResult(
        config=config,
        report={
            "status": "canonical_iq_architecture_schema_valid",
            "model_class": "IQHybridForCausalLM",
            "architecture": "Mamba-3 MIMO + Stable LatentMoE + CSA/HCA + Block AttnRes + reasoning",
            "physical_layers": len(config.schedule.layers),
            "mamba3_mimo_layers": counter[HybridLayerType.MAMBA3],
            "mimo_rank": config.mamba3.mimo_rank,
            "context_layers": {
                "csa": counter[HybridLayerType.CSA],
                "hca": counter[HybridLayerType.HCA],
            },
            "moe_layers": counter[HybridLayerType.MOE],
            "residual_width": config.model.hidden_size,
            "glm_tokenizer_vocab": config.model.vocab_size,
            "glm_pad_token_id": GLM53_PAD_TOKEN_ID,
            "glm_eos_token_ids": list(GLM53_EOS_TOKEN_IDS),
            "embedding_weight_shape": [GLM53_VOCAB_SIZE, config.model.hidden_size],
            "lm_head_weight_shape": [GLM53_VOCAB_SIZE, config.model.hidden_size],
            "expected_state_tensor_keys": len(keys),
            "schedule_unchanged": True,
            "donor_neural_weights_loaded": False,
            "glm_weight_transfer_executed": False,
            "model_forward_executed": False,
            "pretrained_capabilities_claimed": False,
            "iq_config_fingerprint": config.fingerprint,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-config", action="store_true",
        help="Write ./iq_glm53_architecture.json (refuse to overwrite an existing file)",
    )
    args = parser.parse_args(argv)
    result = inspect_architecture_only()
    report = dict(result.report)
    if args.write_config:
        target = Path("iq_glm53_architecture.json")
        if target.exists():
            raise ArchitectureOnlyError(
                f"architecture config already exists; refusing overwrite: {target}"
            )
        result.config.write_json(str(target))
        report["written_config"] = str(target.resolve())
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
