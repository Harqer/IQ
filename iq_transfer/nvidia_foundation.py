"""Verified NVIDIA Nemotron 3 Super donor contract for IQ's parallel-hybrid design.

This module validates actual HF model metadata and generates a fail-closed
transfer *eligibility* report. It does not mislabel Mamba-2 recurrence as
Mamba-3 MIMO or claim Falcon-H1-native weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Mapping
import json

NVIDIA_BASE_REPO = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"
FALCON_H1_REFERENCE = (
    "https://developer.nvidia.com/blog/"
    "implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/"
)
TARGET_HIDDEN_SIZE = 4096
TARGET_HYBRID_BLOCKS = 78


class NvidiaFoundationError(ValueError):
    pass


@dataclass(frozen=True)
class NvidiaDonorGeometry:
    model_type: str
    vocab_size: int
    hidden_size: int
    layer_count: int
    mamba2_layers: int
    attention_layers: int
    moe_layers: int
    latent_size: int
    routed_experts: int
    token_experts: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "NvidiaDonorGeometry":
        if config.get("model_type") != "nemotron_h":
            raise NvidiaFoundationError("NVIDIA foundation must be nemotron_h")
        if "quantization_config" in config:
            raise NvidiaFoundationError("Use the official Base-BF16, not quantized donor")
        pattern = config.get("hybrid_override_pattern")
        if not isinstance(pattern, str) or any(c not in "ME*" for c in pattern):
            raise NvidiaFoundationError("Unsupported Nemotron hybrid override pattern")
        try:
            result = cls(
                model_type=str(config["model_type"]),
                vocab_size=int(config["vocab_size"]),
                hidden_size=int(config["hidden_size"]),
                layer_count=int(config["num_hidden_layers"]),
                mamba2_layers=pattern.count("M"),
                attention_layers=pattern.count("*"),
                moe_layers=pattern.count("E"),
                latent_size=int(config["moe_latent_size"]),
                routed_experts=int(config["n_routed_experts"]),
                token_experts=int(config["num_experts_per_tok"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NvidiaFoundationError(f"Missing Nemotron donor geometry: {exc}") from exc
        if len(pattern) != result.layer_count:
            raise NvidiaFoundationError("Donor block pattern differs from declared depth")
        # These values are from the real published NVIDIA Base BF16 config.
        expected = {
            "vocab_size": 131072,
            "hidden_size": 4096,
            "layer_count": 88,
            "mamba2_layers": 40,
            "attention_layers": 8,
            "moe_layers": 40,
            "latent_size": 1024,
            "routed_experts": 512,
            "token_experts": 22,
        }
        mismatches = [
            f"{key}={getattr(result, key)} != {value}"
            for key, value in expected.items()
            if getattr(result, key) != value
        ]
        if mismatches:
            raise NvidiaFoundationError(
                "Unexpected NVIDIA checkpoint config; do not transfer: "
                + "; ".join(mismatches)
            )
        return result


@dataclass(frozen=True)
class NvidiaFoundationPlan:
    donor_repo: str
    donor_revision: str
    donor_config_sha256: str
    geometry: NvidiaDonorGeometry
    recipient_hidden_size: int
    recipient_parallel_blocks: int
    expected_attention_branches: int
    expected_moe_branches: int
    direct_mamba2_to_mamba3: bool
    same_tokenizer_as_mamba3: bool
    weight_dispositions: Mapping[str, str]
    reference_implementation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_foundation(
    config: Mapping[str, Any],
    *,
    donor_revision: str,
    config_bytes: bytes,
) -> NvidiaFoundationPlan:
    if not isinstance(donor_revision, str) or len(donor_revision) != 40 or not all(
        c in "0123456789abcdef" for c in donor_revision.lower()
    ):
        raise NvidiaFoundationError("Pin the exact 40-character Hub commit SHA")
    if not config_bytes:
        raise NvidiaFoundationError("Checkpoint config bytes are empty")
    geometry = NvidiaDonorGeometry.from_config(config)
    return NvidiaFoundationPlan(
        donor_repo=NVIDIA_BASE_REPO,
        donor_revision=donor_revision,
        donor_config_sha256=sha256(config_bytes).hexdigest(),
        geometry=geometry,
        recipient_hidden_size=TARGET_HIDDEN_SIZE,
        recipient_parallel_blocks=TARGET_HYBRID_BLOCKS,
        expected_attention_branches=TARGET_HYBRID_BLOCKS,
        expected_moe_branches=TARGET_HYBRID_BLOCKS,
        direct_mamba2_to_mamba3=False,
        same_tokenizer_as_mamba3=False,
        weight_dispositions={
            "embedding_and_lm_head": (
                "eligible only if IQ adopts NVIDIA tokenizer and 131072-token "
                "vocabulary; do not overwrite existing Mamba lexical weights blindly"
            ),
            "mamba2_recurrence": (
                "NOT directly portable to native rank-4 Mamba-3 MIMO; "
                "retain separately verified Mamba-3 MIMO source"
            ),
            "attention": (
                "8 NVIDIA attention donors for 78 IQ parallel attention branches; "
                "operator transport and correspondence required"
            ),
            "latent_moe": (
                "40 NVIDIA MoE donors for 78 IQ MoE branches; "
                "expert selection and latent-dimension transport required"
            ),
            "glm53_later": (
                "reserved secondary capability donor; "
                "requires its own BF16 tensor manifest, geometry, "
                "teacher/student alignment and validation"
            ),
        },
        reference_implementation=FALCON_H1_REFERENCE,
    )


def serialize_plan(plan: NvidiaFoundationPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
