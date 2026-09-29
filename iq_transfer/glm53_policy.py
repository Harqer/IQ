from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GLM53TransferDisposition(str, Enum):
    OPERATOR_TRANSPORT = "operator_transport"
    FUNCTIONAL_TRANSFER = "functional_transfer"
    RECIPIENT_NATIVE = "recipient_native"


@dataclass(frozen=True)
class GLM53SourcePolicy:
    role_prefix: str
    disposition: GLM53TransferDisposition
    iq_target_family: str
    rationale: str


GLM53_SOURCE_POLICIES: tuple[GLM53SourcePolicy, ...] = (
    GLM53SourcePolicy(
        "embedding",
        GLM53TransferDisposition.OPERATOR_TRANSPORT,
        "embedding",
        "Transport through a learned residual/lexical coordinate map; never row-copy across hidden bases.",
    ),
    GLM53SourcePolicy(
        "lm_head",
        GLM53TransferDisposition.OPERATOR_TRANSPORT,
        "lm_head",
        "Transport through the inverse final-residual coordinate map when the tokenizer/vocabulary contract is identical.",
    ),
    GLM53SourcePolicy(
        "norm.",
        GLM53TransferDisposition.RECIPIENT_NATIVE,
        "normalization",
        "RMSNorm scales are basis-dependent and remain recipient-native after hidden-space transport.",
    ),
    GLM53SourcePolicy(
        "attn.",
        GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
        "mamba3+compressed_context",
        "GLM MLA is factorized and normalized internally; its q/kv factors are not raw IQ Q/K/V or Mamba x/B/C matrices.",
    ),
    GLM53SourcePolicy(
        "dsa.indexer.",
        GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
        "csa_indexer",
        "Transfer sparse-selection behavior/logits rather than assuming GLM DSA indexer tensors equal IQ CSA indexer tensors.",
    ),
    GLM53SourcePolicy(
        "mlp.",
        GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
        "stable_latent_moe",
        "Dense donor MLP behavior can supervise IQ experts but does not match Stable LatentMoE topology.",
    ),
    GLM53SourcePolicy(
        "moe.router",
        GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
        "stable_latent_moe.router",
        "Both routers are sigmoid-based, but expert identities/topology and latent expert space require routing alignment.",
    ),
    GLM53SourcePolicy(
        "moe.shared",
        GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
        "stable_latent_moe.shared_experts",
        "GLM shared experts operate at donor width/activation semantics and must be behaviorally aligned.",
    ),
    GLM53SourcePolicy(
        "moe.expert.",
        GLM53TransferDisposition.FUNCTIONAL_TRANSFER,
        "stable_latent_moe.routed_experts",
        "GLM routed experts are full-width donor experts while IQ routed experts operate in latent width with SiTU.",
    ),
)


def classify_glm53_source_role(role: str) -> GLM53SourcePolicy:
    matches = [
        policy
        for policy in GLM53_SOURCE_POLICIES
        if role == policy.role_prefix or role.startswith(policy.role_prefix)
    ]
    if not matches:
        raise ValueError(f"unclassified GLM-5.3 source role: {role!r}")
    # Prefer the most specific prefix so moe.router/shared/expert override generic forms.
    return max(matches, key=lambda policy: len(policy.role_prefix))


IQ_RECIPIENT_NATIVE_FAMILIES: tuple[str, ...] = (
    "mamba3.recurrence",
    "block_attnres",
    "reasoning_recurrence",
    "adaptive_halting",
    "energy_critic",
    "mtp_native_state",
)
