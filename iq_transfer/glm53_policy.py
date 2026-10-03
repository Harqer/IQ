from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GLM53TransferDisposition(str, Enum):
    OPERATOR_TRANSPORT = "operator_transport"
    DIRECT_REFACTOR = "direct_refactor"
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
        "attn.q_a",
        GLM53TransferDisposition.OPERATOR_TRANSPORT,
        "compressed_context.q_a",
        "Transport the donor residual basis into IQ while retaining the GLM query-latent coordinate system.",
    ),
    GLM53SourcePolicy(
        "attn.q_b",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "compressed_context.q_b",
        "Fold GLM KV expansion into the query so IQ scores directly in compressed-KV coordinates, then reduce that shared coordinate system.",
    ),
    GLM53SourcePolicy(
        "attn.kv_a_mqa",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "compressed_context.kv",
        "Transport GLM's cached compressed-KV operator into IQ's shared candidate projection; do not expand it into dense K/V first.",
    ),
    GLM53SourcePolicy(
        "attn.kv_a_norm",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "compressed_context.kv_norm",
        "GLM normalizes only the KV latent while IQ normalizes the reduced candidate, so the scale is handled by the compressed-latent refactor rather than row-copying.",
    ),
    GLM53SourcePolicy(
        "attn.kv_b",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "compressed_context.q_and_output",
        "GLM's K/V expansion is analytically folded into IQ's effective compressed query and output projections.",
    ),
    GLM53SourcePolicy(
        "attn.o",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "compressed_context.output",
        "Compose GLM V expansion and O projection, decode from the reduced compressed-KV basis, then map the residual output into IQ coordinates.",
    ),
    GLM53SourcePolicy(
        "dsa.indexer.",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "csa_indexer",
        "Refactor GLM's token-level DSA indexer weights into IQ's compressed indexer geometry; the two top-k mechanisms are not shape-identical.",
    ),
    GLM53SourcePolicy(
        "mlp.",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "stable_latent_moe",
        "Morph dense GLM SwiGLU weights directly into IQ's latent/shared expert coordinates; no teacher loss is used.",
    ),
    GLM53SourcePolicy(
        "moe.router",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "stable_latent_moe.router",
        "Both routers are sigmoid-based; directly select/align donor expert identities before transporting the retained router rows.",
    ),
    GLM53SourcePolicy(
        "moe.routing_bias",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "stable_latent_moe.routing_bias",
        "Select the same donor expert correction-bias entries as the retained router rows and re-center them in IQ's selected expert set.",
    ),
    GLM53SourcePolicy(
        "moe.shared",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "stable_latent_moe.shared_experts",
        "Directly transport shared-expert gate/up/down operators through residual and intermediate coordinate maps.",
    ),
    GLM53SourcePolicy(
        "moe.expert.",
        GLM53TransferDisposition.DIRECT_REFACTOR,
        "stable_latent_moe.routed_experts",
        "Directly transport selected routed experts through the IQ latent basis; activation mismatch is recorded as morphology error rather than hidden behind distillation.",
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
