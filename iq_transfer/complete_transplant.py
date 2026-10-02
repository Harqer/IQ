from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json
import math
import shutil
import tempfile

import torch

from iq_model import (
    BlockAttnResConfig,
    BlockAttentionResidual,
    CompressedContextConfig,
    HybridLayerType,
    HybridSchedule,
    IQHybridConfig,
    IQHybridForCausalLM,
    IQModelConfig,
    IQMultimodalConfig,
    Mamba3MIMOConfig,
    ReasoningRecurrence,
    ReasoningRecurrenceConfig,
    ReasoningStateInjector,
    RoutedMoEConfig,
    StableLatentMoEConfig,
    validate_canonical_hybrid_backbone,
)

from .gpt_oss20b import (
    GPT_OSS_20B_REPO,
    GptOss20BOriginalCheckpoint,
)
from .mamba3_direct import (
    MAMBA3_MIMO_15B_REPO,
    Mamba3Layout,
    compile_official_mamba3_mimo_15b_transplant,
)


class CompleteTransplantError(RuntimeError):
    pass


GPT_OSS_SOURCE_WIDTH = 2880
IQ_TARGET_WIDTH = 4096
RESIDUAL_EMBED_SCALE = math.sqrt(IQ_TARGET_WIDTH / GPT_OSS_SOURCE_WIDTH)


def canonical_complete_schedule() -> HybridSchedule:
    """32 Mamba + 24 context + 24 MoE physical layers.

    Every GPT-OSS donor layer maps to one context and one Stable LatentMoE
    layer. A Mamba layer precedes every donor-derived pair; eight additional
    Mamba layers are distributed every third donor layer.
    """
    layers: list[HybridLayerType] = []
    for donor_layer in range(24):
        layers.append(HybridLayerType.MAMBA3)
        layers.append(
            HybridLayerType.CSA
            if donor_layer % 2 == 0
            else HybridLayerType.HCA
        )
        layers.append(HybridLayerType.MOE)
        if (donor_layer + 1) % 3 == 0:
            layers.append(HybridLayerType.MAMBA3)
    schedule = HybridSchedule(tuple(layers))
    if schedule.count(HybridLayerType.MAMBA3) != 32:
        raise CompleteTransplantError("canonical schedule must contain 32 Mamba layers")
    if schedule.count(HybridLayerType.MOE) != 24:
        raise CompleteTransplantError("canonical schedule must contain 24 MoE layers")
    if len(schedule.attention_positions) != 24:
        raise CompleteTransplantError("canonical schedule must contain 24 context layers")
    return schedule


def canonical_complete_config() -> IQHybridConfig:
    schedule = canonical_complete_schedule()
    model = IQModelConfig(
        vocab_size=201088,
        hidden_size=4096,
        num_hidden_layers=len(schedule.layers),
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=11520,
        max_position_embeddings=131072,
        rope_theta=150000.0,
        rotary_fraction=1.0,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        residual_dropout=0.0,
        tie_word_embeddings=False,
        initializer_range=0.02,
    )
    mamba = Mamba3MIMOConfig.production_4096x32()
    control_moe = RoutedMoEConfig(
        hidden_size=4096,
        expert_intermediate_size=2880,
        num_experts=32,
        top_k=4,
        shared_expert_intermediate_size=2880,
        router_bias=True,
    )
    stable = StableLatentMoEConfig(
        hidden_size=4096,
        latent_size=2880,
        expert_intermediate_size=2880,
        num_experts=32,
        top_k=4,
        num_shared_experts=1,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        expert_bias=True,
        router_bias=True,
    )
    compressed = CompressedContextConfig(
        hidden_size=4096,
        num_attention_heads=64,
        head_dim=64,
        q_lora_rank=4096,
        partial_rotary_dim=32,
        max_position_embeddings=131072,
        sliding_window=128,
        csa_compress_rate=4,
        hca_compress_rate=128,
        o_groups=8,
        o_lora_rank=512,
        index_n_heads=64,
        index_head_dim=64,
        index_topk=512,
        compress_rope_theta=150000.0,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        projection_bias=True,
    )
    attnres = BlockAttnResConfig(
        hidden_size=4096,
        num_layers=len(schedule.layers),
        block_size=12,
        rms_norm_eps=1e-5,
    )
    reasoning = ReasoningRecurrenceConfig(
        hidden_size=4096,
        state_dim=512,
        transition_hidden_dim=1024,
        max_steps=4,
        min_steps=1,
        halt_threshold=0.9,
        state_delta_epsilon=1e-3,
        energy_delta_epsilon=1e-3,
        dropout=0.0,
        rms_norm_eps=1e-5,
        depth_base=10000.0,
    )
    return validate_canonical_hybrid_backbone(
        IQHybridConfig(
            model=model,
            schedule=schedule,
            mamba3=mamba,
            moe=control_moe,
            moe_variant="stable_latent",
            stable_moe=stable,
            attnres=attnres,
            compressed_context=compressed,
            reasoning=reasoning,
            energy_critic=None,
            require_energy_stability=False,
        )
    )



def canonical_glm53_config() -> IQHybridConfig:
    """IQ recipient geometry for direct GLM-5.3-BF16 language transfer.

    The recurrent schedule stays IQ-native/Mamba-3. Context geometry keeps the
    donor's query-latent and rotary semantics where doing so enables direct
    algebraic transfer, while Stable LatentMoE retains IQ's compressed expert
    topology.
    """
    base = canonical_complete_config()
    assert base.stable_moe is not None
    assert base.compressed_context is not None
    model = replace(
        base.model,
        vocab_size=154880,
        num_key_value_heads=64,
        max_position_embeddings=202752,
        rope_theta=8_000_000.0,
    )
    control_moe = replace(
        base.moe,
        expert_intermediate_size=2048,
        top_k=8,
        shared_expert_intermediate_size=2048,
    )
    stable = replace(
        base.stable_moe,
        latent_size=2048,
        expert_intermediate_size=2048,
        top_k=8,
        situ_beta=1_000_000.0,
        situ_linear_beta=1_000_000.0,
        routed_scaling_factor=2.5,
    )
    compressed = replace(
        base.compressed_context,
        head_dim=128,
        q_lora_rank=2048,
        partial_rotary_dim=64,
        max_position_embeddings=202752,
        o_lora_rank=1024,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        compress_rope_theta=8_000_000.0,
        normalize_query_output=False,
        normalize_candidate_nonrotary_only=True,
        direct_token_indexer=True,
    )
    return validate_canonical_hybrid_backbone(
        replace(
            base,
            model=model,
            moe=control_moe,
            stable_moe=stable,
            compressed_context=compressed,
        )
    )


def canonical_glm53_multimodal_config() -> IQHybridConfig:
    base = canonical_glm53_config()
    context_layers = tuple(base.schedule.attention_positions)
    fusion_layers = tuple(sorted(set((*context_layers, 39))))
    multimodal = IQMultimodalConfig(
        vision_model_name="zai-org/GLM-5.3-Flash-BF16",
        vision_backend="glm5_next",
        fusion_layers=fusion_layers,
        transv_layers=(7, 39),
        visual_mamba_layers=1,
        cross_attention_heads=8,
        transv_shallow_keep_ratio=0.5,
        transv_deep_keep_ratio=0.1,
        min_visual_tokens=16,
        max_frames=16384,
        freeze_vision_tower=True,
        drop_cls_token=False,
    )
    return validate_canonical_hybrid_backbone(
        replace(base, multimodal=multimodal)
    )



def canonical_multimodal_config() -> IQHybridConfig:
    """Canonical IQ backbone plus native GLM-5.3 vision and long-video fusion.

    Vamba-style cross-modal fusion is attached to every CSA/HCA depth. TransV
    follows the documented shallow/deep placement pattern at physical depths
    7 and 39: 50% uniform retention, then 10% attention-guided retention.
    """
    base = canonical_complete_config()
    context_layers = tuple(base.schedule.attention_positions)
    fusion_layers = tuple(sorted(set((*context_layers, 39))))
    multimodal = IQMultimodalConfig(
        vision_model_name="zai-org/GLM-5.3-Flash-BF16",
        vision_backend="glm5_next",
        fusion_layers=fusion_layers,
        transv_layers=(7, 39),
        visual_mamba_layers=1,
        cross_attention_heads=8,
        transv_shallow_keep_ratio=0.5,
        transv_deep_keep_ratio=0.1,
        min_visual_tokens=16,
        max_frames=16384,
        freeze_vision_tower=True,
        drop_cls_token=False,
    )
    return validate_canonical_hybrid_backbone(
        replace(base, multimodal=multimodal)
    )

def donor_layer_positions(
    config: IQHybridConfig,
) -> tuple[tuple[int, int], ...]:
    """Return (context_physical, moe_physical) for donor layers 0..23."""
    context = list(config.schedule.attention_positions)
    moe = list(config.schedule.positions(HybridLayerType.MOE))
    if len(context) != 24 or len(moe) != 24:
        raise CompleteTransplantError("canonical donor-layer mapping requires 24 context/MoE layers")
    return tuple(zip(context, moe, strict=True))


def _expand_norm(weight: torch.Tensor) -> torch.Tensor:
    if tuple(weight.shape) != (GPT_OSS_SOURCE_WIDTH,):
        raise CompleteTransplantError(f"norm shape mismatch: {tuple(weight.shape)}")
    out = torch.ones(IQ_TARGET_WIDTH, dtype=weight.dtype)
    out[:GPT_OSS_SOURCE_WIDTH] = weight
    return out


def _project_residual_input(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[1] != GPT_OSS_SOURCE_WIDTH:
        raise CompleteTransplantError(
            f"input projection must end in {GPT_OSS_SOURCE_WIDTH}: {tuple(weight.shape)}"
        )
    out = torch.zeros(
        weight.shape[0],
        IQ_TARGET_WIDTH,
        dtype=weight.dtype,
    )
    out[:, :GPT_OSS_SOURCE_WIDTH] = weight / RESIDUAL_EMBED_SCALE
    return out


def _embed_residual_output(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[0] != GPT_OSS_SOURCE_WIDTH:
        raise CompleteTransplantError(
            f"output projection must start with {GPT_OSS_SOURCE_WIDTH}: {tuple(weight.shape)}"
        )
    out = torch.zeros(
        IQ_TARGET_WIDTH,
        weight.shape[1],
        dtype=weight.dtype,
    )
    out[:GPT_OSS_SOURCE_WIDTH] = weight * RESIDUAL_EMBED_SCALE
    return out


def _embed_residual_bias(bias: torch.Tensor) -> torch.Tensor:
    if tuple(bias.shape) != (GPT_OSS_SOURCE_WIDTH,):
        raise CompleteTransplantError(f"residual bias shape mismatch: {tuple(bias.shape)}")
    out = torch.zeros(IQ_TARGET_WIDTH, dtype=bias.dtype)
    out[:GPT_OSS_SOURCE_WIDTH] = bias * RESIDUAL_EMBED_SCALE
    return out


def _residual_left_inverse(dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(
        GPT_OSS_SOURCE_WIDTH,
        IQ_TARGET_WIDTH,
        dtype=dtype,
    )
    diagonal = torch.arange(GPT_OSS_SOURCE_WIDTH)
    out[diagonal, diagonal] = 1.0 / RESIDUAL_EMBED_SCALE
    return out


def _residual_embedding(dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(
        IQ_TARGET_WIDTH,
        GPT_OSS_SOURCE_WIDTH,
        dtype=dtype,
    )
    diagonal = torch.arange(GPT_OSS_SOURCE_WIDTH)
    out[diagonal, diagonal] = RESIDUAL_EMBED_SCALE
    return out


def _context_state(
    donor: Mapping[str, torch.Tensor],
    *,
    mode: HybridLayerType,
    physical_layer: int,
) -> dict[str, torch.Tensor]:
    if mode not in {HybridLayerType.CSA, HybridLayerType.HCA}:
        raise CompleteTransplantError("context mode must be CSA or HCA")
    prefix = f"layers.{physical_layer}"
    q_weight = donor["q_weight"]
    q_bias = donor["q_bias"]
    if tuple(q_weight.shape) != (4096, 2880):
        raise CompleteTransplantError(f"unexpected donor Q shape {tuple(q_weight.shape)}")

    k_heads = donor["k_weight"].reshape(8, 64, 2880)
    v_heads = donor["v_weight"].reshape(8, 64, 2880)
    shared_kv = torch.cat((k_heads, v_heads), dim=0).float().mean(dim=0).to(k_heads.dtype)
    k_bias = donor["k_bias"].reshape(8, 64)
    v_bias = donor["v_bias"].reshape(8, 64)
    shared_kv_bias = torch.cat((k_bias, v_bias), dim=0).float().mean(dim=0).to(k_bias.dtype)
    target_kv = _project_residual_input(shared_kv)

    state: dict[str, torch.Tensor] = {
        f"{prefix}.norm.weight": _expand_norm(donor["norm"]),
        f"{prefix}.attention.q_a_proj.weight": _project_residual_input(q_weight),
        f"{prefix}.attention.q_a_proj.bias": q_bias.clone(),
        f"{prefix}.attention.q_a_norm.weight": torch.ones(4096, dtype=q_weight.dtype),
        f"{prefix}.attention.q_b_proj.weight": torch.eye(4096, dtype=q_weight.dtype),
        f"{prefix}.attention.q_b_proj.bias": torch.zeros(4096, dtype=q_bias.dtype),
        f"{prefix}.attention.kv_proj.weight": target_kv,
        f"{prefix}.attention.kv_proj.bias": shared_kv_bias.clone(),
        f"{prefix}.attention.kv_norm.weight": torch.ones(64, dtype=q_weight.dtype),
        f"{prefix}.attention.compressor_gate_proj.weight": torch.zeros(
            128 if mode is HybridLayerType.CSA else 64,
            4096,
            dtype=q_weight.dtype,
        ),
        f"{prefix}.attention.compressor_position_bias": torch.zeros(
            4 if mode is HybridLayerType.CSA else 128,
            128 if mode is HybridLayerType.CSA else 64,
            dtype=q_weight.dtype,
        ),
        f"{prefix}.attention.compressor_kv_norm.weight": torch.ones(64, dtype=q_weight.dtype),
        f"{prefix}.attention.sinks": donor["sinks"].clone(),
    }

    if mode is HybridLayerType.CSA:
        state[f"{prefix}.attention.compressor_kv_proj.weight"] = torch.cat(
            (target_kv, target_kv),
            dim=0,
        )
        state[f"{prefix}.attention.compressor_kv_proj.bias"] = torch.cat(
            (shared_kv_bias, shared_kv_bias),
            dim=0,
        )
        state[f"{prefix}.attention.index_kv_proj.weight"] = torch.cat(
            (target_kv, target_kv),
            dim=0,
        )
        state[f"{prefix}.attention.index_kv_proj.bias"] = torch.cat(
            (shared_kv_bias, shared_kv_bias),
            dim=0,
        )
        state[f"{prefix}.attention.index_gate_proj.weight"] = torch.zeros(
            128,
            4096,
            dtype=q_weight.dtype,
        )
        state[f"{prefix}.attention.index_gate_proj.bias"] = torch.zeros(
            128,
            dtype=q_bias.dtype,
        )
        state[f"{prefix}.attention.index_position_bias"] = torch.zeros(
            4,
            128,
            dtype=q_weight.dtype,
        )
        state[f"{prefix}.attention.index_kv_norm.weight"] = torch.ones(
            64,
            dtype=q_weight.dtype,
        )
        state[f"{prefix}.attention.index_q_proj.weight"] = torch.eye(
            4096,
            dtype=q_weight.dtype,
        )
        state[f"{prefix}.attention.index_q_proj.bias"] = torch.zeros(
            4096,
            dtype=q_bias.dtype,
        )
        state[f"{prefix}.attention.index_weight_proj.weight"] = torch.zeros(
            64,
            4096,
            dtype=q_weight.dtype,
        )
        state[f"{prefix}.attention.index_weight_proj.bias"] = torch.zeros(
            64,
            dtype=q_bias.dtype,
        )
    else:
        state[f"{prefix}.attention.compressor_kv_proj.weight"] = target_kv
        state[f"{prefix}.attention.compressor_kv_proj.bias"] = shared_kv_bias.clone()

    # GroupedLowRankOutput can exactly represent GPT-OSS O because each of the
    # eight groups contains 8*64 = 512 features and IQ uses rank 512.
    state[f"{prefix}.attention.output.weight"] = torch.eye(
        512,
        dtype=q_weight.dtype,
    ).unsqueeze(0).repeat(8, 1, 1)
    state[f"{prefix}.attention.output.out_proj.weight"] = _embed_residual_output(
        donor["out_weight"]
    )
    state[f"{prefix}.attention.output.out_proj.bias"] = _embed_residual_bias(
        donor["out_bias"]
    )
    return state


def _moe_state(
    donor: Mapping[str, torch.Tensor],
    *,
    physical_layer: int,
) -> dict[str, torch.Tensor]:
    prefix = f"layers.{physical_layer}.moe"
    gate = donor["gate_weight"]
    up = donor["up_weight"]
    down = donor["down_weight"]
    if tuple(gate.shape) != (32, 2880, 2880):
        raise CompleteTransplantError(f"unexpected gate expert shape {tuple(gate.shape)}")
    if tuple(up.shape) != (32, 2880, 2880):
        raise CompleteTransplantError(f"unexpected up expert shape {tuple(up.shape)}")
    if tuple(down.shape) != (32, 2880, 2880):
        raise CompleteTransplantError(f"unexpected down expert shape {tuple(down.shape)}")

    dtype = gate.dtype
    state: dict[str, torch.Tensor] = {
        f"layers.{physical_layer}.norm.weight": _expand_norm(donor["norm"]),
        f"{prefix}.router.weight": _project_residual_input(donor["router_weight"]),
        f"{prefix}.router.bias": donor["router_bias"].clone(),
        f"{prefix}.latent_down.weight": _residual_left_inverse(dtype),
        f"{prefix}.latent_up.weight": _residual_embedding(dtype),
        f"{prefix}.latent_norm.weight": torch.ones(2880, dtype=dtype),
        f"{prefix}.routing_bias": torch.zeros(32, dtype=torch.float32),
    }
    for expert in range(32):
        base = f"{prefix}.routed_experts.{expert}"
        state[f"{base}.gate_proj.weight"] = gate[expert].contiguous()
        state[f"{base}.gate_proj.bias"] = donor["gate_bias"][expert].contiguous()
        state[f"{base}.up_proj.weight"] = up[expert].contiguous()
        state[f"{base}.up_proj.bias"] = donor["up_bias"][expert].contiguous()
        state[f"{base}.down_proj.weight"] = down[expert].contiguous()
        state[f"{base}.down_proj.bias"] = donor["down_bias"][expert].contiguous()

    # Stable LatentMoE requires a full-width shared expert. GPT-OSS has no
    # shared expert, so initialize it as an exact no-op while keeping it present.
    shared = f"{prefix}.shared_experts.0"
    state[f"{shared}.gate_proj.weight"] = torch.zeros(2880, 4096, dtype=dtype)
    state[f"{shared}.gate_proj.bias"] = torch.zeros(2880, dtype=dtype)
    state[f"{shared}.up_proj.weight"] = torch.zeros(2880, 4096, dtype=dtype)
    state[f"{shared}.up_proj.bias"] = torch.zeros(2880, dtype=dtype)
    state[f"{shared}.down_proj.weight"] = torch.zeros(4096, 2880, dtype=dtype)
    state[f"{shared}.down_proj.bias"] = torch.zeros(4096, dtype=dtype)
    return state


def _identity_mamba_state(
    *,
    physical_layer: int,
    config: Mamba3MIMOConfig,
) -> dict[str, torch.Tensor]:
    layout = Mamba3Layout(
        d_model=config.d_model,
        d_state=config.d_state,
        expand=config.expand,
        headdim=config.headdim,
        ngroups=1,
        rope_fraction=config.rope_fraction,
        is_mimo=True,
        mimo_rank=config.mimo_rank,
    )
    q = f"layers.{physical_layer}"
    return {
        f"{q}.norm.weight": torch.ones(config.d_model, dtype=torch.bfloat16),
        f"{q}.mamba.core.in_proj.weight": torch.zeros(
            layout.in_proj_shape,
            dtype=torch.bfloat16,
        ),
        f"{q}.mamba.core.dt_bias": torch.zeros(layout.nheads, dtype=torch.float32),
        f"{q}.mamba.core.B_bias": torch.ones(
            layout.nheads,
            config.mimo_rank,
            config.d_state,
            dtype=torch.float32,
        ),
        f"{q}.mamba.core.C_bias": torch.ones(
            layout.nheads,
            config.mimo_rank,
            config.d_state,
            dtype=torch.float32,
        ),
        f"{q}.mamba.core.B_norm.weight": torch.ones(config.d_state, dtype=torch.bfloat16),
        f"{q}.mamba.core.C_norm.weight": torch.ones(config.d_state, dtype=torch.bfloat16),
        f"{q}.mamba.core.mimo_x": torch.full(
            (layout.nheads, config.mimo_rank, config.headdim),
            1.0 / config.mimo_rank,
            dtype=torch.float32,
        ),
        f"{q}.mamba.core.mimo_z": torch.ones(
            layout.nheads,
            config.mimo_rank,
            config.headdim,
            dtype=torch.float32,
        ),
        f"{q}.mamba.core.mimo_o": torch.full(
            (layout.nheads, config.mimo_rank, config.headdim),
            1.0 / config.mimo_rank,
            dtype=torch.float32,
        ),
        f"{q}.mamba.core.D": torch.ones(layout.nheads, dtype=torch.float32),
        f"{q}.mamba.core.out_proj.weight": torch.zeros(
            layout.out_proj_shape,
            dtype=torch.bfloat16,
        ),
    }


def _global_state(donor: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    embedding = donor["embedding"]
    lm_head = donor["lm_head"]
    if tuple(embedding.shape) != (201088, 2880):
        raise CompleteTransplantError(f"unexpected embedding shape {tuple(embedding.shape)}")
    if tuple(lm_head.shape) != (201088, 2880):
        raise CompleteTransplantError(f"unexpected LM-head shape {tuple(lm_head.shape)}")
    target_embedding = torch.zeros(
        201088,
        4096,
        dtype=embedding.dtype,
    )
    target_embedding[:, :2880] = embedding * RESIDUAL_EMBED_SCALE
    target_lm = torch.zeros(
        201088,
        4096,
        dtype=lm_head.dtype,
    )
    target_lm[:, :2880] = lm_head / RESIDUAL_EMBED_SCALE
    return {
        "embed_tokens.weight": target_embedding,
        "norm.weight": _expand_norm(donor["norm"]),
        "lm_head.weight": target_lm,
    }


def _native_controller_state(config: IQHybridConfig) -> dict[str, torch.Tensor]:
    if config.attnres is None or config.reasoning is None:
        raise CompleteTransplantError("canonical complete config requires AttnRes and reasoning")
    torch.manual_seed(260929)
    attnres = BlockAttentionResidual(config.attnres).to(dtype=torch.bfloat16)
    reasoning = ReasoningRecurrence(config.reasoning).to(dtype=torch.bfloat16)
    injector = ReasoningStateInjector(
        config.model.hidden_size,
        config.reasoning.state_dim,
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        # Keep the new reasoning controller function-neutral until it has been
        # trained: all native reasoning state may run, but cannot perturb token
        # representations in the transplanted checkpoint.
        injector.proj.weight.zero_()
        injector.gate_logit.fill_(-20.0)

    result: dict[str, torch.Tensor] = {}
    for key, value in attnres.state_dict().items():
        result[f"attnres.{key}"] = value.cpu()
    for key, value in reasoning.state_dict().items():
        result[f"reasoning.{key}"] = value.cpu()
    for key, value in injector.state_dict().items():
        result[f"reasoning_injector.{key}"] = value.cpu()
    return result


def expected_complete_state_keys(
    config: IQHybridConfig | None = None,
) -> frozenset[str]:
    """Derive the exact serialized state schema without allocating model weights."""
    if config is None:
        config = canonical_complete_config()
    keys: set[str] = {
        "embed_tokens.weight",
        "norm.weight",
        "lm_head.weight",
    }
    with torch.device("meta"):
        from iq_model import (
            CompressedContextResidualLayer,
            StableLatentMoELayer,
        )
        for physical, layer_type in enumerate(config.schedule.layers):
            if layer_type is HybridLayerType.MAMBA3:
                q = f"layers.{physical}"
                keys.update(
                    {
                        f"{q}.norm.weight",
                        f"{q}.mamba.core.in_proj.weight",
                        f"{q}.mamba.core.dt_bias",
                        f"{q}.mamba.core.B_bias",
                        f"{q}.mamba.core.C_bias",
                        f"{q}.mamba.core.B_norm.weight",
                        f"{q}.mamba.core.C_norm.weight",
                        f"{q}.mamba.core.mimo_x",
                        f"{q}.mamba.core.mimo_z",
                        f"{q}.mamba.core.mimo_o",
                        f"{q}.mamba.core.D",
                        f"{q}.mamba.core.out_proj.weight",
                    }
                )
            elif layer_type is HybridLayerType.MOE:
                assert config.stable_moe is not None
                module = StableLatentMoELayer(
                    config.stable_moe,
                    norm_eps=config.model.rms_norm_eps,
                    residual_dropout=config.model.residual_dropout,
                )
                keys.update(
                    f"layers.{physical}.{name}"
                    for name in module.state_dict().keys()
                )
            elif layer_type in {HybridLayerType.CSA, HybridLayerType.HCA}:
                assert config.compressed_context is not None
                module = CompressedContextResidualLayer(
                    config.model,
                    config.compressed_context,
                    mode=layer_type,
                )
                keys.update(
                    f"layers.{physical}.{name}"
                    for name in module.state_dict().keys()
                )
            else:
                raise CompleteTransplantError(
                    f"unsupported canonical layer type: {layer_type}"
                )

        if config.attnres is not None:
            module = BlockAttentionResidual(config.attnres)
            keys.update(f"attnres.{name}" for name in module.state_dict().keys())
        if config.reasoning is not None:
            module = ReasoningRecurrence(config.reasoning)
            keys.update(f"reasoning.{name}" for name in module.state_dict().keys())
            injector = ReasoningStateInjector(
                config.model.hidden_size,
                config.reasoning.state_dim,
            )
            keys.update(
                f"reasoning_injector.{name}"
                for name in injector.state_dict().keys()
            )
    return frozenset(keys)


def _save_shard(
    output_dir: Path,
    filename: str,
    tensors: Mapping[str, torch.Tensor],
    weight_map: dict[str, str],
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise CompleteTransplantError("safetensors is required") from exc
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = {}
    for key, tensor in tensors.items():
        if key in weight_map:
            raise CompleteTransplantError(f"duplicate target tensor: {key}")
        if not bool(torch.isfinite(tensor.float()).all()):
            raise CompleteTransplantError(f"non-finite target tensor: {key}")
        materialized[key] = tensor.detach().cpu().contiguous()
        weight_map[key] = filename
    save_file(materialized, str(path))


def _translate_mamba_overlay(
    *,
    overlay_dir: Path,
    config: IQHybridConfig,
    output_dir: Path,
    weight_map: dict[str, str],
) -> dict[int, str]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise CompleteTransplantError("safetensors is required") from exc
    manifest = json.loads(
        (overlay_dir / "mamba3_transplant.json").read_text(encoding="utf-8")
    )
    placements = {
        int(item["target_mamba_ordinal"]): item
        for item in manifest["placements"]
    }
    identity = {int(x) for x in manifest["identity_mamba_ordinals"]}
    positions = config.schedule.positions(HybridLayerType.MAMBA3)
    if len(positions) != 32:
        raise CompleteTransplantError("complete config requires 32 Mamba positions")

    provenance: dict[int, str] = {}
    for ordinal, physical in enumerate(positions):
        if ordinal in placements:
            item = placements[ordinal]
            source = load_file(str(overlay_dir / item["shard"]), device="cpu")
            canonical_prefix = f"mamba_layers.{ordinal}."
            target: dict[str, torch.Tensor] = {}
            for key, value in source.items():
                if not key.startswith(canonical_prefix):
                    raise CompleteTransplantError(
                        f"unexpected Mamba overlay key {key}"
                    )
                suffix = key[len(canonical_prefix):]
                if suffix == "norm.weight":
                    target[f"layers.{physical}.norm.weight"] = value
                elif suffix.startswith("core."):
                    target[f"layers.{physical}.mamba.{suffix}"] = value
                else:
                    raise CompleteTransplantError(
                        f"unsupported Mamba overlay suffix {suffix}"
                    )
            filename = f"model-mamba-{ordinal:03d}.safetensors"
            _save_shard(output_dir, filename, target, weight_map)
            provenance[physical] = f"{MAMBA3_MIMO_15B_REPO}:layer:{item['source_layer']}"
        elif ordinal in identity:
            filename = f"model-mamba-{ordinal:03d}.safetensors"
            _save_shard(
                output_dir,
                filename,
                _identity_mamba_state(
                    physical_layer=physical,
                    config=config.mamba3,
                ),
                weight_map,
            )
            provenance[physical] = "recipient_native_identity"
        else:
            raise CompleteTransplantError(
                f"Mamba ordinal {ordinal} is neither transplanted nor identity"
            )
    return provenance


def _copy_tokenizer_assets(gpt_oss_original_dir: Path, output_dir: Path) -> None:
    root = gpt_oss_original_dir.parent
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
        "USAGE_POLICY",
    ):
        source = root / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def compile_complete_iq_checkpoint(
    *,
    gpt_oss_original_dir: str | Path,
    mamba3_checkpoint_dir: str | Path,
    output_dir: str | Path,
    gpt_oss_revision: str,
    mamba3_revision: str,
    verify_hashes: bool = True,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = canonical_complete_config()
    gpt = GptOss20BOriginalCheckpoint(
        gpt_oss_original_dir,
        verify_hash=verify_hashes,
    )

    weight_map: dict[str, str] = {}
    provenance: dict[str, object] = {
        "schema_version": 1,
        "method": "direct_cross_architecture_weight_transplant",
        "teacher_student_distillation": False,
        "donors": {
            "language": {
                "repo": GPT_OSS_20B_REPO,
                "revision": gpt_oss_revision,
                "checkpoint_sha256": gpt.sha256,
            },
            "mamba": {
                "repo": MAMBA3_MIMO_15B_REPO,
                "revision": mamba3_revision,
            },
        },
        "recipient_config_fingerprint": config.fingerprint,
        "residual_embedding": {
            "source_width": 2880,
            "target_width": 4096,
            "scale": RESIDUAL_EMBED_SCALE,
            "method": "rms_preserving_scaled_prefix_embedding",
        },
        "layers": {},
        "recipient_native": [
            "block_attnres",
            "reasoning_recurrence",
            "adaptive_halting",
            "stable_latent_moe_shared_expert",
            "csa_hca_compression_gates",
            "csa_index_weighting",
        ],
    }

    globals_state = gpt.globals()
    _save_shard(
        output,
        "model-global.safetensors",
        _global_state(globals_state),
        weight_map,
    )
    del globals_state

    positions = donor_layer_positions(config)
    for donor_layer, (context_physical, moe_physical) in enumerate(positions):
        mode = config.schedule.layers[context_physical]
        attention = gpt.attention(donor_layer)
        _save_shard(
            output,
            f"model-context-{donor_layer:03d}.safetensors",
            _context_state(
                attention,
                mode=mode,
                physical_layer=context_physical,
            ),
            weight_map,
        )
        del attention

        moe = gpt.moe(donor_layer)
        _save_shard(
            output,
            f"model-moe-{donor_layer:03d}.safetensors",
            _moe_state(
                moe,
                physical_layer=moe_physical,
            ),
            weight_map,
        )
        del moe
        provenance["layers"][str(context_physical)] = (
            f"{GPT_OSS_20B_REPO}:attention:{donor_layer}"
        )
        provenance["layers"][str(moe_physical)] = (
            f"{GPT_OSS_20B_REPO}:moe:{donor_layer}"
        )

    with tempfile.TemporaryDirectory(prefix="iq-mamba-overlay-") as temporary:
        overlay = Path(temporary)
        mamba_result = compile_official_mamba3_mimo_15b_transplant(
            checkpoint=mamba3_checkpoint_dir,
            output_dir=overlay,
            checkpoint_revision=mamba3_revision,
            verify_checkpoint_hash=verify_hashes,
        )
        mamba_provenance = _translate_mamba_overlay(
            overlay_dir=overlay,
            config=config,
            output_dir=output,
            weight_map=weight_map,
        )
        provenance["mamba_target_fingerprint"] = mamba_result.target_fingerprint
        for physical, source in mamba_provenance.items():
            provenance["layers"][str(physical)] = source

    _save_shard(
        output,
        "model-native.safetensors",
        _native_controller_state(config),
        weight_map,
    )

    expected_keys = expected_complete_state_keys(config)
    actual_keys = frozenset(weight_map)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise CompleteTransplantError(
            "compiled checkpoint state coverage mismatch: "
            f"missing={missing[:50]} unexpected={unexpected[:50]}"
        )

    config.write_json(str(output / "iq_config.json"))
    _copy_tokenizer_assets(Path(gpt_oss_original_dir), output)

    index = {
        "metadata": {
            "total_size": sum(
                (output / filename).stat().st_size
                for filename in sorted(set(weight_map.values()))
            )
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "transplant_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_complete_iq_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> IQHybridForCausalLM:
    """Load a compiled complete IQ checkpoint with strict tensor coverage."""
    root = Path(checkpoint_dir)
    config = IQHybridConfig.from_json(str(root / "iq_config.json"))
    model = IQHybridForCausalLM(config, dtype=dtype, device=device)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise CompleteTransplantError("safetensors is required") from exc
    try:
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CompleteTransplantError("invalid checkpoint index") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CompleteTransplantError("checkpoint index has no weight_map")

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    targets = {**parameters, **buffers}
    expected = set(model.state_dict().keys())
    if set(weight_map) != expected:
        missing = sorted(expected - set(weight_map))
        unexpected = sorted(set(weight_map) - expected)
        raise CompleteTransplantError(
            f"checkpoint coverage mismatch: missing={missing[:20]} unexpected={unexpected[:20]}"
        )

    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(str(filename), []).append(str(key))

    with torch.no_grad():
        for filename, keys in sorted(by_file.items()):
            shard = load_file(str(root / filename), device="cpu")
            if set(shard) != set(keys):
                raise CompleteTransplantError(
                    f"shard/index disagreement for {filename}"
                )
            for key, value in shard.items():
                target = targets[key]
                if tuple(target.shape) != tuple(value.shape):
                    raise CompleteTransplantError(
                        f"shape mismatch {key}: checkpoint={tuple(value.shape)} "
                        f"model={tuple(target.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))
    model.eval()
    return model
