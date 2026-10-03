from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json
import shutil
import tempfile

import numpy as np
import torch

from iq_model import HybridLayerType, IQHybridConfig

from .checkpoint import SafetensorsSource
from .complete_transplant import (
    CompleteTransplantError,
    _native_controller_state,
    _save_shard,
    _translate_mamba_overlay,
    canonical_glm53_config,
    expected_complete_state_keys,
)
from .glm53 import GLM53Inspector
from .glm53_calibration import GLM53CalibrationSolution, GLM53StageCalibration
from .glm53_direct import GLM53MLALayout, transform_glm53_mla
from .glm53_dsa import transform_glm53_dsa_indexer
from .glm53_job import validate_glm53_donor
from .glm53_moe import (
    ExpertWeights,
    select_experts_by_usage,
    transform_glm53_dense_mlp,
    transform_glm53_moe_selected,
)
from .mamba3_direct import (
    compile_official_mamba3_mimo_15b_transplant,
    load_official_mamba3_foundation_globals,
)
from .transport import CoordinateMap


class GLM53CompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class GLM53CompileResult:
    output_dir: Path
    donor_fingerprint: str
    target_fingerprint: str
    source_layer_map: Mapping[int, int]


def _tensor(source: SafetensorsSource, key: str) -> torch.Tensor:
    return source.get(key)


def _as_tensor(value: np.ndarray | torch.Tensor, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
    else:
        out = torch.from_numpy(np.asarray(value))
    if not bool(torch.isfinite(out.float()).all()):
        raise GLM53CompileError("transformed tensor contains non-finite values")
    return out.to(dtype=dtype).contiguous()


def _identity_map(width: int, *, space: str) -> CoordinateMap:
    return CoordinateMap(
        matrix=np.eye(width, dtype=np.float64),
        ridge=1e-12,
        source_space=space,
        target_space=space,
        diagnostics=None,
    )


def _transport_input_only(weight: torch.Tensor, residual_map: CoordinateMap) -> torch.Tensor:
    array = weight.detach().cpu().double().numpy()
    transformed = array @ np.linalg.pinv(residual_map.matrix).T
    return _as_tensor(transformed, dtype=weight.dtype)


def _context_state(
    *,
    source: SafetensorsSource,
    inspector: GLM53Inspector,
    stage: GLM53StageCalibration,
    config: IQHybridConfig,
) -> dict[str, torch.Tensor]:
    assert config.compressed_context is not None
    source_layer = stage.source_layer
    physical = stage.context_physical_layer
    mode = config.schedule.layers[physical]
    if mode not in {HybridLayerType.CSA, HybridLayerType.HCA}:
        raise GLM53CompileError("GLM context stage must target CSA or HCA")

    prefix = f"model.layers.{source_layer}"
    target = f"layers.{physical}.attention"
    c = config.compressed_context
    layout = inspector.layout

    mla = transform_glm53_mla(
        q_b_weight=_tensor(source, f"{prefix}.self_attn.q_b_proj.weight"),
        kv_a_weight=_tensor(source, f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
        kv_b_weight=_tensor(source, f"{prefix}.self_attn.kv_b_proj.weight"),
        o_weight=_tensor(source, f"{prefix}.self_attn.o_proj.weight"),
        residual_input_map=stage.attention_input_map,
        residual_output_map=stage.residual_map,
        compressed_kv_map=stage.compressed_kv_map,
        layout=GLM53MLALayout(
            source_hidden_size=inspector.config.hidden_size,
            num_heads=inspector.config.num_attention_heads,
            q_lora_rank=layout.q_lora_rank,
            kv_lora_rank=layout.kv_lora_rank,
            qk_nope_head_dim=layout.qk_nope_head_dim,
            qk_rope_head_dim=layout.qk_rope_head_dim,
            v_head_dim=layout.v_head_dim,
            target_hidden_size=config.model.hidden_size,
            target_head_dim=c.head_dim,
            output_groups=c.o_groups,
            output_rank=c.o_lora_rank,
        ),
    )

    q_a = _transport_input_only(
        _tensor(source, f"{prefix}.self_attn.q_a_proj.weight"),
        stage.attention_input_map,
    )
    state: dict[str, torch.Tensor] = {
        f"layers.{physical}.norm.weight": torch.ones(config.model.hidden_size, dtype=torch.bfloat16),
        f"{target}.q_a_proj.weight": q_a,
        f"{target}.q_a_norm.weight": _tensor(
            source, f"{prefix}.self_attn.q_a_layernorm.weight"
        ).to(torch.bfloat16),
        f"{target}.q_b_proj.weight": _as_tensor(mla.q_b_weight),
        f"{target}.kv_proj.weight": _as_tensor(mla.kv_weight),
        f"{target}.kv_norm.weight": torch.ones(
            c.head_dim - c.partial_rotary_dim
            if c.normalize_candidate_nonrotary_only else c.head_dim,
            dtype=torch.bfloat16,
        ),
        f"{target}.compressor_gate_proj.weight": torch.zeros(
            c.head_dim * (2 if mode is HybridLayerType.CSA else 1),
            config.model.hidden_size,
            dtype=torch.bfloat16,
        ),
        f"{target}.compressor_position_bias": torch.zeros(
            c.csa_compress_rate if mode is HybridLayerType.CSA else c.hca_compress_rate,
            c.head_dim * (2 if mode is HybridLayerType.CSA else 1),
            dtype=torch.bfloat16,
        ),
        f"{target}.compressor_kv_norm.weight": torch.ones(
            c.head_dim - c.partial_rotary_dim
            if c.normalize_candidate_nonrotary_only else c.head_dim,
            dtype=torch.bfloat16,
        ),
        f"{target}.sinks": torch.zeros(c.num_attention_heads, dtype=torch.bfloat16),
        f"{target}.output.weight": _as_tensor(mla.output_group_weight),
        f"{target}.output.out_proj.weight": _as_tensor(mla.output_weight),
    }

    kv = _as_tensor(mla.kv_weight)
    if mode is HybridLayerType.CSA:
        state[f"{target}.compressor_kv_proj.weight"] = torch.cat((kv, kv), dim=0)
        if not c.direct_token_indexer:
            raise GLM53CompileError(
                "canonical GLM transfer requires direct_token_indexer=True"
            )
        if inspector.layout.indexer_types[source_layer] != "full":
            raise GLM53CompileError(
                "CSA stage mapped to a GLM shared-indexer layer; calibration must constrain CSA to full indexer layers"
            )
        indexer = f"{prefix}.self_attn.indexer"
        dsa = transform_glm53_dsa_indexer(
            q_weight=_tensor(source, f"{indexer}.wq_b.weight"),
            k_weight=_tensor(source, f"{indexer}.wk.weight"),
            head_weight=_tensor(source, f"{indexer}.weights_proj.weight"),
            k_norm_weight=_tensor(source, f"{indexer}.k_norm.weight"),
            k_norm_bias=_tensor(source, f"{indexer}.k_norm.bias"),
            residual_map=stage.attention_input_map,
            num_heads=c.index_n_heads,
            head_dim=c.index_head_dim,
            rope_dim=c.partial_rotary_dim,
            q_lora_rank=c.q_lora_rank,
        )
        state.update(
            {
                f"{target}.index_kv_proj.weight": _as_tensor(dsa.k_weight),
                f"{target}.index_kv_norm.weight": _as_tensor(dsa.k_norm_weight),
                f"{target}.index_kv_norm.bias": _as_tensor(dsa.k_norm_bias),
                f"{target}.index_q_proj.weight": _as_tensor(dsa.q_weight),
                f"{target}.index_weight_proj.weight": _as_tensor(dsa.head_weight),
            }
        )
    else:
        state[f"{target}.compressor_kv_proj.weight"] = kv

    return state


def _expert(source: SafetensorsSource, prefix: str) -> ExpertWeights:
    return ExpertWeights(
        gate=_tensor(source, f"{prefix}.gate_proj.weight"),
        up=_tensor(source, f"{prefix}.up_proj.weight"),
        down=_tensor(source, f"{prefix}.down_proj.weight"),
    )


def _write_expert(
    state: dict[str, torch.Tensor],
    prefix: str,
    expert: ExpertWeights,
) -> None:
    state[f"{prefix}.gate_proj.weight"] = _as_tensor(expert.gate)
    state[f"{prefix}.up_proj.weight"] = _as_tensor(expert.up)
    state[f"{prefix}.down_proj.weight"] = _as_tensor(expert.down)


def _moe_state(
    *,
    source: SafetensorsSource,
    inspector: GLM53Inspector,
    stage: GLM53StageCalibration,
    config: IQHybridConfig,
) -> dict[str, torch.Tensor]:
    assert config.stable_moe is not None
    source_layer = stage.source_layer
    physical = stage.moe_physical_layer
    donor_prefix = f"model.layers.{source_layer}"
    target = f"layers.{physical}.moe"
    stable = config.stable_moe
    state: dict[str, torch.Tensor] = {
        f"layers.{physical}.norm.weight": torch.ones(config.model.hidden_size, dtype=torch.bfloat16),
        f"{target}.latent_norm.weight": torch.ones(stable.latent_size, dtype=torch.bfloat16),
    }

    if source_layer < inspector.layout.first_k_dense_replace:
        if stage.dense_intermediate_map is None:
            raise GLM53CompileError(
                "dense GLM stage is missing dense_intermediate_map"
            )
        transformed = transform_glm53_dense_mlp(
            dense_expert=_expert(source, f"{donor_prefix}.mlp"),
            target_experts=stable.num_experts,
            residual_map=stage.residual_map,
            latent_map=stage.latent_map,
            intermediate_map=stage.dense_intermediate_map,
            input_map=stage.moe_input_map,
        )
        state[f"{target}.router.weight"] = _as_tensor(transformed.router_weight)
        state[f"{target}.routing_bias"] = _as_tensor(
            transformed.routing_bias, dtype=torch.float32
        )
        state[f"{target}.latent_down.weight"] = _as_tensor(transformed.latent_down_weight)
        state[f"{target}.latent_up.weight"] = _as_tensor(transformed.latent_up_weight)
        for i, expert in enumerate(transformed.routed_experts):
            _write_expert(state, f"{target}.routed_experts.{i}", expert)
        _write_expert(state, f"{target}.shared_experts.0", transformed.shared_expert)
        return state

    if stage.expert_usage is None:
        raise GLM53CompileError("sparse GLM stage is missing captured expert usage")

    intermediate = _identity_map(
        inspector.layout.moe_intermediate_size,
        space=f"glm53.layer.{source_layer}.expert_intermediate",
    )
    selected_ids = select_experts_by_usage(
        stage.expert_usage,
        target_experts=stable.num_experts,
    )
    selected_experts = {
        index: _expert(source, f"{donor_prefix}.mlp.experts.{index}")
        for index in selected_ids
    }
    transformed = transform_glm53_moe_selected(
        router_weight=_tensor(source, f"{donor_prefix}.mlp.gate.weight"),
        routing_bias=_tensor(
            source, f"{donor_prefix}.mlp.gate.e_score_correction_bias"
        ),
        selected_experts=selected_experts,
        shared_expert=_expert(source, f"{donor_prefix}.mlp.shared_experts"),
        expert_usage=stage.expert_usage,
        target_experts=stable.num_experts,
        residual_map=stage.residual_map,
        latent_map=stage.latent_map,
        intermediate_map=intermediate,
        input_map=stage.moe_input_map,
    )
    state[f"{target}.router.weight"] = _as_tensor(transformed.router_weight)
    state[f"{target}.routing_bias"] = _as_tensor(
        transformed.routing_bias, dtype=torch.float32
    )
    state[f"{target}.latent_down.weight"] = _as_tensor(transformed.latent_down_weight)
    state[f"{target}.latent_up.weight"] = _as_tensor(transformed.latent_up_weight)
    for i, expert in enumerate(transformed.routed_experts):
        _write_expert(state, f"{target}.routed_experts.{i}", expert)
    _write_expert(state, f"{target}.shared_experts.0", transformed.shared_expert)
    return state


def _copy_tokenizer_assets(source_dir: Path, output_dir: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
        "README.md",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def compile_glm53_iq_checkpoint(
    *,
    glm53_checkpoint: str | Path,
    calibration_dir: str | Path,
    mamba3_checkpoint: str | Path,
    output_dir: str | Path,
    glm53_revision: str,
    mamba3_revision: str,
    donor_license: str,
    verify_hashes: bool = True,
) -> GLM53CompileResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = canonical_glm53_config()
    calibration = GLM53CalibrationSolution.load(calibration_dir)
    if calibration.target_config_fingerprint != config.fingerprint:
        raise GLM53CompileError(
            "GLM calibration target fingerprint does not match canonical GLM recipient"
        )

    donor = validate_glm53_donor(
        checkpoint=glm53_checkpoint,
        checkpoint_revision=glm53_revision,
        donor_license=donor_license,
        require_bf16=True,
    )
    source = SafetensorsSource(donor.checkpoint_dir)
    config_data = json.loads(
        (donor.checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    inspector = GLM53Inspector.from_config_mapping(config_data)
    if calibration.source_layers != inspector.config.num_hidden_layers:
        raise GLM53CompileError(
            "calibration source layer count does not match GLM checkpoint"
        )

    weight_map: dict[str, str] = {}
    provenance: dict[str, object] = {
        "schema_version": 2,
        "method": "mamba3_foundation_plus_glm53_capability_weight_transfer",
        "foundation": {
            "repo": "state-spaces/mamba3-mimo-1.5b",
            "revision": mamba3_revision,
            "role": "pretrained lexical + recurrent foundation",
            "width_transfer": "2048->4096 exact replication embedding",
        },
        "capability_source": {
            "repo": "zai-org/GLM-5.3-BF16",
            "revision": glm53_revision,
            "fingerprint": donor.manifest.fingerprint,
            "checkpoint_hash": donor.manifest.checkpoint_hash,
            "role": "WARM-remapped context/DSA/MoE capability weights",
        },
        "recipient_config_fingerprint": config.fingerprint,
        "source_layer_map": {
            str(stage.stage): stage.source_layer for stage in calibration.stages
        },
        "context": "MLA compressed-latent refactor + DSA direct-token indexer",
        "moe": "usage-selected expert subcloning into Stable LatentMoE",
        "mamba": "official Mamba-3 MIMO foundation; no GLM QKV-to-recurrence fabrication",
    }

    # Mamba-3 is the pretrained foundation and owns the lexical boundary.
    # GLM-5.3 never replaces the tokenizer, embedding, final norm, or LM head.
    foundation_globals = load_official_mamba3_foundation_globals(
        checkpoint=mamba3_checkpoint,
        checkpoint_revision=mamba3_revision,
        verify_checkpoint_hash=verify_hashes,
        target_hidden_size=config.model.hidden_size,
    )
    if foundation_globals["embed_tokens.weight"].shape[0] != config.model.vocab_size:
        raise GLM53CompileError(
            "Mamba foundation vocabulary does not match canonical IQ config"
        )
    _save_shard(
        output,
        "model-global.safetensors",
        {
            key: value.to(torch.bfloat16)
            for key, value in foundation_globals.items()
        },
        weight_map,
    )

    for stage in calibration.stages:
        _save_shard(
            output,
            f"model-context-{stage.stage:03d}.safetensors",
            _context_state(
                source=source,
                inspector=inspector,
                stage=stage,
                config=config,
            ),
            weight_map,
        )
        _save_shard(
            output,
            f"model-moe-{stage.stage:03d}.safetensors",
            _moe_state(
                source=source,
                inspector=inspector,
                stage=stage,
                config=config,
            ),
            weight_map,
        )

    with tempfile.TemporaryDirectory(prefix="iq-glm53-mamba-") as temporary:
        overlay = Path(temporary)
        compile_official_mamba3_mimo_15b_transplant(
            checkpoint=mamba3_checkpoint,
            output_dir=overlay,
            checkpoint_revision=mamba3_revision,
            verify_checkpoint_hash=verify_hashes,
        )
        _translate_mamba_overlay(
            overlay_dir=overlay,
            config=config,
            output_dir=output,
            weight_map=weight_map,
        )

    _save_shard(
        output,
        "model-native.safetensors",
        _native_controller_state(config),
        weight_map,
    )

    expected = expected_complete_state_keys(config)
    actual = frozenset(weight_map)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise GLM53CompileError(
            "compiled GLM IQ checkpoint coverage mismatch: "
            f"missing={missing[:60]} unexpected={unexpected[:60]}"
        )

    config.write_json(str(output / "iq_config.json"))
    _copy_tokenizer_assets(Path(mamba3_checkpoint), output)
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
    return GLM53CompileResult(
        output_dir=output,
        donor_fingerprint=donor.manifest.fingerprint,
        target_fingerprint=config.fingerprint,
        source_layer_map=calibration.source_layer_map,
    )
