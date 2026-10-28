from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from iq_model import HybridLayerType, IQHybridConfig

from .checkpoint import SafetensorsSource
from .complete_transplant import (
    _native_controller_state,
    _save_shard,
    _translate_mamba_overlay,
    canonical_glm53_config,
    canonical_glm53_full78_config,
    donor_layer_positions,
    expected_complete_state_keys,
)
from .glm53 import GLM53Inspector
from .glm53_calibration import GLM53CalibrationSolution, GLM53StageCalibration
from .glm53_direct import GLM53MLALayout, transform_glm53_mla
from .glm53_dsa import transform_glm53_dsa_indexer
from .glm53_job import validate_glm53_donor, validate_glm53_streaming_donor
from .foundation_reuse import reuse_frozen_mamba_foundation
from .glm53_moe import (
    ExpertWeights,
    select_experts_by_usage,
    transform_glm53_dense_mlp,
    transform_glm53_moe_selected,
)
from .glm53_stream_capture import HubSafetensorsSource
from .glm53_incremental_moe import save_incremental_moe_stage
from .glm53_resume import register_saved_shard, validate_legacy_partial
from .glm53_provenance import enforce_run_provenance
from .glm53_lazy_calibration import LazyGLM53Calibration
from .glm53_incremental_context import transport_input_projection_rows
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


def _as_tensor(
    value: np.ndarray | torch.Tensor, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
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


def _transport_input_only(
    weight: torch.Tensor, residual_map: CoordinateMap
) -> torch.Tensor:
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

    q_a_key = f"{prefix}.self_attn.q_a_proj.weight"
    q_a = (
        transport_input_projection_rows(
            source=source,
            key=q_a_key,
            residual_map=stage.attention_input_map,
        )
        if isinstance(source, HubSafetensorsSource)
        else _transport_input_only(_tensor(source, q_a_key), stage.attention_input_map)
    )
    state: dict[str, torch.Tensor] = {
        f"layers.{physical}.norm.weight": torch.ones(
            config.model.hidden_size, dtype=torch.bfloat16
        ),
        f"{target}.q_a_proj.weight": q_a,
        f"{target}.q_a_norm.weight": _tensor(
            source, f"{prefix}.self_attn.q_a_layernorm.weight"
        ).to(torch.bfloat16),
        f"{target}.q_b_proj.weight": _as_tensor(mla.q_b_weight),
        f"{target}.kv_proj.weight": _as_tensor(mla.kv_weight),
        f"{target}.kv_norm.weight": torch.ones(
            c.head_dim - c.partial_rotary_dim
            if c.normalize_candidate_nonrotary_only
            else c.head_dim,
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
            if c.normalize_candidate_nonrotary_only
            else c.head_dim,
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
        f"layers.{physical}.norm.weight": torch.ones(
            config.model.hidden_size, dtype=torch.bfloat16
        ),
        f"{target}.latent_norm.weight": torch.ones(
            stable.latent_size, dtype=torch.bfloat16
        ),
    }

    if source_layer < inspector.layout.first_k_dense_replace:
        if stage.dense_intermediate_map is None:
            raise GLM53CompileError("dense GLM stage is missing dense_intermediate_map")
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
        state[f"{target}.latent_down.weight"] = _as_tensor(
            transformed.latent_down_weight
        )
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
            from .glm53_bucket_persistence import durable_output, publish_file
            if durable_output(output_dir):
                publish_file(output_dir, name, source, idempotent=True)
            else:
                shutil.copy2(source, output_dir / name)


def _validate_expanded_calibration(
    calibration: GLM53CalibrationSolution, config: IQHybridConfig
) -> None:
    """Allow prior *weight-only* calibration when only routed capacity grows.

    canonical_glm53_config() supplies identical backbone, dimensions, and
    map geometry for the permitted expert-count variants. Do not silently
    reuse activation-refined or unrelated calibration configurations.
    """
    if calibration.target_config_fingerprint == config.fingerprint:
        return
    original = canonical_glm53_config()
    if (
        config.stable_moe is not None
        and config.stable_moe.num_experts in (64, 128, 256)
        and calibration.calibration_mode == "weight_only"
        and calibration.target_config_fingerprint == original.fingerprint
    ):
        return
    raise GLM53CompileError(
        "GLM calibration fingerprint mismatch: expanded compile only supports "
        "canonical weight-only calibration from the original 32-expert IQ layout"
    )


def compile_glm53_iq_checkpoint(
    *,
    glm53_checkpoint: str | Path,
    calibration_dir: str | Path,
    mamba3_checkpoint: str | Path | None,
    output_dir: str | Path,
    glm53_revision: str,
    mamba3_revision: str,
    donor_license: str,
    verify_hashes: bool = True,
    streaming_source: bool = False,
    glm53_repo_id: str = "zai-org/GLM-5.3-BF16",
    target_config: IQHybridConfig | None = None,
    reuse_foundation_checkpoint: str | Path | None = None,
    resume_existing: bool = False,
) -> GLM53CompileResult:
    if (mamba3_checkpoint is None) == (reuse_foundation_checkpoint is None):
        raise GLM53CompileError(
            "choose exactly one Mamba source: raw donor checkpoint or existing IQ foundation"
        )
    output = Path(output_dir)
    config = target_config if target_config is not None else canonical_glm53_config()
    if (
        config.stable_moe is not None
        and config.stable_moe.num_experts > 32
        and output.exists()
        and (not output.is_dir() or any(output.iterdir()))
        and not resume_existing
    ):
        raise GLM53CompileError(
            "expanded GLM compile requires a NEW empty directory; refusing "
            "to overwrite the original checkpoint or an earlier candidate"
        )
    from .glm53_bucket_persistence import durable_output
    if not durable_output(output):
        output.mkdir(parents=True, exist_ok=True)
    calibration = (LazyGLM53Calibration(calibration_dir) if streaming_source
                   else GLM53CalibrationSolution.load(calibration_dir))
    if resume_existing:
        if not streaming_source or len(calibration.stages) != 78:
            raise GLM53CompileError(
                "restart requires a full78 streaming donor compilation"
            )
        validate_legacy_partial(output)
    if config.stable_moe is None or config.stable_moe.num_experts not in (32, 64, 128, 256):
        raise GLM53CompileError("unsupported GLM routed expert capacity")
    stage_count = len(donor_layer_positions(config))
    if stage_count not in (24, 78):
        raise GLM53CompileError("IQ must map exactly 24 or 78 donor stages")
    if stage_count == 24:
        expected_config = canonical_glm53_config(
            routed_experts=config.stable_moe.num_experts
        )
        if config.fingerprint != expected_config.fingerprint:
            raise GLM53CompileError("24-stage configuration is not canonical")
    if len(calibration.stages) != stage_count:
        raise GLM53CompileError(
            "calibration must cover all target stages before weight transport"
        )
    if stage_count == 78 and [
        calibration.source_layer_map[i] for i in range(78)
    ] != list(range(78)):
        raise GLM53CompileError(
            "full-78 transfer requires exact GLM source layers 0..77"
        )
    _validate_expanded_calibration(calibration, config)

    if streaming_source:
        donor = validate_glm53_streaming_donor(
            checkpoint=glm53_checkpoint,
            checkpoint_revision=glm53_revision,
            donor_license=donor_license,
        )
    else:
        donor = validate_glm53_donor(
            checkpoint=glm53_checkpoint,
            checkpoint_revision=glm53_revision,
            donor_license=donor_license,
            require_bf16=True,
        )
    source = (
        HubSafetensorsSource(
            donor.checkpoint_dir,
            repo_id=glm53_repo_id,
            revision=glm53_revision,
            # Cloud-only compilation needs full dense-prefix GLM MLP tensors
            # (larger than 128 MiB); HTTP requests remain capped at 4 MiB.
            max_tensor_bytes=512 * 1024 * 1024,
            max_request_bytes=16 * 1024 * 1024,
        )
        if streaming_source
        else SafetensorsSource(donor.checkpoint_dir)
    )
    config_data = json.loads(
        (donor.checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    inspector = GLM53Inspector.from_config_mapping(config_data)
    if stage_count == 78:
        official_full_config = canonical_glm53_full78_config(
            source_indexer_types=inspector.layout.indexer_types,
            routed_experts=config.stable_moe.num_experts,
        )
        if config.fingerprint != official_full_config.fingerprint:
            raise GLM53CompileError(
                "full-78 IQ context layer types do not match official GLM indexer layout"
            )
    if calibration.source_layers != inspector.config.num_hidden_layers:
        raise GLM53CompileError(
            "calibration source layer count does not match GLM checkpoint"
        )

    if stage_count == 78 and streaming_source:
        enforce_run_provenance(
            output, Path(calibration_dir),
            glm_revision=glm53_revision,
            mamba_revision=mamba3_revision,
            target_fingerprint=config.fingerprint,
            resume=resume_existing,
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
        "calibration_mode": calibration.calibration_mode,
        "routed_experts_per_sparse_stage": config.stable_moe.num_experts,
        "selected_source_expert_ids": {},
        "source_layer_map": {
            str(i): int(source) for i, source in calibration.source_layer_map.items()
        },
        "context": "MLA compressed-latent refactor + DSA direct-token indexer",
        "moe": "usage-selected expert subcloning into Stable LatentMoE",
        "mamba": "official Mamba-3 MIMO foundation; no GLM QKV-to-recurrence fabrication",
    }

    # Keep the original pretrained Mamba/lexical foundation byte-identical.
    # Using a completed IQ checkpoint avoids reloading/retransplanting 1.5B
    # raw Mamba donor weights for every GLM expert capacity experiment.
    reused_foundation: dict[str, object] | None = None
    if reuse_foundation_checkpoint is not None:
        reused_foundation = reuse_frozen_mamba_foundation(
            original=Path(reuse_foundation_checkpoint),
            output=output,
            expected_source_fingerprint=canonical_glm53_config().fingerprint,
            target_config=config,
            mamba_revision=mamba3_revision,
            glm_revision=glm53_revision,
            weight_map=weight_map,
        )
        provenance["frozen_mamba_foundation_reuse"] = reused_foundation
    elif not (
        resume_existing and register_saved_shard(
            output, "model-global.safetensors", weight_map
        )
    ):
        assert mamba3_checkpoint is not None
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
            {key: value.to(torch.bfloat16) for key, value in foundation_globals.items()},
            weight_map,
        )

    for stage in calibration.stages:
        if stage.source_layer < inspector.layout.first_k_dense_replace:
            provenance["selected_source_expert_ids"][str(stage.stage)] = None
        else:
            if stage.expert_usage is None:
                raise GLM53CompileError(
                    f"source layer {stage.source_layer} missing measured or prior routing usage"
                )
            chosen_ids = select_experts_by_usage(
                stage.expert_usage,
                target_experts=config.stable_moe.num_experts,
            )
            if len(chosen_ids) != config.stable_moe.num_experts:
                raise GLM53CompileError("selected routed expert count mismatch")
            provenance["selected_source_expert_ids"][str(stage.stage)] = list(chosen_ids)
        context_name = f"model-context-{stage.stage:03d}.safetensors"
        if not (
            resume_existing and register_saved_shard(
                output, context_name, weight_map,
                layer=stage.context_physical_layer,
            )
        ):
            _save_shard(
                output,
                context_name,
                _context_state(
                    source=source,
                    inspector=inspector,
                    stage=stage,
                    config=config,
                ),
                weight_map,
            )
        if streaming_source:
            # Persist each transformed expert before reading the next donor
            # expert. Avoid holding all routed experts in one stage dictionary.
            save_incremental_moe_stage(
                source=source,
                inspector=inspector,
                stage=stage,
                config=config,
                output_dir=output,
                weight_map=weight_map,
                resume_existing=resume_existing,
            )
        else:
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
        clear = getattr(source, "clear_ephemeral_shards", None)
        if callable(clear):
            clear(completed_layer=stage.source_layer)

    if reuse_foundation_checkpoint is None:
        assert mamba3_checkpoint is not None
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
                resume_existing=resume_existing,
            )

    if not (
        resume_existing and register_saved_shard(
            output, "model-native.safetensors", weight_map
        )
    ):
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

    from .glm53_bucket_persistence import (
        durable_output, local_stage_dir, publish_file,
        remote_checkpoint_sizes, write_checkpoint_text,
    )
    if durable_output(output):
        with local_stage_dir(output) as td:
            local_config = Path(td) / "iq_config.json"
            config.write_json(str(local_config))
            publish_file(output, "iq_config.json", local_config)
    else:
        config.write_json(str(output / "iq_config.json"))

    _copy_tokenizer_assets(
        Path(reuse_foundation_checkpoint)
        if reuse_foundation_checkpoint is not None else Path(mamba3_checkpoint),
        output,
    )

    files = sorted(set(weight_map.values()))
    if durable_output(output):
        # These are remote Hub metadata, not stale NFS-mounted stat values.
        sizes = remote_checkpoint_sizes(output, files)
        total_size = sum(sizes.values())
    else:
        total_size = sum((output / filename).stat().st_size for filename in files)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_checkpoint_text(
        output, "model.safetensors.index.json",
        json.dumps(index, indent=2, sort_keys=True) + "\n",
    )
    write_checkpoint_text(
        output, "transplant_manifest.json",
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )
    return GLM53CompileResult(
        output_dir=output,
        donor_fingerprint=donor.manifest.fingerprint,
        target_fingerprint=config.fingerprint,
        source_layer_map=calibration.source_layer_map,
    )
