from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json

from iq_model import HybridLayerType, IQHybridConfig

from .complete_transplant import (
    canonical_glm53_config,
    donor_layer_positions,
)
from .glm53 import GLM53Inspector
from .glm53_calibration import (
    GLM53CalibrationSolution,
    _bootstrap_source_mapping,
)
from .glm53_moe import select_experts_by_usage


class GLM53ShardPlanError(RuntimeError):
    pass


def load_safetensors_weight_map(
    index_path: str | Path,
) -> dict[str, str]:
    path = Path(index_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GLM53ShardPlanError(
            f"invalid safetensors index: {path}"
        ) from exc
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise GLM53ShardPlanError(
            "safetensors index is missing weight_map"
        )
    return {str(k): str(v) for k, v in weight_map.items()}


def _shards_for_keys(
    weight_map: Mapping[str, str],
    keys: set[str],
) -> tuple[str, ...]:
    missing = sorted(key for key in keys if key not in weight_map)
    if missing:
        raise GLM53ShardPlanError(
            f"checkpoint index is missing required tensors: {missing[:20]}"
        )
    return tuple(sorted({weight_map[key] for key in keys}))


def _bootstrap_tensor_keys(
    *,
    inspector: GLM53Inspector,
    target_config: IQHybridConfig,
) -> set[str]:
    stage_positions = donor_layer_positions(target_config)
    mapping = _bootstrap_source_mapping(
        source_layers=inspector.config.num_hidden_layers,
        stage_positions=stage_positions,
        target_config=target_config,
        source_indexer_types=inspector.layout.indexer_types,
    )
    keys: set[str] = set()
    for source_layer in mapping.values():
        prefix = f"model.layers.{source_layer}"
        keys.update(
            {
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.o_proj.weight",
            }
        )
        if inspector.layout.indexer_types[source_layer] == "full":
            indexer = f"{prefix}.self_attn.indexer"
            keys.update(
                {
                    f"{indexer}.wk.weight",
                    f"{indexer}.weights_proj.weight",
                }
            )
        mlp = f"{prefix}.mlp"
        if source_layer < inspector.layout.first_k_dense_replace:
            keys.update(
                {
                    f"{mlp}.gate_proj.weight",
                    f"{mlp}.up_proj.weight",
                    f"{mlp}.down_proj.weight",
                }
            )
        else:
            shared = f"{mlp}.shared_experts"
            keys.update(
                {
                    f"{mlp}.gate.weight",
                    f"{mlp}.gate.e_score_correction_bias",
                    f"{shared}.gate_proj.weight",
                    f"{shared}.up_proj.weight",
                    f"{shared}.down_proj.weight",
                }
            )
    return keys
def plan_glm53_bootstrap_shards(
    *,
    config_path: str | Path,
    index_path: str | Path,
    target_config: IQHybridConfig | None = None,
) -> tuple[str, ...]:
    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inspector = GLM53Inspector.from_config_mapping(config_data)
    target = target_config or canonical_glm53_config()
    keys = _bootstrap_tensor_keys(
        inspector=inspector,
        target_config=target,
    )
    return _shards_for_keys(load_safetensors_weight_map(index_path), keys)


def _compile_tensor_keys(
    *,
    calibration: GLM53CalibrationSolution,
    inspector: GLM53Inspector,
    target_config: IQHybridConfig,
) -> set[str]:
    if target_config.stable_moe is None:
        raise GLM53ShardPlanError("target config is missing Stable LatentMoE")
    keys: set[str] = set()
    stable = target_config.stable_moe
    for stage in calibration.stages:
        source_layer = stage.source_layer
        prefix = f"model.layers.{source_layer}"
        attn = f"{prefix}.self_attn"
        keys.update(
            {
                f"{attn}.q_a_proj.weight",
                f"{attn}.q_a_layernorm.weight",
                f"{attn}.q_b_proj.weight",
                f"{attn}.kv_a_proj_with_mqa.weight",
                f"{attn}.kv_b_proj.weight",
                f"{attn}.o_proj.weight",
            }
        )
        mode = target_config.schedule.layers[stage.context_physical_layer]
        if mode is HybridLayerType.CSA:
            indexer = f"{attn}.indexer"
            keys.update(
                {
                    f"{indexer}.wq_b.weight",
                    f"{indexer}.wk.weight",
                    f"{indexer}.weights_proj.weight",
                    f"{indexer}.k_norm.weight",
                    f"{indexer}.k_norm.bias",
                }
            )
        mlp = f"{prefix}.mlp"
        if source_layer < inspector.layout.first_k_dense_replace:
            keys.update(
                {
                    f"{mlp}.gate_proj.weight",
                    f"{mlp}.up_proj.weight",
                    f"{mlp}.down_proj.weight",
                }
            )
        else:
            if stage.expert_usage is None:
                raise GLM53ShardPlanError(
                    f"sparse source layer {source_layer} is missing expert usage"
                )
            selected = select_experts_by_usage(
                stage.expert_usage,
                target_experts=stable.num_experts,
            )
            keys.update(
                {
                    f"{mlp}.gate.weight",
                    f"{mlp}.gate.e_score_correction_bias",
                    f"{mlp}.shared_experts.gate_proj.weight",
                    f"{mlp}.shared_experts.up_proj.weight",
                    f"{mlp}.shared_experts.down_proj.weight",
                }
            )
            for expert in selected:
                expert_prefix = f"{mlp}.experts.{expert}"
                keys.update(
                    {
                        f"{expert_prefix}.gate_proj.weight",
                        f"{expert_prefix}.up_proj.weight",
                        f"{expert_prefix}.down_proj.weight",
                    }
                )
    return keys


def plan_glm53_compile_shards(
    *,
    config_path: str | Path,
    index_path: str | Path,
    calibration_dir: str | Path,
    target_config: IQHybridConfig | None = None,
) -> tuple[str, ...]:
    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inspector = GLM53Inspector.from_config_mapping(config_data)
    calibration = GLM53CalibrationSolution.load(calibration_dir)
    target = target_config or canonical_glm53_config()
    if calibration.target_config_fingerprint != target.fingerprint:
        raise GLM53ShardPlanError(
            "calibration target fingerprint does not match target config"
        )
    keys = _compile_tensor_keys(
        calibration=calibration,
        inspector=inspector,
        target_config=target,
    )
    return _shards_for_keys(load_safetensors_weight_map(index_path), keys)
