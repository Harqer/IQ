from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json

import numpy as np

from iq_model import HybridLayerType, IQHybridConfig

from .calibration import solve_layer_correspondence
from .capture_runner import ActivationBundle
from .complete_transplant import donor_layer_positions
from .glm53_direct import (
    SubcloningMap,
    fit_importance_subcloning_map,
    fit_mla_compressed_subspace,
)
from .glm53_moe import router_usage_from_topk
from .transport import (
    CoordinateMap,
    fit_ridge_coordinate_map,
    load_coordinate_map,
    save_coordinate_map,
)


class GLM53CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GLM53StageCalibration:
    stage: int
    source_layer: int
    context_physical_layer: int
    moe_physical_layer: int
    residual_map: CoordinateMap
    compressed_kv_map: CoordinateMap
    latent_map: CoordinateMap
    expert_usage: np.ndarray | None = None
    dense_intermediate_map: CoordinateMap | None = None

    def __post_init__(self) -> None:
        if min(
            self.stage,
            self.source_layer,
            self.context_physical_layer,
            self.moe_physical_layer,
        ) < 0:
            raise GLM53CalibrationError("stage/layer ids must be non-negative")
        if (self.expert_usage is None) == (self.dense_intermediate_map is None):
            raise GLM53CalibrationError(
                "each stage must be exactly one of sparse-MoE or dense-MLP"
            )
        if self.expert_usage is not None:
            usage = np.asarray(self.expert_usage, dtype=np.float64)
            if usage.ndim != 1 or not np.isfinite(usage).all():
                raise GLM53CalibrationError(
                    "expert_usage must be a finite rank-1 vector"
                )
            if np.any(usage < 0) or float(usage.sum()) <= 0:
                raise GLM53CalibrationError(
                    "expert_usage must be non-negative and non-empty"
                )


@dataclass(frozen=True)
class GLM53CalibrationSolution:
    stages: tuple[GLM53StageCalibration, ...]
    source_layers: int
    target_config_fingerprint: str

    def __post_init__(self) -> None:
        if self.source_layers <= 0 or not self.target_config_fingerprint:
            raise GLM53CalibrationError("invalid calibration solution metadata")
        if not self.stages:
            raise GLM53CalibrationError("calibration solution has no stages")
        stage_ids = [stage.stage for stage in self.stages]
        source_ids = [stage.source_layer for stage in self.stages]
        if stage_ids != list(range(len(self.stages))):
            raise GLM53CalibrationError(
                "calibration stages must be contiguous from zero"
            )
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise GLM53CalibrationError(
                "source layer correspondence must be monotonic and unique"
            )

    @property
    def source_layer_map(self) -> Mapping[int, int]:
        return {stage.stage: stage.source_layer for stage in self.stages}

    def write(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for stage in self.stages:
            prefix = root / f"stage-{stage.stage:02d}"
            save_coordinate_map(stage.residual_map, str(prefix) + "-residual")
            save_coordinate_map(stage.compressed_kv_map, str(prefix) + "-kv")
            save_coordinate_map(stage.latent_map, str(prefix) + "-latent")
            dense_name: str | None = None
            if stage.dense_intermediate_map is not None:
                dense_name = f"stage-{stage.stage:02d}-dense-intermediate"
                save_coordinate_map(
                    stage.dense_intermediate_map,
                    root / dense_name,
                )
            usage_name: str | None = None
            if stage.expert_usage is not None:
                usage_name = f"stage-{stage.stage:02d}-expert-usage.npy"
                np.save(root / usage_name, np.asarray(stage.expert_usage, dtype=np.float64))
            records.append(
                {
                    "stage": stage.stage,
                    "source_layer": stage.source_layer,
                    "context_physical_layer": stage.context_physical_layer,
                    "moe_physical_layer": stage.moe_physical_layer,
                    "residual_map": f"stage-{stage.stage:02d}-residual",
                    "compressed_kv_map": f"stage-{stage.stage:02d}-kv",
                    "latent_map": f"stage-{stage.stage:02d}-latent",
                    "expert_usage": usage_name,
                    "dense_intermediate_map": dense_name,
                }
            )
        manifest = {
            "schema_version": 1,
            "source_layers": self.source_layers,
            "target_config_fingerprint": self.target_config_fingerprint,
            "stages": records,
        }
        path = root / "glm53_calibration.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "GLM53CalibrationSolution":
        root = Path(directory)
        path = root / "glm53_calibration.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GLM53CalibrationError(
                f"invalid GLM calibration manifest: {path}"
            ) from exc
        if raw.get("schema_version") != 1:
            raise GLM53CalibrationError(
                f"unsupported GLM calibration schema: {raw.get('schema_version')}"
            )
        stages: list[GLM53StageCalibration] = []
        for record in raw.get("stages", []):
            usage = (
                np.load(root / record["expert_usage"])
                if record.get("expert_usage")
                else None
            )
            dense = (
                load_coordinate_map(root / record["dense_intermediate_map"])
                if record.get("dense_intermediate_map")
                else None
            )
            stages.append(
                GLM53StageCalibration(
                    stage=int(record["stage"]),
                    source_layer=int(record["source_layer"]),
                    context_physical_layer=int(record["context_physical_layer"]),
                    moe_physical_layer=int(record["moe_physical_layer"]),
                    residual_map=load_coordinate_map(root / record["residual_map"]),
                    compressed_kv_map=load_coordinate_map(
                        root / record["compressed_kv_map"]
                    ),
                    latent_map=load_coordinate_map(root / record["latent_map"]),
                    expert_usage=usage,
                    dense_intermediate_map=dense,
                )
            )
        return cls(
            stages=tuple(stages),
            source_layers=int(raw["source_layers"]),
            target_config_fingerprint=str(raw["target_config_fingerprint"]),
        )


def solve_glm53_calibration(
    source_fit: ActivationBundle,
    target_fit: ActivationBundle,
    source_validation: ActivationBundle,
    target_validation: ActivationBundle,
    *,
    target_config: IQHybridConfig,
    source_layers: int,
    source_first_dense_layers: int,
    source_num_experts: int,
    ridge: float = 1e-3,
    measurements: int = 256,
    seed: int = 0,
    depth_prior: float = 0.05,
) -> GLM53CalibrationSolution:
    if source_layers <= 0:
        raise GLM53CalibrationError("source_layers must be positive")
    if source_first_dense_layers < 0 or source_first_dense_layers > source_layers:
        raise GLM53CalibrationError("invalid source_first_dense_layers")
    if source_num_experts <= 0:
        raise GLM53CalibrationError("source_num_experts must be positive")
    if target_config.compressed_context is None or target_config.stable_moe is None:
        raise GLM53CalibrationError(
            "GLM transfer requires compressed context and Stable LatentMoE"
        )

    stage_positions = donor_layer_positions(target_config)
    source_residuals = {
        layer: source_fit.require(f"layer.{layer}.residual_in")
        for layer in range(source_layers)
    }
    target_residuals = {
        stage: target_fit.require(
            f"layer.{context_physical}.residual_in"
        )
        for stage, (context_physical, _) in enumerate(stage_positions)
    }
    correspondence = solve_layer_correspondence(
        source_residuals,
        target_residuals,
        measurements=measurements,
        seed=seed,
        depth_prior=depth_prior,
    )

    stages: list[GLM53StageCalibration] = []
    target_latent = target_config.stable_moe.latent_size
    target_expert_intermediate = target_config.stable_moe.expert_intermediate_size
    target_head = target_config.compressed_context.head_dim
    rope = target_config.compressed_context.partial_rotary_dim
    target_nonrotary = target_head - rope
    if target_nonrotary <= 0:
        raise GLM53CalibrationError(
            "GLM MLA transfer requires a non-rotary target candidate subspace"
        )

    for stage, (context_physical, moe_physical) in enumerate(stage_positions):
        source_layer = int(correspondence.mapping[stage])
        source_space = f"layer.{source_layer}.residual_in"
        target_space = f"layer.{context_physical}.residual_in"
        residual_map = fit_ridge_coordinate_map(
            source_fit.require(source_space),
            target_fit.require(target_space),
            ridge=ridge,
            source_space=f"glm53.{source_space}",
            target_space=f"iq.{target_space}",
            validation_source=source_validation.require(source_space),
            validation_target=target_validation.require(target_space),
        )

        kv_map = fit_mla_compressed_subspace(
            source_fit.require(f"layer.{source_layer}.compressed_kv"),
            kv_lora_rank=512,
            rope_dim=rope,
            target_latent_dim=target_nonrotary,
        )
        latent: SubcloningMap = fit_importance_subcloning_map(
            source_fit.require(f"layer.{source_layer}.mlp_in"),
            target_features=target_latent,
        )

        expert_usage: np.ndarray | None = None
        dense_intermediate: CoordinateMap | None = None
        if source_layer < source_first_dense_layers:
            dense = fit_importance_subcloning_map(
                source_fit.require(f"layer.{source_layer}.mlp_hidden"),
                target_features=target_expert_intermediate,
            )
            dense_intermediate = dense.coordinate_map
        else:
            expert_usage = router_usage_from_topk(
                [source_fit.require(f"layer.{source_layer}.router_topk").numpy()],
                num_experts=source_num_experts,
            )

        stages.append(
            GLM53StageCalibration(
                stage=stage,
                source_layer=source_layer,
                context_physical_layer=context_physical,
                moe_physical_layer=moe_physical,
                residual_map=residual_map,
                compressed_kv_map=kv_map,
                latent_map=latent.coordinate_map,
                expert_usage=expert_usage,
                dense_intermediate_map=dense_intermediate,
            )
        )

    return GLM53CalibrationSolution(
        stages=tuple(stages),
        source_layers=source_layers,
        target_config_fingerprint=target_config.fingerprint,
    )
