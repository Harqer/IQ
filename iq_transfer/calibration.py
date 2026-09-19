from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import numpy as np

from .gqa_transport import GQAProjection, fit_gqa_group_projection
from .phi_pipeline import PhiLayerMapIds
from .shadows import FunctionalShadow, MeasurementPlan, extract_shadow, match_layers_monotonic
from .transport import CoordinateMap, fit_ridge_coordinate_map


class CalibrationError(RuntimeError):
    pass


class CalibrationSplit(str, Enum):
    MAP_FIT = "map_fit"
    MAP_VALIDATION = "map_validation"
    TRANSFER_VALIDATION = "transfer_validation"


@dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    split: CalibrationSplit
    category: str
    content_sha256: str
    source_id: str
    byte_start: int = 0
    byte_end: int | None = None

    def __post_init__(self) -> None:
        for name in ("record_id", "category", "content_sha256", "source_id"):
            if not str(getattr(self, name)).strip():
                raise CalibrationError(f"{name} must be non-empty")
        if self.byte_start < 0:
            raise CalibrationError("byte_start must be non-negative")
        if self.byte_end is not None and self.byte_end <= self.byte_start:
            raise CalibrationError("byte_end must be greater than byte_start")


@dataclass(frozen=True)
class CalibrationManifest:
    tokenizer_hash: str
    records: tuple[CalibrationRecord, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise CalibrationError(f"unsupported calibration schema: {self.schema_version}")
        if not self.tokenizer_hash.strip():
            raise CalibrationError("tokenizer_hash must be non-empty")
        if not self.records:
            raise CalibrationError("calibration manifest must contain records")
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise CalibrationError("calibration record ids must be unique")
        present = {record.split for record in self.records}
        required = {
            CalibrationSplit.MAP_FIT,
            CalibrationSplit.MAP_VALIDATION,
            CalibrationSplit.TRANSFER_VALIDATION,
        }
        missing = sorted(split.value for split in required - present)
        if missing:
            raise CalibrationError(f"calibration manifest is missing splits: {', '.join(missing)}")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_hash": self.tokenizer_hash,
            "records": [
                {
                    **asdict(record),
                    "split": record.split.value,
                }
                for record in self.records
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibrationManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(raw.get("schema_version", -1)) != 1:
            raise CalibrationError(f"unsupported calibration schema: {raw.get('schema_version')}")
        records = tuple(
            CalibrationRecord(
                record_id=str(item["record_id"]),
                split=CalibrationSplit(str(item["split"])),
                category=str(item["category"]),
                content_sha256=str(item["content_sha256"]),
                source_id=str(item["source_id"]),
                byte_start=int(item.get("byte_start", 0)),
                byte_end=int(item["byte_end"]) if item.get("byte_end") is not None else None,
            )
            for item in raw["records"]
        )
        return cls(tokenizer_hash=str(raw["tokenizer_hash"]), records=records, schema_version=1)


@dataclass(frozen=True)
class ActivationPair:
    source_fit: Any
    target_fit: Any
    source_validation: Any
    target_validation: Any
    source_space: str
    target_space: str

    def __post_init__(self) -> None:
        if not self.source_space.strip() or not self.target_space.strip():
            raise CalibrationError("activation-pair space names must be non-empty")


@dataclass(frozen=True)
class PhiLayerCalibration:
    attn_in: ActivationPair
    source_k_fit: Any
    attn_out: ActivationPair
    mlp_in: ActivationPair
    mlp_hidden: ActivationPair
    mlp_out: ActivationPair


@dataclass(frozen=True)
class PhiLayerMapSolution:
    map_ids: PhiLayerMapIds
    maps: Mapping[str, CoordinateMap]
    gqa_projection: GQAProjection


@dataclass(frozen=True)
class LayerCorrespondence:
    mapping: Mapping[int, int]
    source_shadows: Mapping[int, FunctionalShadow]
    target_shadows: Mapping[int, FunctionalShadow]


def solve_activation_pair(pair: ActivationPair, *, ridge: float = 1e-3) -> CoordinateMap:
    return fit_ridge_coordinate_map(
        pair.source_fit,
        pair.target_fit,
        ridge=ridge,
        source_space=pair.source_space,
        target_space=pair.target_space,
        validation_source=pair.source_validation,
        validation_target=pair.target_validation,
    )


def solve_layer_correspondence(
    source_activations: Mapping[int, Any],
    target_activations: Mapping[int, Any],
    *,
    measurements: int = 256,
    seed: int = 0,
    depth_prior: float = 0.05,
) -> LayerCorrespondence:
    if not source_activations or not target_activations:
        raise CalibrationError("source and target layer activations must be non-empty")
    source_layers = sorted(source_activations)
    target_layers = sorted(target_activations)
    if len(source_layers) < len(target_layers):
        raise CalibrationError("donor must expose at least as many calibration layers as the recipient")

    sample_count: int | None = None
    for collection in (source_activations, target_activations):
        for layer, activations in collection.items():
            array = np.asarray(activations)
            if array.ndim < 2:
                raise CalibrationError(f"layer {layer} activations must be at least rank-2")
            flattened_samples = int(np.prod(array.shape[:-1]))
            if sample_count is None:
                sample_count = flattened_samples
            elif flattened_samples != sample_count:
                raise CalibrationError("all layer activations must have the same flattened sample count")
    assert sample_count is not None
    plan = MeasurementPlan(sample_count=sample_count, measurements=measurements, seed=seed)
    source_shadows = {
        layer: extract_shadow(source_activations[layer], plan, layer=layer) for layer in source_layers
    }
    target_shadows = {
        layer: extract_shadow(target_activations[layer], plan, layer=layer) for layer in target_layers
    }
    mapping = match_layers_monotonic(source_shadows, target_shadows, depth_prior=depth_prior)
    return LayerCorrespondence(mapping=mapping, source_shadows=source_shadows, target_shadows=target_shadows)


def solve_phi_layer_maps(
    calibration: PhiLayerCalibration,
    *,
    target_layer: int,
    source_q_heads: int,
    source_kv_heads: int,
    target_q_heads: int,
    target_kv_heads: int,
    head_dim: int,
    ridge: float = 1e-3,
) -> PhiLayerMapSolution:
    if target_layer < 0:
        raise CalibrationError("target_layer must be non-negative")
    prefix = f"layer.{target_layer}"

    attn_in = solve_activation_pair(calibration.attn_in, ridge=ridge)
    gqa = fit_gqa_group_projection(
        calibration.source_k_fit,
        source_q_heads=source_q_heads,
        source_kv_heads=source_kv_heads,
        target_q_heads=target_q_heads,
        target_kv_heads=target_kv_heads,
        head_dim=head_dim,
        source_space_prefix=f"donor.{prefix}",
        target_space_prefix=f"iq.{prefix}",
    )
    attn_out = solve_activation_pair(calibration.attn_out, ridge=ridge)
    mlp_in = solve_activation_pair(calibration.mlp_in, ridge=ridge)
    mlp_hidden = solve_activation_pair(calibration.mlp_hidden, ridge=ridge)
    mlp_out = solve_activation_pair(calibration.mlp_out, ridge=ridge)

    ids = PhiLayerMapIds(
        attn_in=f"{prefix}.attn_in",
        q=f"{prefix}.q",
        kv=f"{prefix}.kv",
        attn_out=f"{prefix}.attn_out",
        mlp_in=f"{prefix}.mlp_in",
        mlp_hidden=f"{prefix}.mlp_hidden",
        mlp_out=f"{prefix}.mlp_out",
    )
    maps: dict[str, CoordinateMap] = {
        ids.attn_in: attn_in,
        ids.q: gqa.q_map,
        ids.kv: gqa.kv_map,
        ids.attn_out: attn_out,
        ids.mlp_in: mlp_in,
        ids.mlp_hidden: mlp_hidden,
        ids.mlp_out: mlp_out,
    }
    return PhiLayerMapSolution(map_ids=ids, maps=maps, gqa_projection=gqa)


def merge_coordinate_maps(*collections: Mapping[str, CoordinateMap]) -> dict[str, CoordinateMap]:
    result: dict[str, CoordinateMap] = {}
    for collection in collections:
        for map_id, coordinate_map in collection.items():
            if not str(map_id).strip():
                raise CalibrationError("coordinate map ids must be non-empty")
            if map_id in result:
                raise CalibrationError(f"duplicate coordinate map id: {map_id}")
            result[map_id] = coordinate_map
    return result
