"""Read full-78 GLM calibration metadata eagerly, matrices one stage at a time.

The original 78-stage WARM map files exceed 18 GiB. Eagerly loading them
on a 30 GiB L4 worker would leave an unsafe RAM margin. This lazy view
preserves GLM53StageCalibration validation and drops previous stage arrays
as the streaming compiler advances to the next layer.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .glm53_calibration import GLM53StageCalibration, GLM53CalibrationError
from .transport import load_coordinate_map


class _Stages(Sequence[GLM53StageCalibration]):
    def __init__(self, root: Path, records: list[dict]):
        self.root, self.records = root, records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return tuple(self[i] for i in range(*item.indices(len(self))))
        record = self.records[item]
        usage = (
            np.load(self.root / record["expert_usage"])
            if record.get("expert_usage") else None
        )
        dense = (
            load_coordinate_map(self.root / record["dense_intermediate_map"])
            if record.get("dense_intermediate_map") else None
        )
        return GLM53StageCalibration(
            stage=int(record["stage"]),
            source_layer=int(record["source_layer"]),
            context_physical_layer=int(record["context_physical_layer"]),
            moe_physical_layer=int(record["moe_physical_layer"]),
            residual_map=load_coordinate_map(self.root / record["residual_map"]),
            attention_input_map=load_coordinate_map(
                self.root / record["attention_input_map"]
            ),
            moe_input_map=load_coordinate_map(self.root / record["moe_input_map"]),
            compressed_kv_map=load_coordinate_map(
                self.root / record["compressed_kv_map"]
            ),
            latent_map=load_coordinate_map(self.root / record["latent_map"]),
            expert_usage=usage,
            dense_intermediate_map=dense,
        )


class LazyGLM53Calibration:
    """Duck-compatible read-only calibration view for streaming compilation."""

    def __init__(self, directory: str | Path):
        root = Path(directory)
        try:
            manifest = json.loads(
                (root / "glm53_calibration.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise GLM53CalibrationError("invalid saved GLM calibration manifest") from exc
        if manifest.get("schema_version") != 2:
            raise GLM53CalibrationError("unsupported saved calibration schema")
        records = manifest.get("stages")
        if not isinstance(records, list) or not records:
            raise GLM53CalibrationError("missing saved calibration stages")
        stages = tuple(int(r["stage"]) for r in records)
        sources = tuple(int(r["source_layer"]) for r in records)
        if stages != tuple(range(len(records))) or len(set(sources)) != len(sources):
            raise GLM53CalibrationError("invalid ordered/unique saved GLM stage maps")
        self.stages = _Stages(root, records)
        self.source_layers = int(manifest["source_layers"])
        self.target_config_fingerprint = str(manifest["target_config_fingerprint"])
        self.calibration_mode = str(manifest.get("calibration_mode", "activation_refined"))
        self.source_layer_map = {a: b for a, b in zip(stages, sources)}

    def stage(self, position: int) -> GLM53StageCalibration:
        return self.stages[position]
