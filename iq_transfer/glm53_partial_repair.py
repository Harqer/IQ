"""Incremental recovery of absent GLM-5.3 calibration layers.

The source of truth is an existing all-78 manifest. All previously present
calibration assets (including read-only symlinks in a LOCAL workspace) are
kept byte-for-byte. Only complete missing stage groups are computed and
published, one stage at a time, to avoid holding all 78 WARM matrices in RAM.
"""
from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .complete_transplant import donor_layer_positions
from .glm53_calibration import GLM53CalibrationError, GLM53StageCalibration
from .transport import save_coordinate_map


_MAP_FIELDS = (
    "residual_map", "attention_input_map", "moe_input_map",
    "compressed_kv_map", "latent_map", "dense_intermediate_map",
)


def _files(record: dict) -> set[str]:
    result = set()
    for field in _MAP_FIELDS:
        base = record.get(field)
        if base:
            result.update((base + ".json", base + ".safetensors"))
    if record.get("expert_usage"):
        result.add(record["expert_usage"])
    return result


class PartialCalibrationStageWriter:
    def __init__(self, output: str | Path, *, target_config):
        self.output = Path(output)
        self.manifest = json.loads(
            (self.output / "glm53_calibration.json").read_text(encoding="utf-8")
        )
        manifest = self.manifest
        positions = donor_layer_positions(target_config)
        stages = manifest.get("stages", [])
        if (manifest.get("schema_version") != 2 or
            manifest.get("source_layers") != 78 or
            manifest.get("calibration_mode") != "weight_only" or
            manifest.get("target_config_fingerprint") != target_config.fingerprint or
            len(stages) != 78 or len(positions) != 78):
            raise GLM53CalibrationError(
                "partial calibration manifest mismatches canonical full78 IQ target"
            )
        self.missing_stages = []
        for stage, record in enumerate(stages):
            if (record.get("stage") != stage or record.get("source_layer") != stage or
                (record.get("context_physical_layer"), record.get("moe_physical_layer"))
                != tuple(positions[stage])):
                raise GLM53CalibrationError("incompatible partial calibration stage mapping")
            expected = _files(record)
            for field in _MAP_FIELDS:
                base = record.get(field)
                if base is not None and base != {
                    "residual_map": f"stage-{stage:02d}-residual",
                    "attention_input_map": f"stage-{stage:02d}-attention-input",
                    "moe_input_map": f"stage-{stage:02d}-moe-input",
                    "compressed_kv_map": f"stage-{stage:02d}-kv",
                    "latent_map": f"stage-{stage:02d}-latent",
                    "dense_intermediate_map": f"stage-{stage:02d}-dense-intermediate",
                }[field]:
                    raise GLM53CalibrationError("incorrect source-stage calibration basename")
            actual = {name for name in expected
                      if (self.output / name).is_file() and
                      (self.output / name).stat().st_size > 0}
            if actual == expected:
                continue
            if actual:
                raise GLM53CalibrationError(
                    f"stage {stage} is partially published; refuse recomputation/overwrite"
                )
            self.missing_stages.append(stage)

    @property
    def skip_stages(self) -> frozenset[int]:
        return frozenset(range(78)).difference(self.missing_stages)

    def write_stage(self, stage: GLM53StageCalibration) -> None:
        if stage.stage not in self.missing_stages:
            raise GLM53CalibrationError(
                f"unexpected or already completed calibration stage {stage.stage}"
            )
        record = self.manifest["stages"][stage.stage]
        if (stage.source_layer != record["source_layer"] or
            stage.context_physical_layer != record["context_physical_layer"] or
            stage.moe_physical_layer != record["moe_physical_layer"]):
            raise GLM53CalibrationError("generated stage mapping diverges from saved manifest")
        # Temporary files live next to the output, avoiding cross-filesystem
        # replace. On failed writes leave old canonical files unchanged.
        with tempfile.TemporaryDirectory(
            prefix=f".warm-stage-{stage.stage:02d}-", dir=str(self.output)
        ) as tmp:
            staging = Path(tmp)
            for field in _MAP_FIELDS:
                name = record.get(field)
                if name is None:
                    continue
                value = getattr(stage, field)
                if value is None:
                    raise GLM53CalibrationError(f"missing generated WARM map {field}")
                save_coordinate_map(value, staging / name)
            usage = record.get("expert_usage")
            if usage:
                if stage.expert_usage is None:
                    raise GLM53CalibrationError("missing computed expert usage")
                np.save(staging / usage, np.asarray(stage.expert_usage, dtype=np.float64))
            expected = _files(record)
            if {p.name for p in staging.iterdir()} != expected:
                raise GLM53CalibrationError("incomplete generated stage map inventory")
            for name in sorted(expected):
                destination = self.output / name
                if destination.exists() or destination.is_symlink():
                    raise GLM53CalibrationError(
                        f"refusing overwrite of stage {stage.stage}: {name}"
                    )
                # Read-only or isolated scratch output is deliberate. The
                # upstream 866-file receipt must be built after persistence.
                with (staging / name).open("rb") as fh:
                    os.fsync(fh.fileno())
                os.replace(staging / name, destination)
        self.missing_stages.remove(stage.stage)
        print(f"GLM53_CALIBRATION_REPAIRED_SOURCE_LAYER={stage.stage}", flush=True)
        del stage
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
