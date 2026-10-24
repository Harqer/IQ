"""CPU tests for single-stage raw export selection and map-once batch projection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from iq_transfer.complete_transplant import canonical_glm53_config
from iq_transfer.glm53_raw_expert_export import (
    RawExpertExportError, export_raw_selected_experts, expert_keys,
)
from iq_transfer.glm53_single_expert_projection import GLM53SingleExpertProjectionError
from iq_transfer.glm53_stage_projection import project_raw_stage
from scripts.probe_glm53_remote_ranges import REPO, REVISION


class Source:
    def __init__(self):
        self.values = {}
        prefix = "model.layers.3.mlp"
        self.values[f"{prefix}.gate.weight"] = torch.zeros((256, 4), dtype=torch.bfloat16)
        self.values[f"{prefix}.gate.e_score_correction_bias"] = torch.zeros(
            256, dtype=torch.float32
        )
        for name in (f"{prefix}.shared_experts",
                     f"{prefix}.experts.77", f"{prefix}.experts.4"):
            shapes = ((3, 4), (3, 4), (4, 3))
            for key, shape in zip(expert_keys(name), shapes, strict=True):
                self.values[key] = torch.ones(shape, dtype=torch.bfloat16)

    def shape(self, key):
        return tuple(self.values[key].shape)

    def get(self, key):
        return self.values[key]


class StageBatchTests(unittest.TestCase):
    def setUp(self):
        self.config = canonical_glm53_config(routed_experts=64)

    def test_stage_selector_exports_subset_and_marks_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            result = export_raw_selected_experts(
                source=Source(),
                plan={
                    "target_experts": 64,
                    "target_config_fingerprint": self.config.fingerprint,
                },
                tasks=[(1, 3, (77, 4))],
                root=root, hidden=4, intermediate=3,
                index_sha256="official-index",
                selected_stage=1, max_experts=1,
            )
            self.assertEqual(result["status"], "partial_smoke_only")
            self.assertEqual(result["layers"][0]["exported_donor_ids"], [77])
            self.assertEqual(result["tensor_count"], 8)
            self.assertFalse((root / "INCOMPLETE").exists())

    def test_selector_rejects_incompatible_max_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RawExpertExportError, "cannot combine"):
                export_raw_selected_experts(
                    source=Source(), plan={}, tasks=[(1, 3, (77,))],
                    root=Path(tmp) / "bad", hidden=4, intermediate=3,
                    index_sha256="official-index",
                    max_stages=1, selected_stage=1,
                )
            self.assertFalse((Path(tmp) / "bad").exists())

    def _write_bundle(self, root: Path, count: int = 2):
        root.mkdir()
        manifest = {
            "source_repo": REPO,
            "source_revision": REVISION,
            "not_an_iq_checkpoint": True,
            "target_experts": 64,
            "target_config_fingerprint": self.config.fingerprint,
            "layers": [
                {"stage": 1, "source_layer": 3,
                 "exported_donor_ids": [77, 4][:count]}
            ],
        }
        (root / "raw_expert_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_batch_loads_one_map_and_keeps_expert_id_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            raw = home / "raw"
            self._write_bundle(raw)
            calibration = home / "cal"
            calibration.mkdir()
            output = home / "projected"
            cached = (np.eye(4, 2), 5, {"glm53_calibration.json": "hash"})

            def fake_project(**kwargs):
                slot = kwargs["slot"]
                self.assertIs(kwargs["_verified_calibration"], cached)
                dest = kwargs["output_dir"]
                dest.mkdir()
                (dest / "projection_manifest.json").write_text(
                    json.dumps({"slot": slot}), encoding="utf-8"
                )
                return {
                    "donor_expert_id": [77, 4][slot],
                    "projection": {
                        kind: {"subspace_retained_energy": 0.5}
                        for kind in ("gate", "up", "down")
                    },
                }

            with (
                patch("iq_transfer.glm53_stage_projection._calibration_map",
                      return_value=cached) as calibration_reader,
                patch("iq_transfer.glm53_stage_projection.project_one_raw_expert",
                      side_effect=fake_project) as projector,
            ):
                result = project_raw_stage(
                    raw_dir=raw,
                    calibration_dir=calibration,
                    output_dir=output, experts=2,
                )
            self.assertEqual(calibration_reader.call_count, 1)
            self.assertEqual(projector.call_count, 2)
            self.assertEqual([x["donor_expert_id"] for x in result["entries"]],
                             [77, 4])
            self.assertEqual(result["status"], "partial_stage_experiment")
            self.assertFalse((output / "INCOMPLETE").exists())
            self.assertTrue((output / "stage_projection_manifest.json").exists())

    def test_batch_rejects_more_than_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            raw = home / "raw"
            self._write_bundle(raw)
            with self.assertRaisesRegex(GLM53SingleExpertProjectionError,
                                        "only contains 2"):
                project_raw_stage(
                    raw_dir=raw, calibration_dir=home / "cal",
                    output_dir=home / "out", experts=3,
                )
            self.assertFalse((home / "out").exists())


if __name__ == "__main__":
    unittest.main()
