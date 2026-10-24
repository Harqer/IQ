"""Isolated CPU tests for one raw GLM expert -> IQ latent projection."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from iq_transfer.complete_transplant import (
    canonical_glm53_config,
    donor_layer_positions,
)
from iq_transfer.glm53_single_expert_projection import (
    GLM53SingleExpertProjectionError,
    _calibration_map,
    _project_down,
    _project_gate_or_up,
    project_one_raw_expert,
)
from iq_transfer.transport import CoordinateMap, save_coordinate_map
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _raw_bundle(root: Path) -> dict:
    config = canonical_glm53_config(routed_experts=64)
    root.mkdir()
    stage, layer, donor = 1, 3, 77
    prefix = f"model.layers.{layer}.mlp.experts.{donor}"
    rng = np.random.default_rng(67)
    tensors = {}
    for kind in ("gate", "up", "down"):
        shape = (2048, 4) if kind != "down" else (4, 2048)
        values = rng.normal(size=shape).astype(np.float32)
        tensors[f"{prefix}.{kind}_proj.weight"] = torch.tensor(
            values, dtype=torch.bfloat16
        )
    filename = "stage-01-slot-000-donor-077.safetensors"
    save_file(tensors, str(root / filename))
    manifest = {
        "schema_version": 1,
        "not_an_iq_checkpoint": True,
        "source_repo": REPO,
        "source_revision": REVISION,
        "status": "partial_smoke_only",
        "target_experts": 64,
        "target_config_fingerprint": config.fingerprint,
        "layers": [
            {"stage": stage, "source_layer": layer,
             "exported_donor_ids": [donor]}
        ],
        "weight_map": {k: filename for k in tensors},
        "file_sha256": {filename: _sha(root / filename)},
        "tensor_sha256": {k: _tensor_sha(t) for k, t in tensors.items()},
    }
    (root / "raw_expert_manifest.json").write_text(json.dumps(manifest))
    return manifest


class SingleExpertProjectionTests(unittest.TestCase):
    def test_linear_maps_and_orthogonal_energy(self):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(7, 3)))
        gate = torch.tensor(rng.normal(size=(5, 7)), dtype=torch.bfloat16)
        down = torch.tensor(rng.normal(size=(7, 5)), dtype=torch.bfloat16)
        g, gm = _project_gate_or_up(gate, q, block_rows=2)
        d, dm = _project_down(down, q, block_rows=3)
        expect_gate = gate.float().numpy().astype(np.float64) @ q
        expect_down = q.T @ down.float().numpy().astype(np.float64)
        self.assertTrue(torch.allclose(
            g.float(), torch.from_numpy(expect_gate).to(torch.bfloat16).float()
        ))
        self.assertTrue(torch.allclose(
            d.float(), torch.from_numpy(expect_down).to(torch.bfloat16).float()
        ))
        for measurements in (gm, dm):
            self.assertGreaterEqual(measurements["subspace_retained_energy"], 0)
            self.assertLessEqual(measurements["subspace_retained_energy"], 1)

    def test_project_real_bundle_format_without_touching_raw_data(self):
        rng = np.random.default_rng(6)
        p, _ = np.linalg.qr(rng.normal(size=(4, 2)))
        physical = donor_layer_positions(
            canonical_glm53_config(routed_experts=64)
        )[1][1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            original = _raw_bundle(raw)
            output = root / "projection"
            with patch(
                "iq_transfer.glm53_single_expert_projection._calibration_map",
                return_value=(p, physical, {"stage-01-latent.safetensors": "sha"}),
            ):
                report = project_one_raw_expert(
                    raw_dir=raw, calibration_dir=root / "cal",
                    output_dir=output, stage=1, slot=0, block_rows=8,
                )
            self.assertTrue(report["not_an_iq_checkpoint"])
            self.assertEqual(report["donor_expert_id"], 77)
            self.assertEqual(report["recipient_moe_physical_layer"], physical)
            self.assertEqual(report["target_latent_size"], 2)
            self.assertFalse((output / "INCOMPLETE").exists())
            self.assertFalse((output / "model.safetensors.index.json").exists())
            self.assertEqual(original, json.loads(
                (raw / "raw_expert_manifest.json").read_text()
            ))
            self.assertEqual(_sha(output / report["shard"]), report["shard_sha256"])
            weights = load_file(str(output / report["shard"]))
            self.assertEqual(len(weights), 3)
            for kind in ("gate", "up", "down"):
                key = f"layers.{physical}.moe.routed_experts.0.{kind}_proj.weight"
                shape = (2048, 2) if kind != "down" else (2, 2048)
                self.assertEqual(weights[key].shape, shape)
                self.assertEqual(weights[key].dtype, torch.bfloat16)
                self.assertEqual(_tensor_sha(weights[key]),
                                 report["output_tensor_sha256"][key])
                self.assertTrue(0 <= report["projection"][kind][
                    "subspace_retained_energy"
                ] <= 1)

    def test_donor_hash_change_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            _raw_bundle(raw)
            filename = next(raw.glob("*.safetensors"))
            with filename.open("ab") as handle:
                handle.write(b"changed")
            p = np.eye(4, 2)
            physical = donor_layer_positions(
                canonical_glm53_config(routed_experts=64)
            )[1][1]
            with patch(
                "iq_transfer.glm53_single_expert_projection._calibration_map",
                return_value=(p, physical, {}),
            ):
                with self.assertRaisesRegex(
                    GLM53SingleExpertProjectionError, "SHA mismatch"
                ):
                    project_one_raw_expert(
                        raw_dir=raw, calibration_dir=root / "cal",
                        output_dir=root / "out",
                    )
            self.assertFalse((root / "out").exists())

    def test_calibration_map_checks_hash_and_orthogonality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            configs = {
                32: SimpleNamespace(fingerprint="base"),
                64: SimpleNamespace(
                    fingerprint="target",
                    model=SimpleNamespace(hidden_size=4),
                    stable_moe=SimpleNamespace(latent_size=2),
                ),
            }
            p = np.eye(6, 2)
            save_coordinate_map(
                CoordinateMap(p, 1e-6, "donor", "iq"),
                root / "stage-01-latent",
            )
            entries = [
                {"stage": i, "source_layer": 3 if i == 1 else i,
                 "moe_physical_layer": i,
                 "latent_map": f"stage-{i:02d}-latent"}
                for i in range(24)
            ]
            (root / "glm53_calibration.json").write_text(json.dumps({
                "schema_version": 2, "source_layers": 78,
                "stages": entries, "calibration_mode": "weight_only",
                "target_config_fingerprint": "base",
            }))
            files = {
                name: _sha(root / name)
                for name in (
                    "glm53_calibration.json", "stage-01-latent.json",
                    "stage-01-latent.safetensors",
                )
            }
            (root / ".calibration_integrity.json").write_text(
                json.dumps({"files": files})
            )
            with (
                patch("iq_transfer.glm53_single_expert_projection.canonical_glm53_config",
                      side_effect=lambda routed_experts=32: configs[routed_experts]),
                patch("iq_transfer.glm53_single_expert_projection.donor_layer_positions",
                      return_value=tuple((i, i) for i in range(24))),
            ):
                loaded, physical, hashes = _calibration_map(root, 1, 3, 64)
                self.assertTrue(np.array_equal(loaded, p))
                self.assertEqual(physical, 1)
                self.assertEqual(hashes, files)
                corrupt = root / "stage-01-latent.json"
                corrupt.write_text(corrupt.read_text() + " ", encoding="utf-8")
                with self.assertRaisesRegex(
                    GLM53SingleExpertProjectionError, "integrity mismatch"
                ):
                    _calibration_map(root, 1, 3, 64)


if __name__ == "__main__":
    unittest.main()
