"""End-to-end tiny-tensor tests for held-out functional comparison integrity."""
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
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file as save_torch

from iq_transfer.glm53_basis_functional_compare import (
    ExpertBasisFunctionalError,
    compare_basis_functions,
)
from scripts.probe_glm53_remote_ranges import REPO, REVISION


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class FunctionalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw = root / "raw"
        self.base = root / "base"
        self.candidate = root / "candidate"
        self.calibration = root / "calibration"
        for path in (self.raw, self.base, self.candidate, self.calibration):
            path.mkdir()
        self.ids = [77, 87, 246, 131]
        self.calib_hashes = {"glm53_calibration.json": "verified-sample"}
        self.fingerprint = "test-fingerprint"
        gen = np.random.default_rng(241)
        self.p0, _ = np.linalg.qr(gen.normal(size=(10, 4)))
        direction = gen.normal(size=(10, 1))
        direction -= self.p0 @ (self.p0.T @ direction)
        direction /= np.linalg.norm(direction)
        self.p1 = self.p0.copy()
        self.p1[:, 0] = (
            np.cos(0.2) * self.p0[:, 0] + np.sin(0.2) * direction[:, 0]
        )
        self.fake_config = SimpleNamespace(
            fingerprint=self.fingerprint,
            stable_moe=SimpleNamespace(expert_intermediate_size=6),
        )
        mapping = {}
        files = {}
        hashes = {}
        per_slot_entries = []
        for slot, donor_id in enumerate(self.ids):
            prefix = f"model.layers.3.mlp.experts.{donor_id}."
            tensors = {}
            for kind in ("gate", "up", "down"):
                shape = (6, 10) if kind != "down" else (10, 6)
                tensors[f"{prefix}{kind}_proj.weight"] = torch.tensor(
                    gen.normal(size=shape).astype(np.float32),
                    dtype=torch.bfloat16,
                )
            filename = f"expert-{slot}.safetensors"
            save_torch(tensors, str(self.raw / filename))
            files[filename] = digest(self.raw / filename)
            for key, value in tensors.items():
                mapping[key] = filename
                hashes[key] = tensor_digest(value)
            physical = 5
            root_slot = self.base / f"slot-{slot:03d}"
            root_slot.mkdir()
            donor = {
                k: tensors[f"{prefix}{k}_proj.weight"].float()
                for k in ("gate", "up", "down")
            }
            proj = {
                "gate": (donor["gate"] @ torch.tensor(self.p0, dtype=torch.float32))
                .to(torch.bfloat16),
                "up": (donor["up"] @ torch.tensor(self.p0, dtype=torch.float32))
                .to(torch.bfloat16),
                "down": (
                    torch.tensor(self.p0, dtype=torch.float32).T @ donor["down"]
                ).to(torch.bfloat16),
            }
            target = {
                f"layers.{physical}.moe.routed_experts.{slot}.{kind}_proj.weight": value
                for kind, value in proj.items()
            }
            shard = "projected.safetensors"
            save_torch(target, str(root_slot / shard))
            report = {
                "stage": 1,
                "donor_expert_id": donor_id,
                "recipient_slot": slot,
                "recipient_moe_physical_layer": physical,
                "not_an_iq_checkpoint": True,
                "calibration_sha256": self.calib_hashes,
                "shard": shard,
                "shard_sha256": digest(root_slot / shard),
                "output_tensor_sha256": {
                    key: tensor_digest(value) for key, value in target.items()
                },
                "projection": {
                    kind: {"subspace_retained_energy": 0.4}
                    for kind in ("gate", "up", "down")
                },
            }
            (root_slot / "projection_manifest.json").write_text(json.dumps(report))
            per_slot_entries.append({
                "slot": slot, "donor_expert_id": donor_id,
                "relative_path": f"slot-{slot:03d}",
            })
        manifest = {
            "source_repo": REPO,
            "source_revision": REVISION,
            "not_an_iq_checkpoint": True,
            "target_experts": 64,
            "target_config_fingerprint": self.fingerprint,
            "layers": [{
                "stage": 1, "source_layer": 3,
                "exported_donor_ids": self.ids,
            }],
            "weight_map": mapping,
            "file_sha256": files,
            "tensor_sha256": hashes,
        }
        raw_json = self.raw / "raw_expert_manifest.json"
        raw_json.write_text(json.dumps(manifest))
        raw_hash = digest(raw_json)
        for slot in range(4):
            target_file = self.base / f"slot-{slot:03d}" / "projection_manifest.json"
            report = json.loads(target_file.read_text())
            report["raw_bundle_manifest_sha256"] = raw_hash
            target_file.write_text(json.dumps(report))
            per_slot_entries[slot]["projection_manifest_sha256"] = digest(target_file)
        stage_meta = {
            "schema_version": 1,
            "stage": 1,
            "not_an_iq_checkpoint": True,
            "raw_bundle_manifest_sha256": raw_hash,
            "calibration_file_sha256": self.calib_hashes,
            "entries": per_slot_entries,
        }
        stage_path = self.base / "stage_projection_manifest.json"
        stage_path.write_text(json.dumps(stage_meta))
        matrix_path = self.candidate / "candidate-stage-latent.safetensors"
        save_np({"matrix": self.p1}, str(matrix_path))
        probe = {
            "schema_version": 1,
            "artifact_type": "glm53_expert_informed_basis_energy_probe",
            "not_an_iq_checkpoint": True,
            "not_canonical_calibration": True,
            "weight_energy_only": True,
            "stage": 1,
            "source_layer": 3,
            "source_expert_ids": self.ids,
            "train_slots": [0, 1, 2],
            "validation_slot": 3,
            "calibration_sha256": self.calib_hashes,
            "raw_manifest_sha256": raw_hash,
            "baseline_stage_manifest_sha256": digest(stage_path),
            "basis_dimensions": [10, 4],
            "modified_basis_columns": [0],
            "candidate_matrix_sha256": digest(matrix_path),
        }
        (self.candidate / "basis_probe_report.json").write_text(json.dumps(probe))

    def _run(self, **kwargs):
        with (
            patch(
                "iq_transfer.glm53_basis_functional_compare.canonical_glm53_config",
                return_value=self.fake_config,
            ),
            patch(
                "iq_transfer.glm53_basis_functional_compare._calibration_map",
                return_value=(self.p0, 5, self.calib_hashes),
            ),
        ):
            return compare_basis_functions(
                raw_dir=self.raw,
                baseline_dir=self.base,
                candidate_dir=self.candidate,
                calibration_dir=self.calibration,
                samples=4, **kwargs,
            )

    def test_synthetic_candidate_vs_baseline_common_output(self):
        report = self._run()
        self.assertEqual(report["input_origin"], "synthetic_not_real_activations")
        self.assertEqual(report["validation_slot"], 3)
        self.assertEqual(len(report["experts"]), 4)
        self.assertEqual(report["experts"][3]["role"], "validation")
        self.assertEqual(
            set(report["result"]),
            {
                "isotropic_full_donor_width",
                "original_warm_subspace",
                "candidate_warm_subspace",
            },
        )
        for expert in report["experts"]:
            for measurements in expert["scenarios"].values():
                self.assertTrue(np.isfinite(
                    measurements["relative_rmse_improvement"]
                ))
                self.assertTrue(np.isfinite(
                    measurements["original_warm"]["relative_rmse"]
                ))

    def test_explicit_real_capture_is_not_generated(self):
        data = np.zeros((3, 10), dtype=np.float32)
        data[:, :2] = 1
        file = Path(self.temp.name) / "user-provided.npy"
        np.save(file, data)
        report = self._run(captured_activations=file)
        self.assertEqual(report["input_origin"], "user_supplied_capture_unverified_origin")
        self.assertEqual(report["samples_per_scenario"], 3)
        self.assertEqual(
            list(report["result"]), ["captured_glm_pre_expert_hidden_states"]
        )
        self.assertEqual(report["capture_sha256"], digest(file))

    def test_corrupt_candidate_rejected_before_comparison(self):
        meta = self.candidate / "candidate-stage-latent.safetensors"
        with meta.open("ab") as file:
            file.write(b"unexpected")
        with self.assertRaisesRegex(ExpertBasisFunctionalError,
                                    "candidate basis SHA256 mismatch"):
            self._run()


if __name__ == "__main__":
    unittest.main()
