from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from iq_transfer import (
    ActivationTap,
    CaptureError,
    CoordinateMap,
    DonorError,
    ManifestError,
    Mamba3BootstrapWeights,
    Mamba3InitError,
    Mamba3Layout,
    ParameterProvenance,
    PlanError,
    ProvenanceError,
    ProvenanceLedger,
    MappingTensorSource,
    MeasurementPlan,
    Phi4Inspector,
    ScaleGate,
    SlotError,
    TargetAssignment,
    TargetRegistry,
    TargetSlot,
    TorchActivationCapture,
    TransportPlan,
    TransferMethod,
    TransferMetrics,
    apply_mamba3_bootstrap,
    build_donor_manifest,
    extract_shadow,
    fit_ridge_coordinate_map,
    load_coordinate_map,
    match_layers_monotonic,
    save_coordinate_map,
    shadow_distance,
    transport_linear,
)


class TransferTests(unittest.TestCase):
    def phi_config(self):
        return {
            "_name_or_path": "test-phi4",
            "model_type": "phi3",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "torch_dtype": "float32",
        }

    def phi_source(self):
        tensors = {}
        for layer in range(2):
            p = f"model.layers.{layer}"
            tensors[f"{p}.self_attn.qkv_proj.weight"] = np.zeros((16, 8))
            tensors[f"{p}.self_attn.o_proj.weight"] = np.zeros((8, 8))
            tensors[f"{p}.mlp.gate_up_proj.weight"] = np.zeros((24, 8))
            tensors[f"{p}.mlp.down_proj.weight"] = np.zeros((8, 12))
        return MappingTensorSource(tensors)

    def test_phi4_splits_fused_operators_without_copying_checkpoint(self):
        inspector = Phi4Inspector.from_config_mapping(self.phi_config())
        refs = inspector.operators(self.phi_source())
        self.assertEqual(len(refs), 14)
        roles = [r.role for r in refs[:7]]
        self.assertEqual(roles, ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"])
        self.assertEqual(refs[0].shape, (8, 8))
        self.assertEqual(refs[1].shape, (4, 8))
        self.assertEqual(refs[4].shape, (12, 8))
        self.assertEqual([layer.index for layer in inspector.layers()], [0, 1])
        self.assertTrue(inspector.validate_checkpoint(self.phi_source()).ok)

    def test_phi4_rejects_wrong_checkpoint_shape(self):
        source = self.phi_source()
        broken = dict(source._tensors)
        broken["model.layers.0.self_attn.qkv_proj.weight"] = np.zeros((15, 8))
        inspector = Phi4Inspector.from_config_mapping(self.phi_config())
        with self.assertRaises(DonorError):
            inspector.operators(MappingTensorSource(broken))
        self.assertFalse(inspector.validate_checkpoint(MappingTensorSource(broken)).ok)

    def test_manifest_is_deterministic_and_fails_closed_without_checkpoint_files(self):
        inspector = Phi4Inspector.from_config_mapping(self.phi_config())
        source = self.phi_source()
        kwargs = dict(
            donor_id="phi-test",
            checkpoint_revision="deadbeef",
            tokenizer_hash="tok123",
            license="MIT-test-only",
            operator_layout_version="phi3-fused-v1",
            source_uri="memory://phi-test",
        )
        with self.assertRaises(ManifestError):
            build_donor_manifest(inspector.config, source, **kwargs)
        a = build_donor_manifest(inspector.config, source, allow_metadata_only=True, **kwargs)
        b = build_donor_manifest(inspector.config, source, allow_metadata_only=True, **kwargs)
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.vocab_size, 32)
        self.assertEqual(len(a.tensors), 8)

    def test_target_registry_prevents_multiple_owners(self):
        assignment = TargetAssignment(
            target_module_path="blocks.0.attn.q_proj",
            target_slot=TargetSlot.ATTN_Q,
            target_shape=(8, 8),
            transfer_method=TransferMethod.OPERATOR_TRANSPORT,
            source_donor_id="phi",
            source_layer=0,
            source_operator="attn.q",
            input_map_id="resid-0",
            output_map_id="q-0",
        )
        registry = TargetRegistry([assignment])
        self.assertIs(registry.get("blocks.0.attn.q_proj", TargetSlot.ATTN_Q), assignment)
        with self.assertRaises(SlotError):
            registry.add(assignment)
        with self.assertRaises(SlotError):
            TargetAssignment(
                target_module_path="exec.energy",
                target_slot=TargetSlot.RESIDUAL,
                target_shape=(8, 8),
                transfer_method=TransferMethod.RECIPIENT_NATIVE,
                source_donor_id="phi",
            )

    def test_shadow_is_architecture_width_independent(self):
        rng = np.random.default_rng(2)
        latent = rng.normal(size=(64, 4))
        xs = latent @ rng.normal(size=(4, 12))
        xt = latent @ rng.normal(size=(4, 7))
        noise = rng.normal(size=(64, 7))
        plan = MeasurementPlan(64, measurements=128, seed=42)
        s = extract_shadow(xs, plan, layer=0)
        t = extract_shadow(xt, plan, layer=0)
        bad = extract_shadow(noise, plan, layer=0)
        self.assertLess(shadow_distance(s, t), shadow_distance(s, bad))

    def test_monotonic_layer_match(self):
        rng = np.random.default_rng(3)
        plan = MeasurementPlan(40, measurements=64, seed=7)
        bases = [rng.normal(size=(40, 5)) for _ in range(4)]
        source = {i: extract_shadow(bases[i], plan, layer=i) for i in range(4)}
        target = {
            0: extract_shadow(bases[0] @ rng.normal(size=(5, 3)), plan, layer=0),
            1: extract_shadow(bases[3] @ rng.normal(size=(5, 3)), plan, layer=1),
        }
        mapping = match_layers_monotonic(source, target, depth_prior=0.01)
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[1], 3)

    def test_ridge_map_recovers_paired_coordinates_and_reports_diagnostics(self):
        rng = np.random.default_rng(4)
        xs = rng.normal(size=(80, 6))
        p = rng.normal(size=(6, 4))
        xt = xs @ p
        learned = fit_ridge_coordinate_map(
            xs[:60],
            xt[:60],
            ridge=1e-6,
            source_space="phi.residual.0",
            target_space="iq.residual.0",
            validation_source=xs[60:],
            validation_target=xt[60:],
        )
        self.assertTrue(np.allclose(xs @ learned.matrix, xt, atol=1e-4))
        self.assertEqual(learned.source_space, "phi.residual.0")
        self.assertIsNotNone(learned.diagnostics)
        self.assertLess(learned.diagnostics.validation_rmse, 1e-4)

    def test_coordinate_map_artifact_round_trip(self):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("safetensors not installed")
        coordinate_map = CoordinateMap(np.eye(3), 1e-3, "source", "target")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "map"
            save_coordinate_map(coordinate_map, base)
            loaded = load_coordinate_map(base)
            self.assertTrue(np.array_equal(loaded.matrix, coordinate_map.matrix))
            self.assertEqual(loaded.source_space, "source")
            self.assertEqual(loaded.target_space, "target")

    def test_transport_linear_preserves_function(self):
        rng = np.random.default_rng(5)
        pin = rng.normal(size=(4, 4))
        while abs(np.linalg.det(pin)) < 0.1:
            pin = rng.normal(size=(4, 4))
        pout = rng.normal(size=(3, 3))
        while abs(np.linalg.det(pout)) < 0.1:
            pout = rng.normal(size=(3, 3))
        ws = rng.normal(size=(3, 4))
        wt = transport_linear(ws, CoordinateMap(pin, 1e-3), CoordinateMap(pout, 1e-3))
        x_s = rng.normal(size=(20, 4))
        x_t = x_s @ pin
        y_expected = (x_s @ ws.T) @ pout
        y_actual = x_t @ wt.T
        self.assertTrue(np.allclose(y_actual, y_expected, atol=1e-9))

    def test_torch_activation_capture_is_real_and_removes_hooks(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("PyTorch not installed")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(3, 4), nn.Linear(4, 2)])

            def forward(self, x):
                return self.layers[1](torch.relu(self.layers[0](x)))

        model = Model()
        tap = ActivationTap("hidden", "layers.0", "output")
        capture = TorchActivationCapture(model, [tap])
        with capture:
            _ = model(torch.ones(2, 3))
        records = capture.records()
        self.assertEqual(records["hidden"][0].shape, (2, 4))
        before = len(records["hidden"])
        _ = model(torch.ones(2, 3))
        self.assertEqual(len(capture.records()["hidden"]), before)
        with self.assertRaises(CaptureError):
            TorchActivationCapture(model, [ActivationTap("x", "missing")]).__enter__()

    def test_transport_plan_is_deterministic_and_rejects_unknown_donors(self):
        assignment = TargetAssignment(
            target_module_path="blocks.0.attn.q_proj",
            target_slot=TargetSlot.ATTN_Q,
            target_shape=(8, 8),
            transfer_method=TransferMethod.OPERATOR_TRANSPORT,
            source_donor_id="phi",
            source_layer=0,
            source_operator="attn.q",
            input_map_id="resid-0",
            output_map_id="q-0",
        )
        plan = TransportPlan(
            plan_id="phi-iq-v1",
            donor_fingerprints=(("phi", "abc123"),),
            assignments=(assignment,),
            calibration_manifest_hash="cal123",
            iq_config_hash="cfg123",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan.write_json(path)
            loaded = TransportPlan.from_json(path)
        self.assertEqual(plan.fingerprint, loaded.fingerprint)
        with self.assertRaises(PlanError):
            TransportPlan(
                plan_id="bad",
                donor_fingerprints=(("other", "x"),),
                assignments=(assignment,),
                calibration_manifest_hash="cal",
                iq_config_hash="cfg",
            )

    def test_provenance_ledger_round_trip_and_duplicate_rejection(self):
        record = ParameterProvenance(
            target_parameter="blocks.0.attn.q_proj.weight",
            target_slot=TargetSlot.ATTN_Q,
            transfer_method=TransferMethod.OPERATOR_TRANSPORT,
            training_phase_introduced="T1",
            source_donor_id="phi",
            source_tensor="model.layers.0.self_attn.qkv_proj.weight",
            source_slice="q",
            map_ids=("resid-0", "q-0"),
        )
        ledger = ProvenanceLedger([record])
        with self.assertRaises(ProvenanceError):
            ledger.add(record)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            ledger.write_json(path)
            loaded = ProvenanceLedger.from_json(path)
        self.assertEqual(loaded.get(record.target_parameter), record)

    def test_mamba3_layout_matches_upstream_packing_and_preserves_native_slices(self):
        layout = Mamba3Layout(d_model=8, d_state=4, expand=2, headdim=4, ngroups=1, rope_fraction=0.5)
        self.assertEqual(layout.split_sizes, (16, 16, 4, 4, 4, 4, 4, 1))
        native_in = np.full(layout.in_proj_shape, 7.0)
        native_out = np.full(layout.out_proj_shape, 8.0)
        bootstrap = Mamba3BootstrapWeights(
            x=np.full((16, 8), 1.0),
            B=np.full((4, 8), 2.0),
            C=np.full((4, 8), 3.0),
            out_proj=np.full(layout.out_proj_shape, 4.0),
        )
        new_in, new_out, report = apply_mamba3_bootstrap(native_in, native_out, layout, bootstrap)
        slices = layout.slices()
        self.assertTrue(np.all(new_in[slices["z"], :] == 7.0))
        self.assertTrue(np.all(new_in[slices["x"], :] == 1.0))
        self.assertTrue(np.all(new_in[slices["B"], :] == 2.0))
        self.assertTrue(np.all(new_in[slices["C"], :] == 3.0))
        self.assertTrue(np.all(new_in[slices["dd_dt"], :] == 7.0))
        self.assertTrue(np.all(new_out == 4.0))
        self.assertEqual(report.written_slices, ("x", "B", "C", "out_proj"))

    def test_mamba3_bootstrap_rejects_wrong_shapes(self):
        layout = Mamba3Layout(d_model=8, d_state=4, expand=2, headdim=4)
        native_in = np.zeros(layout.in_proj_shape)
        native_out = np.zeros(layout.out_proj_shape)
        bad = Mamba3BootstrapWeights(
            x=np.zeros((15, 8)),
            B=np.zeros((4, 8)),
            C=np.zeros((4, 8)),
            out_proj=np.zeros(layout.out_proj_shape),
        )
        with self.assertRaises(Mamba3InitError):
            apply_mamba3_bootstrap(native_in, native_out, layout, bad)

    def test_scale_gate(self):
        gate = ScaleGate(min_retention=0.8, max_compute_ratio=0.5)
        good = TransferMetrics(100.0, 85.0, 70.0, 40.0, 100.0)
        bad = TransferMetrics(100.0, 75.0, 70.0, 40.0, 100.0)
        self.assertTrue(gate.passes(good))
        self.assertFalse(gate.passes(bad))


if __name__ == "__main__":
    unittest.main()
