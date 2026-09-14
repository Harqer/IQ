from __future__ import annotations

import unittest
import numpy as np

from iq_transfer import (
    CoordinateMap,
    DonorError,
    MappingTensorSource,
    MeasurementPlan,
    Phi4Inspector,
    ScaleGate,
    TransferMetrics,
    extract_shadow,
    fit_ridge_coordinate_map,
    match_layers_monotonic,
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

    def test_phi4_rejects_wrong_checkpoint_shape(self):
        source = self.phi_source()
        broken = dict(source._tensors)
        broken["model.layers.0.self_attn.qkv_proj.weight"] = np.zeros((15, 8))
        inspector = Phi4Inspector.from_config_mapping(self.phi_config())
        with self.assertRaises(DonorError):
            inspector.operators(MappingTensorSource(broken))

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

    def test_ridge_map_recovers_paired_coordinates(self):
        rng = np.random.default_rng(4)
        xs = rng.normal(size=(80, 6))
        p = rng.normal(size=(6, 4))
        xt = xs @ p
        learned = fit_ridge_coordinate_map(xs, xt, ridge=1e-6)
        self.assertTrue(np.allclose(xs @ learned.matrix, xt, atol=1e-4))

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

    def test_scale_gate(self):
        gate = ScaleGate(min_retention=0.8, max_compute_ratio=0.5)
        good = TransferMetrics(100.0, 85.0, 70.0, 40.0, 100.0)
        bad = TransferMetrics(100.0, 75.0, 70.0, 40.0, 100.0)
        self.assertTrue(gate.passes(good))
        self.assertFalse(gate.passes(bad))


if __name__ == "__main__":
    unittest.main()
