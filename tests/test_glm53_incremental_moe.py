"""Incremental MoE shard output must match the existing single-stage compiler."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

from iq_transfer.glm53_compile import (
    GLM53CompileError,
    _moe_state,
    _validate_expanded_calibration,
)
from iq_transfer.complete_transplant import canonical_glm53_config
from iq_transfer.glm53_incremental_moe import (
    _inverse,
    save_incremental_moe_stage,
)
from iq_transfer.transport import CoordinateMap


def _maps(hidden: int, middle: int):
    h = CoordinateMap(np.eye(hidden, dtype=np.float64), 1e-6)
    m = CoordinateMap(np.eye(middle, dtype=np.float64), 1e-6)
    return h, m


class _Tensors:
    def __init__(self, tensors):
        self.tensors = tensors
        self.read_keys = []

    def get(self, key: str):
        self.read_keys.append(key)
        return self.tensors[key]

    def shape(self, key: str):
        return tuple(self.tensors[key].shape)

    def get_rows(self, key: str, row_indices):
        self.read_keys.append(key + ":rows")
        return self.tensors[key][list(row_indices)]


def _donor(*, sparse: bool, bf16: bool = False):
    rng = np.random.default_rng(19)
    hidden, middle = 4, 3
    source_layer = 3 if sparse else 0
    prefix = f"model.layers.{source_layer}.mlp"
    dtype = torch.bfloat16 if bf16 else torch.float32
    tensors = {}

    def make_expert(root: str):
        for name, shape in (
            ("gate", (middle, hidden)),
            ("up", (middle, hidden)),
            ("down", (hidden, middle)),
        ):
            tensors[f"{root}.{name}_proj.weight"] = torch.tensor(
                rng.normal(size=shape), dtype=dtype
            )

    if sparse:
        # Only two routed donor experts are populated: unselected weights
        # must never be requested from the remote source.
        make_expert(f"{prefix}.shared_experts")
        make_expert(f"{prefix}.experts.7")
        make_expert(f"{prefix}.experts.25")
        tensors[f"{prefix}.gate.weight"] = torch.tensor(
            rng.normal(size=(256, hidden)), dtype=dtype
        )
        tensors[f"{prefix}.gate.e_score_correction_bias"] = torch.tensor(
            rng.normal(size=(256,)), dtype=dtype
        )
        usage = np.full(256, 1e-6)
        usage[7] = 5.0
        usage[25] = 4.0
    else:
        make_expert(prefix)
        usage = None

    hidden_map, middle_map = _maps(hidden, middle)
    stage = SimpleNamespace(
        stage=0,
        source_layer=source_layer,
        moe_physical_layer=2,
        residual_map=hidden_map,
        latent_map=hidden_map,
        moe_input_map=hidden_map,
        dense_intermediate_map=middle_map if not sparse else None,
        expert_usage=usage,
    )
    stable = SimpleNamespace(
        hidden_size=hidden,
        latent_size=hidden,
        expert_intermediate_size=middle,
        num_experts=2,
    )
    config = SimpleNamespace(
        stable_moe=stable, model=SimpleNamespace(hidden_size=hidden)
    )
    inspector = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden),
        layout=SimpleNamespace(
            first_k_dense_replace=3, moe_intermediate_size=middle,
        ),
    )
    return _Tensors(tensors), stage, config, inspector


class TestIncrementalMoE(unittest.TestCase):
    def _compare_with_old(self, *, sparse: bool):
        source, stage, config, inspector = _donor(sparse=sparse)
        expected = _moe_state(
            source=source, stage=stage, config=config, inspector=inspector
        )
        source.read_keys.clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {}
            save_incremental_moe_stage(
                source=source,
                inspector=inspector,
                stage=stage,
                config=config,
                output_dir=root,
                weight_map=index,
            )
            actual = {}
            for filename in set(index.values()):
                actual.update(load_file(str(root / filename)))
            self.assertEqual(set(actual), set(expected))
            self.assertEqual(set(actual), set(index))
            for key, want in expected.items():
                self.assertTrue(
                    torch.allclose(
                        actual[key].float(), want.float(), rtol=0.01, atol=0.01
                    ), msg=f"wrong tensor: {key}"
                )
            # Common + shared + one file per routed expert.
            self.assertEqual(len(set(index.values())), 2 + config.stable_moe.num_experts)
            self.assertFalse((root / "model-moe-000.safetensors").exists())
        if sparse:
            self.assertFalse(any(".experts.6." in key for key in source.read_keys))
            self.assertEqual(
                {k.split(".experts.")[1].split(".")[0] for k in source.read_keys
                 if ".experts." in k and ".shared_experts." not in k},
                {"7", "25"},
            )

    def test_sparse_matches_existing_selected_moe_outputs(self):
        self._compare_with_old(sparse=True)

    def test_dense_matches_existing_neutral_routed_outputs(self):
        self._compare_with_old(sparse=False)

    def test_dense_donor_never_materializes_full_matrices(self):
        source, stage, config, inspector = _donor(sparse=False)

        def forbidden_get(key):
            raise AssertionError(f"dense full-tensor read forbidden: {key}")

        source.get = forbidden_get
        with tempfile.TemporaryDirectory() as tmp:
            index = {}
            save_incremental_moe_stage(
                source=source, inspector=inspector,
                stage=stage, config=config,
                output_dir=Path(tmp), weight_map=index,
            )
            self.assertEqual(len(index), 6 + 3 + 2 * 3)
        self.assertEqual(
            sorted(set(source.read_keys)),
            [
                "model.layers.0.mlp.down_proj.weight:rows",
                "model.layers.0.mlp.gate_proj.weight:rows",
                "model.layers.0.mlp.up_proj.weight:rows",
            ],
        )

    def test_orthogonal_subcloning_inverse_skips_svd(self):
        from unittest.mock import patch

        source_map = CoordinateMap(
            np.array([
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ]),
            1e-6,
        )
        with patch("iq_transfer.glm53_incremental_moe.np.linalg.pinv",
                   side_effect=AssertionError("unnecessary SVD")):
            self.assertTrue(np.array_equal(
                _inverse(source_map), source_map.matrix
            ))


    def test_nonidentity_maps_match_original_transform(self):
        for sparse in (True, False):
            with self.subTest(sparse=sparse):
                source, stage, config, inspector = _donor(sparse=sparse)
                stage.moe_input_map = CoordinateMap(
                    np.array([
                        [2.0, 0.3, 0.0, 0.0],
                        [0.0, 1.5, 0.4, 0.0],
                        [0.0, 0.0, 1.2, 0.2],
                        [0.1, 0.0, 0.0, 1.8],
                    ]),
                    1e-6,
                )
                stage.residual_map = CoordinateMap(
                    np.diag([1.2, 0.9, 1.1, 2.2]), 1e-6
                )
                stage.latent_map = CoordinateMap(
                    np.diag([0.8, 1.1, 1.3, 1.5]), 1e-6
                )
                if not sparse:
                    stage.dense_intermediate_map = CoordinateMap(
                        np.array([
                            [1.1, 0.2, 0.0],
                            [0.0, 0.8, 0.3],
                            [0.1, 0.0, 1.4],
                        ]),
                        1e-6,
                    )
                baseline = _moe_state(
                    source=source, stage=stage, config=config, inspector=inspector
                )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    index = {}
                    save_incremental_moe_stage(
                        source=source, inspector=inspector, stage=stage,
                        config=config, output_dir=root, weight_map=index,
                    )
                    for key, shard in index.items():
                        observed = load_file(str(root / shard))[key]
                        self.assertTrue(torch.allclose(
                            observed.float(), baseline[key].float(),
                            atol=0.015, rtol=0.015,
                        ), msg=f"nonidentity mapping differs: {key}")


    def test_bf16_donor_weights_work_without_numpy_bfloat16_support(self):
        source, stage, config, inspector = _donor(sparse=True, bf16=True)
        with tempfile.TemporaryDirectory() as tmp:
            index = {}
            save_incremental_moe_stage(
                source=source,
                inspector=inspector,
                stage=stage,
                config=config,
                output_dir=Path(tmp),
                weight_map=index,
            )
            self.assertEqual(len(index), 6 + 3 + 2 * 3)
            for name in set(index.values()):
                values = load_file(str(Path(tmp) / name))
                self.assertTrue(all(bool(torch.isfinite(v).all()) for v in values.values()))


class TestExpandedCalibrationCompatibility(unittest.TestCase):
    def test_original_weight_only_maps_can_expand_expert_count(self):
        base = canonical_glm53_config()
        expanded = canonical_glm53_config(routed_experts=64)
        old = SimpleNamespace(
            target_config_fingerprint=base.fingerprint,
            calibration_mode="weight_only",
        )
        _validate_expanded_calibration(old, expanded)

    def test_activation_refined_maps_cannot_silently_expand(self):
        base = canonical_glm53_config()
        expanded = canonical_glm53_config(routed_experts=64)
        old = SimpleNamespace(
            target_config_fingerprint=base.fingerprint,
            calibration_mode="activation_refined",
        )
        with self.assertRaises(GLM53CompileError):
            _validate_expanded_calibration(old, expanded)

    def test_unrelated_fingerprint_fails_closed(self):
        expanded = canonical_glm53_config(routed_experts=64)
        wrong = SimpleNamespace(
            target_config_fingerprint="unrelated",
            calibration_mode="weight_only",
        )
        with self.assertRaises(GLM53CompileError):
            _validate_expanded_calibration(wrong, expanded)


if __name__ == "__main__":
    unittest.main()
