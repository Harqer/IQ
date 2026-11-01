"""CPU orchestration tests for H200 *production-geometry* CUDA gate.

The real native kernel runs only in user-authorized H200 sessions; this test
exercises the geometry, first/second mixer and prefill/step comparison wiring.
It must not be represented as GPU validation.
"""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.verify_iq_mamba3_native_step import verify_full78_mimo_geometry


class FakeMamba:
    instances = []
    def __init__(self, **options):
        self.options = options
        self.layer_idx = options["layer_idx"]
        FakeMamba.instances.append(self)
    def eval(self):
        return self
    def __call__(self, tokens, inference_params=None):
        if inference_params is not None:
            zero = torch.zeros_like(tokens[:, 0, :])
            summed = tokens.sum(dim=1)
            inference_params.key_value_memory_dict[self.layer_idx] = (
                zero.clone(), summed, zero.clone(), zero.clone(),
            )
        return tokens + 0.25 * torch.cumsum(tokens, dim=1)
    def _get_states_from_cache(self, inference_params, batch_size):
        if self.layer_idx not in inference_params.key_value_memory_dict:
            zero = torch.zeros((batch_size, self.options["d_model"]), dtype=torch.bfloat16)
            inference_params.key_value_memory_dict[self.layer_idx] = (
                zero.clone(), zero.clone(), zero.clone(), zero.clone(),
            )
        return inference_params.key_value_memory_dict[self.layer_idx]

    def step(self, token, angle, state, k, v):
        next_state = state + token
        return token + 0.25 * next_state, angle, next_state, k, v


class ProductionGeometryTests(unittest.TestCase):
    def setUp(self):
        FakeMamba.instances = []

    def test_exact_width_4096_two_mixer_stateful_parity_contract(self):
        config = SimpleNamespace(
            d_model=4096, d_state=128, expand=2.0, headdim=64,
            rope_fraction=0.5, outproj_norm=False,
            mimo_rank=4, chunk_size=16, num_layers=79,
        )
        original_randn = torch.randn
        def make_cpu_random(shape, **kwargs):
            kwargs["device"] = "cpu"
            return original_randn(shape, **kwargs)
        with patch.dict(sys.modules, {"mamba_ssm": SimpleNamespace(Mamba3=FakeMamba)}), \
             patch("scripts.verify_iq_mamba3_native_step.torch.cuda.is_available",
                   return_value=True), \
             patch("scripts.verify_iq_mamba3_native_step.torch.cuda.synchronize"), \
             patch("scripts.verify_iq_mamba3_native_step.torch.cuda.empty_cache"), \
             patch("scripts.verify_iq_mamba3_native_step.torch.randn",
                   side_effect=make_cpu_random):
            report = verify_full78_mimo_geometry(torch.device("cuda:0"), config)
        self.assertEqual(report["width"], 4096)
        self.assertEqual(report["checked_lengths"], [13, 16, 17, 18, 33])
        self.assertLess(report["max_absolute_error"], 0.01)
        self.assertEqual([m.layer_idx for m in FakeMamba.instances], [0, 1])
        self.assertEqual(FakeMamba.instances[0].options["d_model"], 4096)
        self.assertEqual(FakeMamba.instances[0].options["mimo_rank"], 4)

    def test_fail_closed_on_wrong_rank_and_width(self):
        config = SimpleNamespace(
            d_model=256, d_state=128, expand=2.0, headdim=64,
            rope_fraction=0.5, outproj_norm=False,
            mimo_rank=1, chunk_size=16, num_layers=79,
        )
        with patch.dict(sys.modules, {"mamba_ssm": SimpleNamespace(Mamba3=FakeMamba)}), \
             patch("scripts.verify_iq_mamba3_native_step.torch.cuda.is_available",
                   return_value=True):
            with self.assertRaisesRegex(ValueError, "Expected full78 native width"):
                verify_full78_mimo_geometry(torch.device("cuda:0"), config)


if __name__ == "__main__":
    unittest.main()
