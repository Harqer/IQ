"""Regression tests for the official BF16 GLM-5.3 calibration input types.

The full-78 compiler reads original BF16 tensors over HTTP. NumPy cannot
materialize PyTorch BF16 directly: do not allow any path to use tensor.numpy()
before promoting to FP32.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from iq_transfer.glm53_direct import _array as direct_array
from iq_transfer.glm53_dsa import _array as dsa_array
from iq_transfer.glm53_moe import (
    GLM53MoETransformError,
    _array as moe_array,
    router_weight_prior,
)
from iq_transfer.transport import _numpy as transport_array


class Full78BF16Tests(unittest.TestCase):
    def test_all_transport_converters_accept_original_bf16(self):
        original = torch.tensor(
            [[-1.5, 0.0], [0.25, 3.5]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        baseline = original.detach().float().double().numpy()
        for converter in (direct_array, dsa_array, moe_array, transport_array):
            with self.subTest(converter=converter.__module__):
                converted = converter(original)
                np.testing.assert_array_equal(converted, baseline)
                self.assertEqual(converted.dtype, np.float64)
        self.assertEqual(original.dtype, torch.bfloat16)
        self.assertTrue(original.requires_grad)

    def test_router_weight_prior_accepts_bf16_cloud_router(self):
        weights = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8)
        weights = (weights / 1000).to(dtype=torch.bfloat16)
        bias = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).to(torch.bfloat16)
        prior = router_weight_prior(weights, bias)
        self.assertEqual(prior.shape, (256,))
        self.assertTrue(np.isfinite(prior).all())
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=12)
        expected = router_weight_prior(weights.float(), bias.float())
        np.testing.assert_allclose(prior, expected, rtol=0, atol=0)
        self.assertEqual(weights.dtype, torch.bfloat16)
        self.assertEqual(bias.dtype, torch.bfloat16)

    def test_nonfinite_bf16_still_rejected(self):
        with self.assertRaises(GLM53MoETransformError):
            moe_array(torch.tensor([float("nan")], dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
