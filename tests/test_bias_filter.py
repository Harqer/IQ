from __future__ import annotations

import unittest

import torch

from iq_transfer.bias_filter import (
    BiasFilterConfig,
    CapabilityGate,
    fit_bias_filter,
)


class BiasFilterTests(unittest.TestCase):
    def test_filter_removes_linear_concept_with_small_nuisance_change(self):
        torch.manual_seed(7)
        n, d = 4096, 16
        concept = torch.randint(0, 2, (n,), dtype=torch.float64) * 2 - 1
        nuisance = torch.randn(n, d, dtype=torch.float64)
        direction = torch.zeros(d, dtype=torch.float64)
        direction[3] = 3.0
        direction[5] = -2.0
        x = nuisance + concept[:, None] * direction

        fitted = fit_bias_filter(
            x,
            concept,
            config=BiasFilterConfig(shrinkage=0.02),
        )
        scrubbed = fitted.apply(x)

        before = torch.linalg.norm((x - x.mean(0)).T @ concept)
        after = torch.linalg.norm((scrubbed - scrubbed.mean(0)).T @ concept)
        self.assertLess(after, before * 0.02)
        self.assertGreater(
            scrubbed.var(dim=0).mean(),
            0.4 * x.var(dim=0).mean(),
        )

    def test_protected_coordinate_is_bitwise_preserved(self):
        torch.manual_seed(11)
        n, d = 1024, 8
        concept = torch.randint(0, 2, (n,), dtype=torch.float64)
        x = torch.randn(n, d, dtype=torch.float64)
        x[:, 0] = 1e4 * torch.randn(n, dtype=torch.float64)
        x[:, 4] += concept * 4.0

        fitted = fit_bias_filter(x, concept, protected_indices=(0,))
        scrubbed = fitted.apply(x)
        self.assertTrue(torch.equal(scrubbed[:, 0], x[:, 0]))

    def test_capability_gate_preserves_nlp_coding_and_optional_multimodal(self):
        gate = CapabilityGate(
            min_bias_reduction=0.25,
            max_nlp_drop=0.01,
            max_coding_drop=0.01,
            max_multimodal_drop=0.01,
        )
        self.assertTrue(
            gate.accepts(
                bias_before=0.40,
                bias_after=0.20,
                nlp_before=0.80,
                nlp_after=0.795,
                coding_before=0.75,
                coding_after=0.748,
                multimodal_before=0.70,
                multimodal_after=0.696,
            )
        )
        self.assertFalse(
            gate.accepts(
                bias_before=0.40,
                bias_after=0.20,
                nlp_before=0.80,
                nlp_after=0.795,
                coding_before=0.75,
                coding_after=0.72,
            )
        )


if __name__ == "__main__":
    unittest.main()
