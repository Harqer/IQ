from __future__ import annotations

import unittest

import numpy as np

from iq_transfer import (
    CoordinateMap,
    GLM53DSATransformError,
    resolve_indexer_source_layer,
    transform_glm53_dsa_indexer,
)


class GLM53DSATransformTests(unittest.TestCase):
    def test_shared_indexer_resolves_previous_full_layer(self):
        kinds = ("full", "shared", "shared", "full", "shared")
        self.assertEqual(resolve_indexer_source_layer(kinds, 0), 0)
        self.assertEqual(resolve_indexer_source_layer(kinds, 2), 0)
        self.assertEqual(resolve_indexer_source_layer(kinds, 4), 3)

    def test_direct_indexer_refactor_preserves_pre_rope_scores(self):
        rng = np.random.default_rng(91)
        hidden = 6
        q_rank = 5
        heads = 3
        head_dim = 4
        rope = 2
        q_weight = rng.normal(size=(heads * head_dim, q_rank))
        k_weight = rng.normal(size=(head_dim, hidden))
        head_weight = rng.normal(size=(heads, hidden))
        norm_weight = rng.normal(size=head_dim)
        norm_bias = rng.normal(size=head_dim)
        residual = CoordinateMap(np.eye(hidden), 1e-6)

        out = transform_glm53_dsa_indexer(
            q_weight=q_weight,
            k_weight=k_weight,
            head_weight=head_weight,
            k_norm_weight=norm_weight,
            k_norm_bias=norm_bias,
            residual_map=residual,
            num_heads=heads,
            head_dim=head_dim,
            rope_dim=rope,
            q_lora_rank=q_rank,
        )

        q_resid = rng.normal(size=(7, q_rank))
        h = rng.normal(size=(9, hidden))
        donor_q = (q_resid @ q_weight.T).reshape(7, heads, head_dim)
        donor_k_pre = h @ k_weight.T

        donor_k = (
            (donor_k_pre - donor_k_pre.mean(axis=-1, keepdims=True))
            / np.sqrt(donor_k_pre.var(axis=-1, keepdims=True) + 1e-5)
        ) * norm_weight + norm_bias
        perm = np.array([2, 3, 0, 1])
        target_q = (q_resid @ out.q_weight.T).reshape(7, heads, head_dim)
        target_k_pre = h @ out.k_weight.T
        target_k = (
            (target_k_pre - target_k_pre.mean(axis=-1, keepdims=True))
            / np.sqrt(target_k_pre.var(axis=-1, keepdims=True) + 1e-5)
        ) * out.k_norm_weight + out.k_norm_bias

        donor_scores = np.einsum("qhd,kd->qhk", donor_q, donor_k)
        target_scores = np.einsum("qhd,kd->qhk", target_q, target_k)
        self.assertTrue(np.allclose(target_q, donor_q[:, :, perm]))
        self.assertTrue(np.allclose(target_k, donor_k[:, perm]))
        self.assertTrue(np.allclose(donor_scores, target_scores, atol=1e-9))
        self.assertTrue(np.allclose(out.head_weight, head_weight))

    def test_shared_indexer_without_full_predecessor_fails(self):
        with self.assertRaisesRegex(GLM53DSATransformError, "preceding full"):
            resolve_indexer_source_layer(("shared",), 0)


if __name__ == "__main__":
    unittest.main()
