from __future__ import annotations

import unittest

import numpy as np

from iq_transfer import (
    CoordinateMap,
    GLM53DirectTransformError,
    GLM53MLALayout,
    fit_mla_compressed_subspace,
    fit_orthogonal_subspace,
    transform_glm53_mla,
)


class GLM53DirectTransformTests(unittest.TestCase):
    def layout(self) -> GLM53MLALayout:
        return GLM53MLALayout(
            source_hidden_size=4,
            num_heads=2,
            q_lora_rank=3,
            kv_lora_rank=2,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            v_head_dim=2,
            target_hidden_size=4,
            target_head_dim=3,
            output_groups=2,
            output_rank=3,
        )

    def test_orthogonal_subspace_is_deterministic_and_orthonormal(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(32, 7))
        a = fit_orthogonal_subspace(x, target_features=4)
        b = fit_orthogonal_subspace(x, target_features=4)
        self.assertTrue(np.allclose(a.matrix, b.matrix))
        self.assertTrue(
            np.allclose(a.matrix.T @ a.matrix, np.eye(4), atol=1e-8)
        )

    def test_mla_subspace_preserves_rotary_tail_exactly(self):
        rng = np.random.default_rng(73)
        activations = rng.normal(size=(64, 10))
        mapping = fit_mla_compressed_subspace(
            activations,
            kv_lora_rank=6,
            rope_dim=4,
            target_latent_dim=3,
        )
        self.assertEqual(mapping.matrix.shape, (10, 7))
        self.assertTrue(np.allclose(mapping.matrix[:6, 3:], 0.0))
        self.assertTrue(np.allclose(mapping.matrix[6:, :3], 0.0))
        self.assertTrue(np.array_equal(mapping.matrix[6:, 3:], np.eye(4)))
        self.assertTrue(
            np.allclose(mapping.matrix.T @ mapping.matrix, np.eye(7), atol=1e-8)
        )

    def test_mla_refactor_preserves_scores_and_value_output_at_full_latent_rank(self):
        rng = np.random.default_rng(72)
        layout = self.layout()
        q_b = rng.normal(
            size=(layout.num_heads * layout.qk_head_dim, layout.q_lora_rank)
        )
        kv_a = rng.normal(
            size=(layout.compressed_kv_dim, layout.source_hidden_size)
        )
        kv_b = rng.normal(
            size=(
                layout.num_heads
                * (layout.qk_nope_head_dim + layout.v_head_dim),
                layout.kv_lora_rank,
            )
        )
        o = rng.normal(
            size=(layout.source_hidden_size, layout.num_heads * layout.v_head_dim)
        )
        residual = CoordinateMap(np.eye(4), 1e-6, "glm.residual", "iq.residual")
        latent = CoordinateMap(
            np.eye(layout.compressed_kv_dim),
            1e-6,
            "glm.kv",
            "iq.kv",
        )
        transformed = transform_glm53_mla(
            q_b_weight=q_b,
            kv_a_weight=kv_a,
            kv_b_weight=kv_b,
            o_weight=o,
            residual_input_map=residual,
            residual_output_map=residual,
            compressed_kv_map=latent,
            layout=layout,
        )

        query_residual = rng.normal(size=(5, layout.q_lora_rank))
        key_hidden = rng.normal(size=(7, layout.source_hidden_size))
        compressed = key_hidden @ kv_a.T
        target_kv = key_hidden @ transformed.kv_weight.T

        donor_scores = []
        target_scores = []
        donor_head_outputs = []
        target_head_outputs = []
        weights = rng.random(size=(layout.num_heads, 5, 7))
        weights /= weights.sum(axis=-1, keepdims=True)
        kv_stride = layout.qk_nope_head_dim + layout.v_head_dim
        for head in range(layout.num_heads):
            q_start = head * layout.qk_head_dim
            q = query_residual @ q_b[
                q_start : q_start + layout.qk_head_dim
            ].T
            kv_start = head * kv_stride
            k_expand = kv_b[
                kv_start : kv_start + layout.qk_nope_head_dim
            ]
            v_expand = kv_b[
                kv_start + layout.qk_nope_head_dim : kv_start + kv_stride
            ]
            key = np.concatenate(
                (
                    compressed[:, : layout.kv_lora_rank] @ k_expand.T,
                    compressed[:, layout.kv_lora_rank :],
                ),
                axis=1,
            )
            donor_scores.append(q @ key.T)

            tq_start = head * layout.target_head_dim
            target_q = query_residual @ transformed.q_b_weight[
                tq_start : tq_start + layout.target_head_dim
            ].T
            target_scores.append(target_q @ target_kv.T)

            donor_value = (
                compressed[:, : layout.kv_lora_rank] @ v_expand.T
            )
            donor_head_outputs.append(weights[head] @ donor_value)
            target_head_outputs.append(weights[head] @ target_kv)

        self.assertTrue(
            np.allclose(np.stack(donor_scores), np.stack(target_scores), atol=1e-9)
        )

        donor_concat = np.concatenate(donor_head_outputs, axis=-1)
        donor_output = donor_concat @ o.T
        target_concat = np.concatenate(target_head_outputs, axis=-1)
        # The grouped reduction is identity in this exact-rank control.
        group = transformed.output_group_weight
        grouped = target_concat.reshape(5, layout.output_groups, -1)
        reduced = np.einsum("tgi,gri->tgr", grouped, group).reshape(5, -1)
        target_output = reduced @ transformed.output_weight.T
        self.assertTrue(np.allclose(donor_output, target_output, atol=1e-9))

    def test_mla_refactor_rejects_nonorthogonal_latent_map(self):
        layout = self.layout()
        bad = CoordinateMap(
            np.ones((layout.compressed_kv_dim, layout.target_head_dim)),
            1e-6,
        )
        identity = CoordinateMap(np.eye(4), 1e-6)
        with self.assertRaisesRegex(
            GLM53DirectTransformError, "orthonormal"
        ):
            transform_glm53_mla(
                q_b_weight=np.zeros(
                    (layout.num_heads * layout.qk_head_dim, layout.q_lora_rank)
                ),
                kv_a_weight=np.zeros(
                    (layout.compressed_kv_dim, layout.source_hidden_size)
                ),
                kv_b_weight=np.zeros(
                    (
                        layout.num_heads
                        * (layout.qk_nope_head_dim + layout.v_head_dim),
                        layout.kv_lora_rank,
                    )
                ),
                o_weight=np.zeros(
                    (
                        layout.source_hidden_size,
                        layout.num_heads * layout.v_head_dim,
                    )
                ),
                residual_input_map=identity,
                residual_output_map=identity,
                compressed_kv_map=bad,
                layout=layout,
            )


if __name__ == "__main__":
    unittest.main()
