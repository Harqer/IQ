from __future__ import annotations

import unittest

import numpy as np

from iq_transfer import (
    CoordinateMap,
    ExpertWeights,
    GLM53MoETransformError,
    latent_codec_weights,
    router_usage_from_topk,
    select_experts_by_usage,
    transform_glm53_moe,
)


class GLM53MoETransformTests(unittest.TestCase):
    def test_router_usage_and_selection_are_deterministic(self):
        usage = router_usage_from_topk(
            [
                np.array([[2, 1], [2, 0]]),
                np.array([[3, 2], [1, 2]]),
            ],
            num_experts=4,
        )
        self.assertTrue(np.allclose(usage, [1 / 8, 2 / 8, 4 / 8, 1 / 8]))
        self.assertEqual(
            select_experts_by_usage(usage, target_experts=2),
            (2, 1),
        )

    def test_latent_codec_matches_donor_coordinate_maps(self):
        rng = np.random.default_rng(81)
        donor = rng.normal(size=(128, 6))
        residual_basis = rng.normal(size=(6, 4))
        latent_basis = rng.normal(size=(6, 3))
        residual = CoordinateMap(residual_basis, 1e-6)
        latent = CoordinateMap(latent_basis, 1e-6)
        down, up = latent_codec_weights(residual, latent)
        target_residual = donor @ residual_basis
        expected_latent = donor @ latent_basis
        actual_latent = target_residual @ down.T
        # When the residual map has lower rank than the donor width, this is
        # the least-squares latent reconstruction from the retained residual.
        baseline = donor @ residual_basis @ np.linalg.pinv(residual_basis) @ latent_basis
        self.assertTrue(np.allclose(actual_latent, baseline, atol=1e-9))
        reconstructed_residual = expected_latent @ up.T
        expected_residual_projection = (
            donor @ latent_basis @ np.linalg.pinv(latent_basis) @ residual_basis
        )
        self.assertTrue(
            np.allclose(reconstructed_residual, expected_residual_projection, atol=1e-9)
        )

    def test_direct_moe_transform_preserves_selected_linear_experts_in_full_rank_control(self):
        rng = np.random.default_rng(82)
        hidden = 4
        intermediate = 3
        num_experts = 4
        residual = CoordinateMap(np.eye(hidden), 1e-6)
        latent = CoordinateMap(np.eye(hidden), 1e-6)
        middle = CoordinateMap(np.eye(intermediate), 1e-6)

        experts = tuple(
            ExpertWeights(
                gate=rng.normal(size=(intermediate, hidden)),
                up=rng.normal(size=(intermediate, hidden)),
                down=rng.normal(size=(hidden, intermediate)),
            )
            for _ in range(num_experts)
        )
        shared = ExpertWeights(
            gate=rng.normal(size=(intermediate, hidden)),
            up=rng.normal(size=(intermediate, hidden)),
            down=rng.normal(size=(hidden, intermediate)),
        )
        router = rng.normal(size=(num_experts, hidden))
        usage = np.array([0.1, 0.4, 0.2, 0.3])
        result = transform_glm53_moe(
            router_weight=router,
            routed_experts=experts,
            shared_expert=shared,
            expert_usage=usage,
            target_experts=2,
            residual_map=residual,
            latent_map=latent,
            intermediate_map=middle,
        )
        self.assertEqual(result.source_expert_indices, (1, 3))
        self.assertTrue(np.array_equal(result.router_weight, router[[1, 3]]))
        self.assertTrue(np.array_equal(result.latent_down_weight, np.eye(hidden)))
        self.assertTrue(np.array_equal(result.latent_up_weight, np.eye(hidden)))
        for target, source_index in zip(
            result.routed_experts,
            result.source_expert_indices,
            strict=True,
        ):
            source = experts[source_index]
            self.assertTrue(np.allclose(target.gate, source.gate))
            self.assertTrue(np.allclose(target.up, source.up))
            self.assertTrue(np.allclose(target.down, source.down))
        self.assertTrue(np.allclose(result.shared_expert.gate, shared.gate))

    def test_empty_router_capture_fails_closed(self):
        with self.assertRaisesRegex(GLM53MoETransformError, "empty"):
            router_usage_from_topk([], num_experts=4)


if __name__ == "__main__":
    unittest.main()
