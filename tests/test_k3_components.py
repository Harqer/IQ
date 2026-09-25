from __future__ import annotations

import unittest

import torch

from iq_model import (
    AttentionResidualMixer,
    BlockAttentionResidual,
    BlockAttnResConfig,
    SiTUAndMul,
    StableLatentMoE,
    StableLatentMoEConfig,
)


class K3InspiredComponentsTests(unittest.TestCase):
    def test_situ_is_bounded_and_differentiable(self):
        activation = SiTUAndMul(beta=4.0, linear_beta=25.0)
        gate = torch.tensor([[-1000.0, -1.0, 1.0, 1000.0]], requires_grad=True)
        up = torch.tensor([[1000.0, -3.0, 2.0, -1000.0]], requires_grad=True)
        out = activation(gate, up)
        self.assertTrue(torch.isfinite(out).all())
        # |gate branch| <= beta and |linear branch| <= linear_beta.
        self.assertLessEqual(float(out.detach().abs().max()), 4.0 * 25.0 + 1e-4)
        out.sum().backward()
        self.assertIsNotNone(gate.grad)
        self.assertIsNotNone(up.grad)
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertTrue(torch.isfinite(up.grad).all())

    def config(self) -> StableLatentMoEConfig:
        return StableLatentMoEConfig(
            hidden_size=16,
            latent_size=8,
            expert_intermediate_size=12,
            num_experts=8,
            top_k=2,
            num_shared_experts=2,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )

    def test_stable_latent_moe_shapes_routing_and_gradients(self):
        torch.manual_seed(51)
        moe = StableLatentMoE(self.config())
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = moe(x)

        self.assertEqual(tuple(out.hidden_states.shape), (2, 5, 16))
        self.assertEqual(tuple(out.raw_router_scores.shape), (2, 5, 8))
        self.assertEqual(tuple(out.selected_experts.shape), (2, 5, 2))
        self.assertEqual(tuple(out.selected_weights.shape), (2, 5, 2))
        self.assertEqual(int(out.expert_counts.sum()), 2 * 5 * 2)
        self.assertTrue(
            torch.allclose(
                out.selected_weights.float().sum(dim=-1),
                torch.ones(2, 5),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertEqual(moe.latent_down.out_features, 8)
        self.assertEqual(moe.latent_up.in_features, 8)
        self.assertEqual(len(moe.shared_experts), 2)
        self.assertEqual(moe.shared_experts[0].gate_proj.in_features, 16)
        self.assertEqual(moe.routed_experts[0].gate_proj.in_features, 8)

        out.hidden_states.square().mean().backward()
        required = [
            moe.router.weight,
            moe.latent_down.weight,
            moe.latent_up.weight,
            moe.latent_norm.weight,
        ]
        for parameter in required:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_quantile_balancing_is_delayed_and_mean_centered(self):
        torch.manual_seed(52)
        moe = StableLatentMoE(self.config())
        x = torch.randn(4, 7, 16)
        first = moe(x)
        committed_before = moe.routing_bias.detach().clone()
        next_bias = moe.compute_next_routing_bias(first.raw_router_scores)

        self.assertTrue(torch.equal(moe.routing_bias, committed_before))
        self.assertAlmostEqual(float(next_bias.mean()), 0.0, places=6)
        self.assertTrue(torch.isfinite(next_bias).all())

        moe.commit_routing_bias(next_bias)
        self.assertTrue(torch.allclose(moe.routing_bias, next_bias))
        second = moe(x)
        self.assertTrue(torch.equal(second.routing_bias, next_bias))

    def test_attnres_embedding_and_block_sources_are_separate(self):
        torch.manual_seed(53)
        config = BlockAttnResConfig(
            hidden_size=8,
            block_size=2,
            rms_norm_eps=1e-6,
        )
        attnres = BlockAttentionResidual(config)
        embeddings = torch.randn(1, 3, 8)
        state = attnres.init_state(embeddings)

        self.assertEqual(tuple(state.block_sources.shape), (1, 3, 1, 8))
        self.assertTrue(torch.equal(state.block_sources[:, :, 0], embeddings))
        self.assertTrue(torch.equal(attnres.read(state), embeddings))

        layer0 = torch.randn_like(embeddings)
        state = attnres.advance(state, layer0)
        self.assertEqual(state.layer_index, 1)
        self.assertEqual(state.block_sources.shape[2], 1)
        mixed = attnres.read(state)
        self.assertEqual(tuple(mixed.shape), tuple(embeddings.shape))

        layer1 = torch.randn_like(embeddings)
        state = attnres.advance(state, layer1)
        self.assertEqual(state.layer_index, 2)
        # embedding + completed first block
        self.assertEqual(state.block_sources.shape[2], 2)
        self.assertTrue(torch.equal(state.prefix_sum, torch.zeros_like(state.prefix_sum)))

        final = attnres.finalize(state)
        self.assertEqual(tuple(final.shape), tuple(embeddings.shape))
        self.assertTrue(torch.isfinite(final).all())

    def test_attnres_mixer_softmax_depth_selection(self):
        config = BlockAttnResConfig(hidden_size=4, block_size=2)
        mixer = AttentionResidualMixer(config)
        with torch.no_grad():
            mixer.score.fill_(1.0)
        prefix = torch.ones(1, 1, 4)
        sources = torch.stack(
            [
                torch.ones(1, 1, 4) * 2.0,
                torch.ones(1, 1, 4) * 3.0,
            ],
            dim=2,
        )
        out = mixer(prefix, sources)
        self.assertEqual(tuple(out.shape), (1, 1, 4))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
