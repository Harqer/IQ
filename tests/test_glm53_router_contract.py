"""GLM-5.3 routing parity vs the publisher's documented Torch operations.

Independent functional reference is transcribed from Hugging Face
GlmMoeDsaTopkRouter.forward (Transformers glm_moe_dsa), with no fake
checkpoint weights and no model-identity assumptions.
"""
from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from iq_model.mlp.latent_moe import StableLatentMoE, StableLatentMoEConfig
from iq_transfer.complete_transplant import nvidia_parallel_78_config


def publisher_router(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
    *, groups: int, topk_groups: int, k: int, scaling: float,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reference directly from GlmMoeDsaTopkRouter.forward:
    logits = F.linear(x.float(), w.float())
    scores = logits.sigmoid()
    choice = scores + bias.float()
    grouped = choice.reshape(-1, groups, w.shape[0] // groups)
    grouped_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = grouped_scores.topk(topk_groups, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(grouped_scores)
    group_mask.scatter_(1, selected_groups, 1)
    full_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape(-1, w.shape[0])
    choice = choice.masked_fill(~full_mask.bool(), float("-inf"))
    selected = choice.topk(k, dim=-1, sorted=False).indices
    weights = scores.gather(1, selected)
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return scores, selected, weights * scaling


class GLM53RouterTests(unittest.TestCase):
    def test_full_glm_78_recipe_router_constants(self):
        cfg = nvidia_parallel_78_config()
        expert = cfg.stable_moe
        self.assertEqual(expert.routed_scaling_factor, 2.5)
        self.assertEqual(expert.router_groups, 1)
        self.assertEqual(expert.router_topk_groups, 1)
        self.assertTrue(expert.router_fp32)
        self.assertTrue(expert.normalize_topk_prob)
        self.assertEqual(expert.expert_activation, "silu")
        self.assertEqual(expert.num_experts, 256)

    def test_reference_forward_matches_full_256_expert_glm_scores(self):
        torch.manual_seed(161)
        c = StableLatentMoEConfig(
            hidden_size=12, latent_size=4, expert_intermediate_size=6,
            num_experts=256, top_k=8, num_shared_experts=1,
            expert_activation="silu",
            routed_scaling_factor=2.5, router_groups=1,
            router_topk_groups=1, router_fp32=True,
        )
        module = StableLatentMoE(c)
        x = torch.randn(7, 12, requires_grad=True)
        with torch.no_grad():
            module.routing_bias.copy_(torch.randn(256) * 0.015)
        got_scores, got_ids, got_weights = module._route(x)
        expected_scores, expected_ids, expected_weights = publisher_router(
            x, module.router.weight, module.routing_bias,
            groups=1, topk_groups=1, k=8, scaling=2.5, normalize=True,
        )
        torch.testing.assert_close(got_scores, expected_scores, atol=0, rtol=0)
        torch.testing.assert_close(got_ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(got_weights, expected_weights, atol=2e-6, rtol=2e-6)
        got_weights.sum().backward()
        self.assertIsNotNone(module.router.weight.grad)
        self.assertTrue(torch.isfinite(module.router.weight.grad).all())

    def test_grouped_router_preserves_publisher_group_filtering(self):
        torch.manual_seed(16)
        c = StableLatentMoEConfig(
            hidden_size=8, latent_size=4, expert_intermediate_size=8,
            num_experts=16, top_k=4, num_shared_experts=1,
            router_groups=4, router_topk_groups=2,
            routed_scaling_factor=1.7, router_fp32=True,
        )
        module = StableLatentMoE(c)
        with torch.no_grad():
            module.routing_bias.copy_(torch.randn(16) * 0.03)
        x = torch.randn(9, 8)
        _, actual_ids, actual_weights = module._route(x)
        _, expected_ids, expected_weights = publisher_router(
            x, module.router.weight, module.routing_bias,
            groups=4, topk_groups=2, k=4, scaling=1.7, normalize=True
        )
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        torch.testing.assert_close(actual_weights, expected_weights, atol=2e-6, rtol=2e-6)

    def test_config_rejects_impossible_grouping(self):
        source = StableLatentMoEConfig(
            hidden_size=8, latent_size=4, expert_intermediate_size=8,
            num_experts=16, top_k=4, num_shared_experts=1,
        )
        for opts in ({"router_groups": 3}, {"router_groups": 0},
                     {"router_groups": 2, "router_topk_groups": 3}):
            with self.assertRaisesRegex(ValueError, "router"):
                replace(source, **opts)


if __name__ == "__main__":
    unittest.main()
