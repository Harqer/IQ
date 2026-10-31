"""CPU-only tests for real-shard MoE loader; never contact HF or allocate GPU."""
import unittest

import torch
from torch import nn

from iq_model.mlp.latent_moe import StableLatentMoEConfig
from scripts.test_glm53_real_moe_layer import (
    ZeroExpert, assert_fully_materialized, assign_tensors,
    create_sparse_layer, load_expert,
)


class TestRealMoELoader(unittest.TestCase):
    def setUp(self):
        self.cfg = StableLatentMoEConfig(
            hidden_size=4, latent_size=2, expert_intermediate_size=3,
            num_experts=4, top_k=2, num_shared_experts=1,
            routed_scaling_factor=2.5,
        )
        self.prefix = "layers.11."

    def test_meta_skeleton_has_no_unselected_expert_weights(self):
        layer = create_sparse_layer(self.cfg, eps=1e-5, device=torch.device("cpu"))
        self.assertEqual(len(layer.moe.routed_experts), 4)
        self.assertTrue(all(isinstance(x, nn.Identity) for x in layer.moe.routed_experts))
        self.assertTrue(layer.moe.router.weight.is_meta)

    def test_missing_or_wrong_tensor_keys_fail_closed(self):
        layer = create_sparse_layer(self.cfg, eps=1e-5, device=torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "Wrong saved tensor keys"):
            assign_tensors(
                layer, {self.prefix + "norm.weight": torch.ones(4)},
                {self.prefix + "norm.weight", self.prefix + "moe.router.weight"},
                device=torch.device("cpu"), prefix=self.prefix,
            )

    def test_synthetic_shard_injected_into_actual_production_forward(self):
        device = torch.device("cpu")
        layer = create_sparse_layer(self.cfg, eps=1e-5, device=device)
        p = self.prefix
        common = {
            p + "norm.weight": torch.ones(4),
            p + "moe.latent_norm.weight": torch.ones(2),
            p + "moe.latent_down.weight": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
            p + "moe.latent_up.weight": torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 1.]]),
            p + "moe.router.weight": torch.tensor([
                [3., 0., 0., 0.], [2., 0., 0., 0.],
                [-2., 0., 0., 0.], [-3., 0., 0., 0.]
            ]),
            p + "moe.routing_bias": torch.zeros(4),
        }
        assign_tensors(layer, common, set(common), device=device, prefix=p)
        def expert_state(prefix, width):
            return {
                prefix + "gate_proj.weight": torch.full((3, width), 0.3),
                prefix + "up_proj.weight": torch.full((3, width), 0.2),
                prefix + "down_proj.weight": torch.full((width, 3), 0.4),
            }
        q = p + "moe.shared_experts.0."
        layer.moe.shared_experts[0] = load_expert(
            expert_state(q, 4), config=self.cfg,
            prefix=q, width=4, device=device,
        )
        x = torch.tensor([[[1., 0.5, 0.3, -0.2]]])
        _, selected, weights = layer.moe._route(layer.norm(x).reshape(1, 4))
        indices = sorted(int(i) for i in selected.flatten().tolist())
        self.assertEqual(indices, [0, 1])
        for i in indices:
            q = p + f"moe.routed_experts.{i}."
            layer.moe.routed_experts[i] = load_expert(
                expert_state(q, 2), config=self.cfg,
                prefix=q, width=2, device=device,
            )
        assert_fully_materialized(layer)
        output = layer(x)
        self.assertTrue(torch.isfinite(output.hidden_states).all())
        self.assertEqual(int(output.expert_counts.sum()), 2)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.tensor([2.5])))
        selected_id = indices[0]
        saved = layer.moe.routed_experts[selected_id]
        layer.moe.routed_experts[selected_id] = ZeroExpert()
        ablated = layer(x)
        layer.moe.routed_experts[selected_id] = saved
        self.assertGreater(
            float((ablated.hidden_states - output.hidden_states).abs().max()), 0.
        )


if __name__ == "__main__":
    unittest.main()
