"""GLM SiLU expert semantics, independent from IQ default SiTU."""
from __future__ import annotations

import unittest
import torch
from torch import nn

from iq_model.mlp.latent_moe import SiTUExpert, StableLatentMoEConfig
from iq_transfer.complete_transplant import nvidia_parallel_78_config


class GLMExpertActivationTests(unittest.TestCase):
    def test_glm_variant_preserves_documented_silu_activation(self):
        target = nvidia_parallel_78_config()
        self.assertEqual(target.stable_moe.expert_activation, "silu")
        self.assertEqual(target.stable_moe.num_experts, 256)
        self.assertEqual(target.stable_moe.top_k, 8)
        self.assertEqual(StableLatentMoEConfig(
            hidden_size=8, latent_size=4,
            expert_intermediate_size=8, num_experts=4, top_k=2,
        ).expert_activation, "situ")

    def test_native_silu_expert_matches_swiglu_reference_and_gradients(self):
        torch.manual_seed(77)
        expert = SiTUExpert(
            8, 12, beta=4.0, linear_beta=25.0,
            expert_activation="silu",
        )
        x = torch.randn(3, 8, requires_grad=True)
        output = expert(x)
        reference = nn.functional.linear(
            nn.functional.silu(expert.gate_proj(x)) * expert.up_proj(x),
            expert.down_proj.weight,
            expert.down_proj.bias,
        )
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        output.square().mean().backward()
        self.assertIsNotNone(expert.gate_proj.weight.grad)
        self.assertIsNotNone(expert.up_proj.weight.grad)
        self.assertIsNotNone(expert.down_proj.weight.grad)

    def test_unknown_activation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expert_activation"):
            StableLatentMoEConfig(
                hidden_size=8, latent_size=4,
                expert_intermediate_size=8, num_experts=4, top_k=2,
                expert_activation="linear"
            )


if __name__ == "__main__":
    unittest.main()
