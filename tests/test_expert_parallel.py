"""Real two-rank Gloo expert-parallel parity test with gradient verification.

Uses actual stable latent experts, router and differentiable distributed
all-to-all, not mocked collectives or a simulated expert subset.
"""
from __future__ import annotations

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from iq_model.mlp import (
    ExpertParallelStableLatentMoE, StableLatentMoE, StableLatentMoEConfig,
    IQExpertParallelError,
)


def _check_two_rank_parity(rank: int, size: int, init_url: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_url, rank=rank,
                            world_size=size)
    try:
        config = StableLatentMoEConfig(
            hidden_size=8, latent_size=6, expert_intermediate_size=12,
            num_experts=4, top_k=2, num_shared_experts=1,
            situ_beta=4.0, situ_linear_beta=25.0,
            expert_activation="silu",
        )
        torch.manual_seed(2026)
        reference = StableLatentMoE(config)
        torch.manual_seed(600 + rank)
        distributed = ExpertParallelStableLatentMoE(config, ep_group=dist.group.WORLD)
        with torch.no_grad():
            for key in ("router", "latent_down", "latent_up",
                        "latent_norm", "shared_experts"):
                getattr(distributed, key).load_state_dict(
                    getattr(reference, key).state_dict()
                )
            distributed.routing_bias.copy_(reference.routing_bias)
            for local in range(distributed.local_experts):
                global_id = distributed.global_expert_id(local)
                distributed.routed_experts[local].load_state_dict(
                    reference.routed_experts[global_id].state_dict()
                )
        assert distributed.local_experts == 2
        assert [distributed.global_expert_id(i) for i in range(2)] == [
            2 * rank, 2 * rank + 1
        ]

        torch.manual_seed(1600 + rank)
        x = torch.randn((2 + rank, 8), requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)

        ref = reference(ref_x)
        got = distributed(x)
        torch.testing.assert_close(got.hidden_states, ref.hidden_states,
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(got.selected_weights, ref.selected_weights,
                                   rtol=0, atol=0)
        assert torch.equal(got.selected_experts, ref.selected_experts)
        assert torch.equal(got.expert_counts, ref.expert_counts)
        got.hidden_states.square().sum().backward()
        ref.hidden_states.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-4, atol=2e-4)

        # Each sharded expert's gradient combines tokens from both ranks.
        # Reduce the full reference's corresponding gradient to compare.
        for global_id in range(config.num_experts):
            for name, ref_param in reference.routed_experts[global_id].named_parameters():
                global_grad = (
                    ref_param.grad if ref_param.grad is not None
                    else torch.zeros_like(ref_param)
                )
                # All ranks reduce the SAME global expert key in the SAME
                # order, even when only one rank owns its trained weights.
                dist.all_reduce(global_grad, op=dist.ReduceOp.SUM)
                if global_id // distributed.local_experts == rank:
                    local = global_id - distributed.expert_offset
                    local_param = dict(
                        distributed.routed_experts[local].named_parameters()
                    )[name]
                    local_grad = (
                        local_param.grad if local_param.grad is not None
                        else torch.zeros_like(local_param)
                    )
                    torch.testing.assert_close(
                        local_grad, global_grad, rtol=2e-4, atol=2e-4
                    )
    finally:
        dist.destroy_process_group()


class ExpertParallelTests(unittest.TestCase):
    def test_real_two_rank_forward_backward_with_variable_token_counts(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("CPU Gloo process group is unavailable")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(
            _check_two_rank_parity, args=(2, f"tcp://127.0.0.1:{port}"),
            nprocs=2, join=True,
        )

    def test_uninitialized_process_group_rejected(self):
        if dist.is_initialized():
            self.skipTest("parent process unexpectedly has a distributed group")
        config = StableLatentMoEConfig(
            hidden_size=8, latent_size=6, expert_intermediate_size=12,
            num_experts=4, top_k=2,
        )
        with self.assertRaisesRegex(IQExpertParallelError, "requires torch.distributed"):
            ExpertParallelStableLatentMoE(config, ep_group=None)


if __name__ == "__main__":
    unittest.main()
