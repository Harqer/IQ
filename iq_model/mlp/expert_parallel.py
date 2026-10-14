"""Real differentiable routed-expert parallelism for IQ Stable LatentMoE.

Only routed expert parameters are sharded. Global router, latent input/output,
normalization and shared experts are replicated, as required for this module's
existing semantics. Token dispatch uses autograd-aware torch.distributed
all-to-all collectives, not a simulated expert-count reduction.

The Megatron Core EP process group must be initialized by the application;
this module does not invent an inconsistent parallel topology or recreate
Megatron Core's training checkpointing machinery.
"""
from __future__ import annotations

from collections.abc import Iterator
from math import ceil

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_to_all_single as differentiable_all_to_all

from ..norm import RMSNorm
from .latent_moe import (
    SiTUExpert, StableLatentMoE, StableLatentMoEConfig,
    StableLatentMoEError, StableLatentMoEOutput,
)


class IQExpertParallelError(RuntimeError):
    pass


class ExpertParallelStableLatentMoE(nn.Module):
    """Shards 256 routed experts across EP ranks with gradient-safe dispatch."""

    # These operate only on config/router/routing_bias and preserve the
    # reference StableLatentMoE scoring and delayed-bias semantics.
    _route = StableLatentMoE._route
    compute_next_routing_bias = StableLatentMoE.compute_next_routing_bias
    commit_routing_bias = StableLatentMoE.commit_routing_bias

    def __init__(
        self,
        config: StableLatentMoEConfig,
        *,
        ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise IQExpertParallelError("expert parallel execution requires torch.distributed")
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=ep_group)
        self.ep_rank = dist.get_rank(group=ep_group)
        if self.ep_size <= 1 or config.num_experts % self.ep_size:
            raise IQExpertParallelError(
                "expert parallel group must divide num_experts and contain >1 ranks"
            )
        self.config = config
        self.local_experts = config.num_experts // self.ep_size
        self.expert_offset = self.ep_rank * self.local_experts
        self.router = nn.Linear(
            config.hidden_size, config.num_experts, bias=config.router_bias
        )
        self.latent_down = nn.Linear(
            config.hidden_size, config.latent_size, bias=False
        )
        self.latent_up = nn.Linear(
            config.latent_size, config.hidden_size, bias=False
        )
        self.latent_norm = RMSNorm(config.latent_size, config.rms_norm_eps)
        self.routed_experts = nn.ModuleList(
            SiTUExpert(
                config.latent_size, config.expert_intermediate_size,
                beta=config.situ_beta, linear_beta=config.situ_linear_beta,
                bias=config.expert_bias,
                expert_activation=config.expert_activation,
            )
            for _ in range(self.local_experts)
        )
        self.shared_experts = nn.ModuleList(
            SiTUExpert(
                config.hidden_size, config.expert_intermediate_size,
                beta=config.situ_beta, linear_beta=config.situ_linear_beta,
                bias=config.expert_bias,
                expert_activation=config.expert_activation,
            )
            for _ in range(config.num_shared_experts)
        )
        self.register_buffer(
            "routing_bias", torch.zeros(config.num_experts), persistent=True
        )

    def global_expert_id(self, local_index: int) -> int:
        if not (0 <= local_index < self.local_experts):
            raise IQExpertParallelError("local expert index is out of range")
        return self.expert_offset + local_index

    def owned_expert_parameters(self) -> Iterator[tuple[int, str, nn.Parameter]]:
        """Yield original GLM expert IDs, not ambiguous local shard numbering."""
        for local, expert in enumerate(self.routed_experts):
            global_id = self.global_expert_id(local)
            for name, parameter in expert.named_parameters():
                yield global_id, name, parameter

    def _distributed_routed(
        self, latent: torch.Tensor,
        selected: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Two differentiable all-to-all exchanges; no cross-rank token dropping."""
        tokens, width = latent.shape
        flat_indices = selected.reshape(-1)
        destinations = torch.div(
            flat_indices, self.local_experts, rounding_mode="floor"
        )
        # Sorting by EP destination defines matching send/return order.
        permutation = torch.argsort(destinations, stable=True)
        sorted_dest = destinations.index_select(0, permutation)
        send_counts = torch.bincount(
            sorted_dest, minlength=self.ep_size
        ).tolist()
        send_local_expert = (
            flat_indices.index_select(0, permutation)
            - sorted_dest * self.local_experts
        )
        source_rows = torch.arange(
            tokens, device=latent.device
        ).repeat_interleave(self.config.top_k)
        sent_latent = latent.index_select(
            0, source_rows.index_select(0, permutation)
        ).contiguous()
        # Communicate dynamic split sizes before dispatch. The index exchange
        # is metadata; routed activations use the differentiable collective.
        send_sizes = torch.tensor(
            send_counts, device=latent.device, dtype=torch.int64
        )
        recv_sizes = torch.empty_like(send_sizes)
        dist.all_to_all_single(recv_sizes, send_sizes, group=self.ep_group)
        receive_counts = [int(x) for x in recv_sizes.tolist()]
        receive_total = sum(receive_counts)

        recv_latent = torch.empty(
            (receive_total, width), device=latent.device, dtype=latent.dtype
        )
        recv_latent = differentiable_all_to_all(
            recv_latent, sent_latent, output_split_sizes=receive_counts,
            input_split_sizes=send_counts, group=self.ep_group,
        )
        received_local_expert = torch.empty(
            receive_total, dtype=torch.int64, device=latent.device
        )
        dist.all_to_all_single(
            received_local_expert, send_local_expert.contiguous(),
            output_split_sizes=receive_counts, input_split_sizes=send_counts,
            group=self.ep_group,
        )
        computed = torch.zeros_like(recv_latent)
        for local_index, expert in enumerate(self.routed_experts):
            rows = torch.where(received_local_expert == local_index)[0]
            if rows.numel():
                result = expert(recv_latent.index_select(0, rows))
                computed = computed.index_copy(0, rows, result)

        returned = torch.empty_like(sent_latent)
        returned = differentiable_all_to_all(
            returned, computed, output_split_sizes=send_counts,
            input_split_sizes=receive_counts, group=self.ep_group,
        )
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            permutation.numel(), device=permutation.device
        )
        returned = returned.index_select(0, inverse)
        weighted = returned * weights.reshape(-1, 1).to(returned.dtype)
        result = torch.zeros_like(latent)
        return result.index_add(0, source_rows, weighted)

    def forward(self, x: torch.Tensor) -> StableLatentMoEOutput:
        if x.ndim < 2 or x.shape[-1] != self.config.hidden_size:
            raise ValueError("expert-parallel MoE input has invalid hidden width")
        original_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        if flat.shape[0] == 0:
            raise ValueError("expert-parallel MoE requires nonempty token inputs")
        raw, selected, weights = self._route(flat)
        latent = self.latent_down(flat)
        routed = self._distributed_routed(latent, selected, weights)
        routed = self.latent_up(self.latent_norm(routed))
        shared = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared = shared + expert(flat)
        hidden = (routed + shared).reshape(original_shape)
        if not bool(torch.isfinite(hidden).all()):
            raise StableLatentMoEError("expert-parallel MoE produced nonfinite output")
        counts = torch.bincount(
            selected.reshape(-1), minlength=self.config.num_experts
        )
        return StableLatentMoEOutput(
            hidden_states=hidden,
            raw_router_scores=raw.reshape(
                *original_shape[:-1], self.config.num_experts
            ),
            selected_experts=selected.reshape(
                *original_shape[:-1], self.config.top_k
            ),
            selected_weights=weights.reshape(
                *original_shape[:-1], self.config.top_k
            ),
            expert_counts=counts,
            routing_bias=self.routing_bias.detach().clone(),
        )


class ExpertParallelStableLatentMoELayer(nn.Module):
    """Preserves IQ's stable expert layer residual semantics under EP."""

    def __init__(
        self, config: StableLatentMoEConfig, *,
        ep_group: dist.ProcessGroup, norm_eps: float = 1e-5,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= residual_dropout < 1:
            raise ValueError("residual_dropout must be in [0, 1)")
        self.norm = RMSNorm(config.hidden_size, norm_eps)
        self.moe = ExpertParallelStableLatentMoE(config, ep_group=ep_group)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> StableLatentMoEOutput:
        result = self.moe(self.norm(x))
        result.hidden_states = x + self.residual_dropout(result.hidden_states)
        return result
