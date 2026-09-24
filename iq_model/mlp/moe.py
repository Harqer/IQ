from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm


class MoEConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RoutedMoEConfig:
    hidden_size: int
    expert_intermediate_size: int
    num_experts: int
    top_k: int
    shared_expert_intermediate_size: int | None = None
    router_bias: bool = False
    capacity_factor: float | None = None
    overflow_policy: Literal["unbounded", "error"] = "unbounded"

    def __post_init__(self) -> None:
        ints = {
            "hidden_size": self.hidden_size,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise MoEConfigError(
                f"positive MoE fields required: {', '.join(bad)}"
            )
        if self.top_k > self.num_experts:
            raise MoEConfigError("top_k cannot exceed num_experts")
        if (
            self.shared_expert_intermediate_size is not None
            and self.shared_expert_intermediate_size <= 0
        ):
            raise MoEConfigError(
                "shared_expert_intermediate_size must be positive when enabled"
            )
        if self.capacity_factor is not None and self.capacity_factor <= 0:
            raise MoEConfigError("capacity_factor must be positive when configured")
        if self.overflow_policy not in {"unbounded", "error"}:
            raise MoEConfigError(
                "overflow_policy must be 'unbounded' or 'error'"
            )


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    router_probabilities: torch.Tensor
    expert_counts: torch.Tensor
    load_balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    overflow_count: int


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("expert dimensions must be positive")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class RoutedSwiGLUMoE(nn.Module):
    """Reference routed SwiGLU MoE with a shared expert and no token dropping.

    Routing uses full softmax probabilities, selects top-k experts per token,
    and renormalizes the selected probabilities. Capacity can be observed or
    enforced with an error, but tokens are never silently discarded.
    """

    def __init__(self, config: RoutedMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=config.router_bias,
        )
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    config.hidden_size,
                    config.expert_intermediate_size,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.shared_expert = (
            SwiGLUExpert(
                config.hidden_size,
                config.shared_expert_intermediate_size,
            )
            if config.shared_expert_intermediate_size is not None
            else None
        )

    def _capacity(
        self,
        token_count: int,
    ) -> int | None:
        factor = self.config.capacity_factor
        if factor is None:
            return None
        return max(
            1,
            ceil(
                factor
                * token_count
                * self.config.top_k
                / self.config.num_experts
            ),
        )

    def forward(self, x: torch.Tensor) -> MoEOutput:
        if x.ndim < 2 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "MoE input must end in hidden_size="
                f"{self.config.hidden_size}"
            )
        original_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        token_count = int(flat.shape[0])
        if token_count == 0:
            raise ValueError("MoE input must contain at least one token")

        router_logits = self.router(flat)
        router_probabilities = torch.softmax(
            router_logits.float(),
            dim=-1,
        )
        selected_probabilities, selected_experts = torch.topk(
            router_probabilities,
            k=self.config.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        selected_probabilities = selected_probabilities / selected_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(selected_probabilities.dtype).tiny)

        expert_counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.config.num_experts,
        )
        capacity = self._capacity(token_count)
        overflow_count = 0
        if capacity is not None:
            overflow = torch.clamp(expert_counts - capacity, min=0)
            overflow_count = int(overflow.sum().item())
            if (
                overflow_count > 0
                and self.config.overflow_policy == "error"
            ):
                raise RuntimeError(
                    "MoE routing exceeded configured capacity without token "
                    f"dropping: capacity={capacity}, counts={expert_counts.tolist()}"
                )

        routed = torch.zeros(
            flat.shape,
            dtype=flat.dtype,
            device=flat.device,
        )
        for expert_index, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(
                selected_experts == expert_index
            )
            if token_indices.numel() == 0:
                continue
            expert_input = flat.index_select(0, token_indices)
            expert_output = expert(expert_input)
            weights = selected_probabilities[
                token_indices,
                topk_slots,
            ].to(expert_output.dtype)
            routed.index_add_(
                0,
                token_indices,
                expert_output * weights.unsqueeze(-1),
            )

        if self.shared_expert is not None:
            routed = routed + self.shared_expert(flat)

        assignment_fraction = expert_counts.float() / float(
            token_count * self.config.top_k
        )
        mean_router_probability = router_probabilities.mean(dim=0)
        load_balance_loss = self.config.num_experts * torch.sum(
            assignment_fraction * mean_router_probability
        )
        router_z_loss = torch.mean(
            torch.logsumexp(router_logits.float(), dim=-1).square()
        )

        return MoEOutput(
            hidden_states=routed.reshape(original_shape),
            router_logits=router_logits.reshape(
                *original_shape[:-1],
                self.config.num_experts,
            ),
            router_probabilities=router_probabilities.reshape(
                *original_shape[:-1],
                self.config.num_experts,
            ),
            expert_counts=expert_counts,
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
            overflow_count=overflow_count,
        )


class RoutedSwiGLUMoELayer(nn.Module):
    """Pre-norm residual expert-compute layer for HybridLayerType.MOE."""

    def __init__(
        self,
        config: RoutedMoEConfig,
        *,
        norm_eps: float = 1e-5,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if not (0.0 <= residual_dropout < 1.0):
            raise ValueError("residual_dropout must be in [0, 1)")
        self.norm = RMSNorm(config.hidden_size, norm_eps)
        self.moe = RoutedSwiGLUMoE(config)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        routed = self.moe(self.norm(x))
        routed.hidden_states = x + self.residual_dropout(routed.hidden_states)
        return routed
