from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..norm import RMSNorm
from .situ import SiTUAndMul


class StableLatentMoEError(RuntimeError):
    pass


@dataclass(frozen=True)
class StableLatentMoEConfig:
    hidden_size: int
    latent_size: int
    expert_intermediate_size: int
    num_experts: int
    top_k: int
    num_shared_experts: int = 2
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        ints = {
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_shared_experts": self.num_shared_experts,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise ValueError(
                f"positive Stable LatentMoE fields required: {', '.join(bad)}"
            )
        if self.top_k >= self.num_experts:
            raise ValueError("top_k must be smaller than num_experts")
        if self.situ_beta <= 0 or self.situ_linear_beta <= 0:
            raise ValueError("SiTU beta values must be positive")
        if self.routed_scaling_factor <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("routing scale and norm epsilon must be positive")


@dataclass
class StableLatentMoEOutput:
    hidden_states: torch.Tensor
    raw_router_scores: torch.Tensor
    selected_experts: torch.Tensor
    selected_weights: torch.Tensor
    expert_counts: torch.Tensor
    routing_bias: torch.Tensor


class SiTUExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        beta: float,
        linear_beta: float,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = SiTUAndMul(
            beta=beta,
            linear_beta=linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.activation(
                self.gate_proj(x),
                self.up_proj(x),
            )
        )


class StableLatentMoE(nn.Module):
    """Trainable Kimi-K3-style Stable LatentMoE reference.

    Routing is performed in full model width using sigmoid affinities.
    Routed experts operate only in latent width. Their weighted aggregate is
    RMS-normalized and projected back to model width. Shared experts operate at
    full width and are always active.

    Quantile-Balancing bias is a persistent non-gradient buffer. A forward pass
    uses the currently committed bias; compute_next_routing_bias() derives the
    next bias from that batch and commit_routing_bias() makes it active only for
    future forwards.
    """

    def __init__(self, config: StableLatentMoEConfig) -> None:
        super().__init__()
        self.config = config

        self.router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )
        self.latent_down = nn.Linear(
            config.hidden_size,
            config.latent_size,
            bias=False,
        )
        self.latent_up = nn.Linear(
            config.latent_size,
            config.hidden_size,
            bias=False,
        )
        self.latent_norm = RMSNorm(
            config.latent_size,
            config.rms_norm_eps,
        )

        self.routed_experts = nn.ModuleList(
            SiTUExpert(
                config.latent_size,
                config.expert_intermediate_size,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )
            for _ in range(config.num_experts)
        )
        self.shared_experts = nn.ModuleList(
            SiTUExpert(
                config.hidden_size,
                config.expert_intermediate_size,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )
            for _ in range(config.num_shared_experts)
        )
        self.register_buffer(
            "routing_bias",
            torch.zeros(config.num_experts),
            persistent=True,
        )

    def _route(
        self,
        flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.sigmoid(self.router(flat).float())
        choice_scores = raw + self.routing_bias.float().unsqueeze(0)
        _, selected = torch.topk(
            choice_scores,
            k=self.config.top_k,
            dim=-1,
            largest=True,
            sorted=False,
        )
        weights = raw.gather(1, selected)
        weights = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(weights.dtype).tiny)
        weights = weights * self.config.routed_scaling_factor
        return raw, selected, weights

    def forward(self, x: torch.Tensor) -> StableLatentMoEOutput:
        if x.ndim < 2 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Stable LatentMoE input must end in hidden_size="
                f"{self.config.hidden_size}"
            )
        original_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        if flat.shape[0] == 0:
            raise ValueError("Stable LatentMoE requires at least one token")

        raw_scores, selected, weights = self._route(flat)
        latent = self.latent_down(flat)

        routed = torch.zeros_like(latent)
        for expert_index, expert in enumerate(self.routed_experts):
            token_rows, slots = torch.where(selected == expert_index)
            if token_rows.numel() == 0:
                continue
            expert_input = latent.index_select(0, token_rows)
            expert_output = expert(expert_input)
            token_weights = weights[
                token_rows,
                slots,
            ].to(expert_output.dtype)
            routed.index_add_(
                0,
                token_rows,
                expert_output * token_weights.unsqueeze(-1),
            )

        routed = self.latent_up(self.latent_norm(routed))

        shared = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared = shared + expert(flat)

        hidden = (routed + shared).reshape(original_shape)
        if not bool(torch.isfinite(hidden).all()):
            raise StableLatentMoEError(
                "Stable LatentMoE produced non-finite hidden states"
            )
        counts = torch.bincount(
            selected.reshape(-1),
            minlength=self.config.num_experts,
        )
        return StableLatentMoEOutput(
            hidden_states=hidden,
            raw_router_scores=raw_scores.reshape(
                *original_shape[:-1],
                self.config.num_experts,
            ),
            selected_experts=selected.reshape(
                *original_shape[:-1],
                self.config.top_k,
            ),
            selected_weights=weights.reshape(
                *original_shape[:-1],
                self.config.top_k,
            ),
            expert_counts=counts,
            routing_bias=self.routing_bias.detach().clone(),
        )

    @torch.no_grad()
    def compute_next_routing_bias(
        self,
        raw_router_scores: torch.Tensor,
    ) -> torch.Tensor:
        if raw_router_scores.shape[-1] != self.config.num_experts:
            raise ValueError(
                "raw_router_scores final dimension must equal num_experts"
            )
        scores = raw_router_scores.reshape(
            -1,
            self.config.num_experts,
        ).float()
        if scores.shape[0] == 0:
            raise ValueError("router-score batch is empty")

        biased = scores + self.routing_bias.float().unsqueeze(0)
        top_values = torch.topk(
            biased,
            k=self.config.top_k + 1,
            dim=-1,
            largest=True,
            sorted=True,
        ).values
        cutoff = top_values[:, -1]
        required_bias = cutoff.unsqueeze(-1) - scores
        q = self.config.top_k / self.config.num_experts
        next_bias = torch.quantile(
            required_bias,
            q=q,
            dim=0,
        )
        return next_bias - next_bias.mean()

    @torch.no_grad()
    def commit_routing_bias(
        self,
        next_bias: torch.Tensor,
    ) -> None:
        if next_bias.shape != self.routing_bias.shape:
            raise ValueError(
                f"next routing bias must have shape {tuple(self.routing_bias.shape)}"
            )
        if not bool(torch.isfinite(next_bias).all()):
            raise ValueError("next routing bias contains non-finite values")
        self.routing_bias.copy_(
            next_bias.to(
                device=self.routing_bias.device,
                dtype=self.routing_bias.dtype,
            )
        )


class StableLatentMoELayer(nn.Module):
    def __init__(
        self,
        config: StableLatentMoEConfig,
        *,
        norm_eps: float = 1e-5,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= residual_dropout < 1.0):
            raise ValueError("residual_dropout must be in [0, 1)")
        self.norm = RMSNorm(config.hidden_size, norm_eps)
        self.moe = StableLatentMoE(config)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> StableLatentMoEOutput:
        output = self.moe(self.norm(x))
        output.hidden_states = (
            x + self.residual_dropout(output.hidden_states)
        )
        return output
