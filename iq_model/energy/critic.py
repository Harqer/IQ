from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..norm import RMSNorm


@dataclass(frozen=True)
class ReasoningEnergyCriticConfig:
    """Configuration for the standalone reasoning-state energy critic.

    The critic scores candidate reasoning states conditioned on context. It does
    not update the reasoning state and is therefore independent of the backbone
    and recurrence transition.
    """

    state_dim: int
    context_dim: int
    hidden_dim: int = 512
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")


class ReasoningEnergyCritic(nn.Module):
    """Scalar EBM critic for candidate reasoning states.

    Lower energy means the candidate is more compatible with the supplied
    context. The module is intentionally a scorer only: it has no transition,
    integration, routing, residual-mixing, or halting authority.
    """

    def __init__(self, config: ReasoningEnergyCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.state_norm = RMSNorm(config.state_dim, config.rms_norm_eps)
        self.context_norm = RMSNorm(config.context_dim, config.rms_norm_eps)
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim, bias=False)
        self.context_proj = nn.Linear(
            config.context_dim,
            config.hidden_dim,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.energy_head = nn.Linear(config.hidden_dim, 1, bias=False)

    def _broadcast_context(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim < 2:
            raise ValueError("state must have at least batch and feature dimensions")
        if context.ndim < 2:
            raise ValueError("context must have at least batch and feature dimensions")
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state final dimension must equal state_dim={self.config.state_dim}"
            )
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(
                "context final dimension must equal "
                f"context_dim={self.config.context_dim}"
            )
        if state.shape[0] != context.shape[0]:
            raise ValueError("state and context batch dimensions must match")

        expanded = context
        while expanded.ndim < state.ndim:
            expanded = expanded.unsqueeze(-2)

        target = (*state.shape[:-1], self.config.context_dim)
        try:
            return torch.broadcast_to(expanded, target)
        except RuntimeError as exc:
            raise ValueError(
                "context prefix dimensions must be broadcastable to state prefix dimensions"
            ) from exc

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        context = self._broadcast_context(state, context)
        state_h = self.state_proj(self.state_norm(state))
        context_h = self.context_proj(self.context_norm(context))
        hidden = self.fuse(state_h + context_h)
        energy = self.energy_head(hidden).squeeze(-1)
        if not bool(torch.isfinite(energy).all()):
            raise RuntimeError("reasoning energy critic produced non-finite scores")
        return energy

    def score_candidates(
        self,
        candidates: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Score [batch, candidates, state_dim] states with lower-is-better energy."""

        if candidates.ndim != 3:
            raise ValueError(
                "candidates must have shape [batch, candidates, state_dim]"
            )
        return self(candidates, context)

    @torch.no_grad()
    def best_candidate_index(
        self,
        candidates: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Return the minimum-energy candidate index for each batch item."""

        return self.score_candidates(candidates, context).argmin(dim=-1)


def energy_margin_ranking_loss(
    positive_energy: torch.Tensor,
    negative_energy: torch.Tensor,
    *,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Prefer successful/correct reasoning states to have lower energy."""

    if positive_energy.shape != negative_energy.shape:
        raise ValueError("positive and negative energy tensors must match in shape")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softplus(
        (positive_energy - negative_energy + margin) / temperature
    ).mean()
