from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..energy import ReasoningEnergyCritic
from ..norm import RMSNorm


@dataclass(frozen=True)
class ReasoningRecurrenceConfig:
    hidden_size: int
    state_dim: int = 512
    transition_hidden_dim: int = 1024
    max_steps: int = 4
    min_steps: int = 1
    halt_threshold: float = 0.9
    state_delta_epsilon: float = 1e-3
    energy_delta_epsilon: float = 1e-3
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    depth_base: float = 10000.0

    def __post_init__(self) -> None:
        integer_fields = {
            "hidden_size": self.hidden_size,
            "state_dim": self.state_dim,
            "transition_hidden_dim": self.transition_hidden_dim,
            "max_steps": self.max_steps,
            "min_steps": self.min_steps,
        }
        bad = [name for name, value in integer_fields.items() if int(value) <= 0]
        if bad:
            raise ValueError(
                f"positive integer fields required: {', '.join(bad)}"
            )
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if not 0.0 < self.halt_threshold < 1.0:
            raise ValueError("halt_threshold must be in (0, 1)")
        if self.state_delta_epsilon <= 0:
            raise ValueError("state_delta_epsilon must be positive")
        if self.energy_delta_epsilon <= 0:
            raise ValueError("energy_delta_epsilon must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.depth_base <= 1.0:
            raise ValueError("depth_base must be greater than 1")


@dataclass
class ReasoningRecurrenceOutput:
    state: torch.Tensor
    halt_probabilities: torch.Tensor
    halt_weights: torch.Tensor
    relative_state_deltas: torch.Tensor
    energy_trace: torch.Tensor | None
    energy_deltas: torch.Tensor | None
    expected_steps: torch.Tensor
    steps_executed: int


class SpectralDepthEncoding(nn.Module):
    """Deterministic Fourier encoding for the reasoning-depth coordinate."""

    def __init__(self, dim: int, *, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if base <= 1.0:
            raise ValueError("base must be greater than 1")
        half = max(1, dim // 2)
        frequency_index = torch.arange(half, dtype=torch.float32)
        denominator = max(1, half - 1)
        frequencies = torch.exp(
            -torch.log(torch.tensor(float(base)))
            * frequency_index
            / float(denominator)
        )
        self.dim = int(dim)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(
        self,
        depth: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if depth.ndim != 1:
            raise ValueError("depth must have shape [batch]")
        phase = depth.to(device=device, dtype=torch.float32).unsqueeze(-1)
        phase = phase * self.frequencies.to(device=device)
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if encoded.shape[-1] < self.dim:
            encoded = torch.cat(
                (
                    encoded,
                    torch.zeros(
                        (*encoded.shape[:-1], self.dim - encoded.shape[-1]),
                        device=device,
                        dtype=encoded.dtype,
                    ),
                ),
                dim=-1,
            )
        return encoded[..., : self.dim].to(dtype=dtype)


class ReasoningStateTransition(nn.Module):
    """Gated residual transition for reasoning-time latent recurrence."""

    def __init__(self, config: ReasoningRecurrenceConfig) -> None:
        super().__init__()
        self.config = config
        self.state_norm = RMSNorm(config.state_dim, config.rms_norm_eps)
        self.context_norm = RMSNorm(config.state_dim, config.rms_norm_eps)
        self.depth_encoding = SpectralDepthEncoding(
            config.state_dim,
            base=config.depth_base,
        )
        joint_dim = config.state_dim * 3
        self.update = nn.Sequential(
            nn.Linear(joint_dim, config.transition_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transition_hidden_dim, config.state_dim),
        )
        self.gate = nn.Linear(joint_dim, config.state_dim)

    def forward(
        self,
        state: torch.Tensor,
        context_state: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        if state.shape != context_state.shape:
            raise ValueError("state and context_state must have identical shape")
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                "reasoning state must have shape [batch, state_dim]"
            )
        if step <= 0:
            raise ValueError("step must be positive")

        depth_value = float(step) / float(self.config.max_steps)
        depth = torch.full(
            (state.shape[0],),
            depth_value,
            device=state.device,
            dtype=torch.float32,
        )
        depth_features = self.depth_encoding(
            depth,
            dtype=state.dtype,
            device=state.device,
        )
        joint = torch.cat(
            (
                self.state_norm(state),
                self.context_norm(context_state),
                depth_features,
            ),
            dim=-1,
        )
        update = torch.tanh(self.update(joint))
        gate = torch.sigmoid(self.gate(joint))
        return state + gate * update


class AdaptiveHaltingHead(nn.Module):
    """Independent learned halting head for reasoning-time recurrence."""

    def __init__(self, config: ReasoningRecurrenceConfig) -> None:
        super().__init__()
        self.state_norm = RMSNorm(config.state_dim, config.rms_norm_eps)
        self.proj = nn.Linear(config.state_dim + 3, 1)

    def forward(
        self,
        state: torch.Tensor,
        relative_state_delta: torch.Tensor,
        *,
        energy: torch.Tensor | None,
        energy_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("state must have shape [batch, state_dim]")
        if relative_state_delta.shape != (state.shape[0],):
            raise ValueError(
                "relative_state_delta must have shape [batch]"
            )

        zeros = torch.zeros_like(relative_state_delta)
        energy_feature = (
            torch.tanh(energy.float()).to(state.dtype)
            if energy is not None
            else zeros.to(state.dtype)
        )
        energy_delta_feature = (
            torch.tanh(energy_delta.float()).to(state.dtype)
            if energy_delta is not None
            else zeros.to(state.dtype)
        )
        scalar_features = torch.stack(
            (
                torch.log1p(relative_state_delta.float()).to(state.dtype),
                energy_feature,
                energy_delta_feature,
            ),
            dim=-1,
        )
        features = torch.cat(
            (self.state_norm(state), scalar_features),
            dim=-1,
        )
        return torch.sigmoid(self.proj(features).squeeze(-1))


class ReasoningStateInjector(nn.Module):
    """Inject the final reasoning state back into token states with a near-zero gate."""

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        *,
        gate_init: float = -6.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or state_dim <= 0:
            raise ValueError("hidden_size and state_dim must be positive")
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.proj = nn.Linear(state_dim, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_state: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden]"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states final dimension does not match hidden_size")
        if reasoning_state.shape != (
            hidden_states.shape[0],
            self.state_dim,
        ):
            raise ValueError(
                "reasoning_state must have shape [batch, state_dim]"
            )
        injection = self.proj(reasoning_state).unsqueeze(1)
        return hidden_states + torch.sigmoid(self.gate_logit) * injection


class ReasoningRecurrence(nn.Module):
    """Reference adaptive reasoning loop separated from token-time Mamba recurrence.

    The recurrence owns state transitions. An optional EBM can score each
    generated state, but its energy is observational: it can affect the halting
    head and telemetry without directly changing the transition.
    """

    def __init__(self, config: ReasoningRecurrenceConfig) -> None:
        super().__init__()
        self.config = config
        self.context_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.context_to_state = nn.Linear(
            config.hidden_size,
            config.state_dim,
            bias=False,
        )
        self.initial_state = nn.Linear(
            config.hidden_size,
            config.state_dim,
            bias=False,
        )
        self.transition = ReasoningStateTransition(config)
        self.halting = AdaptiveHaltingHead(config)

    def _pool_context(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden]"
            )
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "hidden_states final dimension does not match configured hidden_size"
            )
        batch, sequence, _ = hidden_states.shape
        if attention_mask is None:
            return self.context_norm(hidden_states).mean(dim=1)
        if attention_mask.shape != (batch, sequence):
            raise ValueError(
                f"attention_mask must have shape {(batch, sequence)}"
            )
        valid = attention_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        )
        counts = valid.sum(dim=-1)
        if bool((counts == 0).any()):
            raise ValueError(
                "each reasoning example must contain at least one valid token"
            )
        normalized = self.context_norm(hidden_states)
        masked = normalized * valid.unsqueeze(-1).to(normalized.dtype)
        return masked.sum(dim=1) / counts.unsqueeze(-1).to(normalized.dtype)

    def _validate_document_isolation(
        self,
        document_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        *,
        batch: int,
        sequence: int,
    ) -> None:
        if document_ids is None:
            return
        if document_ids.shape != (batch, sequence):
            raise ValueError(
                f"document_ids must have shape {(batch, sequence)}"
            )
        if document_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("document_ids must be integer typed")
        valid = (
            torch.ones(
                (batch, sequence),
                device=document_ids.device,
                dtype=torch.bool,
            )
            if attention_mask is None
            else attention_mask.to(
                device=document_ids.device,
                dtype=torch.bool,
            )
        )
        for row in range(batch):
            ids = document_ids[row][valid[row]]
            if ids.numel() == 0:
                continue
            if torch.unique(ids).numel() > 1:
                raise ValueError(
                    "reasoning recurrence requires one document per batch row; "
                    "packed multi-document rows must be unpacked before reasoning"
                )

    @staticmethod
    def _relative_delta(
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(
            (current - previous).float(),
            dim=-1,
        )
        denominator = torch.linalg.vector_norm(
            previous.float(),
            dim=-1,
        ).clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        energy_critic: ReasoningEnergyCritic | None = None,
        require_energy_stability: bool = False,
    ) -> ReasoningRecurrenceOutput:
        batch, sequence, _ = hidden_states.shape
        self._validate_document_isolation(
            document_ids,
            attention_mask,
            batch=batch,
            sequence=sequence,
        )
        context = self._pool_context(hidden_states, attention_mask)
        context_state = self.context_to_state(context)
        state = self.initial_state(context)

        if energy_critic is not None:
            if energy_critic.config.state_dim != self.config.state_dim:
                raise ValueError(
                    "energy critic state_dim must match reasoning state_dim"
                )
            if energy_critic.config.context_dim != self.config.hidden_size:
                raise ValueError(
                    "energy critic context_dim must match reasoning hidden_size"
                )
        elif require_energy_stability:
            raise ValueError(
                "require_energy_stability needs an energy critic"
            )

        halt_probabilities: list[torch.Tensor] = []
        relative_deltas: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        energy_deltas: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        previous_energy: torch.Tensor | None = None

        steps_executed = self.config.max_steps
        for step in range(1, self.config.max_steps + 1):
            previous_state = state
            state = self.transition(
                state,
                context_state,
                step=step,
            )
            delta = self._relative_delta(state, previous_state)

            energy = (
                energy_critic(state, context)
                if energy_critic is not None
                else None
            )
            if energy is not None:
                energy_delta = (
                    torch.zeros_like(energy)
                    if previous_energy is None
                    else (energy - previous_energy).abs()
                )
                previous_energy = energy
                energies.append(energy)
                energy_deltas.append(energy_delta)
            else:
                energy_delta = None

            halt_probability = self.halting(
                state,
                delta,
                energy=energy,
                energy_delta=energy_delta,
            )
            if step < self.config.min_steps:
                halt_probability = torch.zeros_like(halt_probability)

            states.append(state)
            relative_deltas.append(delta)
            halt_probabilities.append(halt_probability)

            if not self.training and step >= self.config.min_steps:
                converged = (
                    halt_probability >= self.config.halt_threshold
                ) & (
                    delta <= self.config.state_delta_epsilon
                )
                if require_energy_stability:
                    assert energy_delta is not None
                    converged = converged & (
                        energy_delta <= self.config.energy_delta_epsilon
                    )
                if bool(converged.all()):
                    steps_executed = step
                    break

        state_stack = torch.stack(states, dim=1)
        probability_stack = torch.stack(halt_probabilities, dim=1)
        delta_stack = torch.stack(relative_deltas, dim=1)

        if self.training:
            remaining = torch.ones(
                batch,
                dtype=state.dtype,
                device=state.device,
            )
            weights: list[torch.Tensor] = []
            for index in range(state_stack.shape[1]):
                if index == state_stack.shape[1] - 1:
                    weight = remaining
                else:
                    probability = probability_stack[:, index].to(state.dtype)
                    weight = remaining * probability
                    remaining = remaining - weight
                weights.append(weight)
            weight_stack = torch.stack(weights, dim=1)
            final_state = (
                state_stack * weight_stack.unsqueeze(-1)
            ).sum(dim=1)
        else:
            weight_stack = torch.zeros(
                (
                    batch,
                    state_stack.shape[1],
                ),
                dtype=state.dtype,
                device=state.device,
            )
            weight_stack[:, -1] = 1.0
            final_state = state_stack[:, -1]

        step_numbers = torch.arange(
            1,
            weight_stack.shape[1] + 1,
            device=weight_stack.device,
            dtype=weight_stack.dtype,
        )
        expected_steps = (
            weight_stack * step_numbers.unsqueeze(0)
        ).sum(dim=-1).mean()

        energy_trace = (
            torch.stack(energies, dim=1)
            if energies
            else None
        )
        energy_delta_trace = (
            torch.stack(energy_deltas, dim=1)
            if energy_deltas
            else None
        )

        return ReasoningRecurrenceOutput(
            state=final_state,
            halt_probabilities=probability_stack,
            halt_weights=weight_stack,
            relative_state_deltas=delta_stack,
            energy_trace=energy_trace,
            energy_deltas=energy_delta_trace,
            expected_steps=expected_steps,
            steps_executed=steps_executed,
        )
