from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Sequence
import json

import torch

from iq_model import ReasoningEnergyCritic, energy_margin_ranking_loss

from .optimizer import IQOptimizer


class ReasoningCriticTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class ReasoningTrajectoryBatch:
    """Detached generated reasoning trajectories with verifier outcomes.

    `verified_success` must come from an external task/verifier outcome. This
    container never infers success from model confidence, energy, LM loss, or
    halting behavior.
    """

    task_ids: tuple[str, ...]
    context: torch.Tensor
    state_trace: torch.Tensor
    verified_success: torch.Tensor
    step_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.context.ndim != 2:
            raise ReasoningCriticTrainingError(
                "context must have shape [trajectories, context_dim]"
            )
        if self.state_trace.ndim != 3:
            raise ReasoningCriticTrainingError(
                "state_trace must have shape [trajectories, steps, state_dim]"
            )
        trajectories = self.context.shape[0]
        if self.state_trace.shape[0] != trajectories:
            raise ReasoningCriticTrainingError(
                "context and state_trace trajectory counts must match"
            )
        if len(self.task_ids) != trajectories:
            raise ReasoningCriticTrainingError(
                "task_ids length must match trajectory count"
            )
        if any(not task_id for task_id in self.task_ids):
            raise ReasoningCriticTrainingError(
                "task_ids must be non-empty"
            )
        if self.verified_success.shape != (trajectories,):
            raise ReasoningCriticTrainingError(
                "verified_success must have shape [trajectories]"
            )
        if self.verified_success.dtype is not torch.bool:
            raise ReasoningCriticTrainingError(
                "verified_success must be boolean"
            )
        if (
            self.state_trace.device != self.context.device
            or self.verified_success.device != self.context.device
        ):
            raise ReasoningCriticTrainingError(
                "context, state_trace, and verified_success must share a device"
            )
        if self.state_trace.shape[1] <= 0:
            raise ReasoningCriticTrainingError(
                "state_trace must contain at least one reasoning step"
            )
        if self.step_mask is not None:
            if self.step_mask.shape != self.state_trace.shape[:2]:
                raise ReasoningCriticTrainingError(
                    "step_mask must have shape [trajectories, steps]"
                )
            if self.step_mask.dtype is not torch.bool:
                raise ReasoningCriticTrainingError(
                    "step_mask must be boolean"
                )
            if self.step_mask.device != self.context.device:
                raise ReasoningCriticTrainingError(
                    "step_mask must share the trajectory device"
                )
            if bool((~self.step_mask.any(dim=-1)).any()):
                raise ReasoningCriticTrainingError(
                    "every trajectory must contain at least one valid reasoning step"
                )
        if not bool(torch.isfinite(self.context).all()):
            raise ReasoningCriticTrainingError(
                "context contains non-finite values"
            )
        if not bool(torch.isfinite(self.state_trace).all()):
            raise ReasoningCriticTrainingError(
                "state_trace contains non-finite values"
            )

    @property
    def effective_step_mask(self) -> torch.Tensor:
        if self.step_mask is not None:
            return self.step_mask
        return torch.ones(
            self.state_trace.shape[:2],
            dtype=torch.bool,
            device=self.state_trace.device,
        )

    @classmethod
    def from_model_output(
        cls,
        output: object,
        *,
        task_ids: Sequence[str],
        verified_success: torch.Tensor,
    ) -> "ReasoningTrajectoryBatch":
        context = getattr(output, "reasoning_context", None)
        state_trace = getattr(output, "reasoning_state_trace", None)
        if not isinstance(verified_success, torch.Tensor):
            raise ReasoningCriticTrainingError(
                "verified_success must be a boolean tensor"
            )
        if verified_success.dtype is not torch.bool:
            raise ReasoningCriticTrainingError(
                "verified_success must be boolean; verifier scores must be thresholded explicitly upstream"
            )
        if not isinstance(context, torch.Tensor):
            raise ReasoningCriticTrainingError(
                "model output is missing reasoning_context"
            )
        if not isinstance(state_trace, torch.Tensor):
            raise ReasoningCriticTrainingError(
                "model output is missing reasoning_state_trace"
            )
        return cls(
            task_ids=tuple(str(task_id) for task_id in task_ids),
            context=context.detach().clone(),
            state_trace=state_trace.detach().clone(),
            verified_success=verified_success.detach().to(
                device=context.device,
            ).clone(),
            step_mask=torch.ones(
                state_trace.shape[:2],
                dtype=torch.bool,
                device=context.device,
            ),
        )

    @classmethod
    def concatenate(
        cls,
        batches: Sequence["ReasoningTrajectoryBatch"],
    ) -> "ReasoningTrajectoryBatch":
        if not batches:
            raise ReasoningCriticTrainingError(
                "at least one trajectory batch is required"
            )
        context_dim = batches[0].context.shape[-1]
        state_dim = batches[0].state_trace.shape[-1]
        device = batches[0].context.device
        context_dtype = batches[0].context.dtype
        state_dtype = batches[0].state_trace.dtype
        max_steps = max(batch.state_trace.shape[1] for batch in batches)
        padded_states: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        for batch in batches:
            if batch.context.shape[-1] != context_dim:
                raise ReasoningCriticTrainingError(
                    "all trajectory batches must share context_dim"
                )
            if batch.state_trace.shape[-1] != state_dim:
                raise ReasoningCriticTrainingError(
                    "all trajectory batches must share state_dim"
                )
            if batch.context.device != device or batch.state_trace.device != device:
                raise ReasoningCriticTrainingError(
                    "all trajectory batches must be on the same device"
                )
            if batch.context.dtype != context_dtype or batch.state_trace.dtype != state_dtype:
                raise ReasoningCriticTrainingError(
                    "all trajectory batches must share context/state dtypes"
                )
            steps = batch.state_trace.shape[1]
            if steps < max_steps:
                state_padding = torch.zeros(
                    (
                        batch.state_trace.shape[0],
                        max_steps - steps,
                        state_dim,
                    ),
                    device=device,
                    dtype=state_dtype,
                )
                padded_states.append(
                    torch.cat((batch.state_trace, state_padding), dim=1)
                )
                mask_padding = torch.zeros(
                    (
                        batch.state_trace.shape[0],
                        max_steps - steps,
                    ),
                    device=device,
                    dtype=torch.bool,
                )
                padded_masks.append(
                    torch.cat(
                        (batch.effective_step_mask, mask_padding),
                        dim=1,
                    )
                )
            else:
                padded_states.append(batch.state_trace)
                padded_masks.append(batch.effective_step_mask)
        return cls(
            task_ids=tuple(
                task_id
                for batch in batches
                for task_id in batch.task_ids
            ),
            context=torch.cat(
                [batch.context for batch in batches],
                dim=0,
            ),
            state_trace=torch.cat(padded_states, dim=0),
            verified_success=torch.cat(
                [batch.verified_success for batch in batches],
                dim=0,
            ),
            step_mask=torch.cat(padded_masks, dim=0),
        )


@dataclass(frozen=True)
class ReasoningEnergyPairBatch:
    """Same-task successful/failed state pairs for EBM ranking."""

    context: torch.Tensor
    positive_states: torch.Tensor
    negative_states: torch.Tensor
    task_ids: tuple[str, ...]
    positive_step_indices: torch.Tensor
    negative_step_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.context.ndim != 2:
            raise ReasoningCriticTrainingError(
                "pair context must have shape [pairs, context_dim]"
            )
        if self.positive_states.ndim != 2 or self.negative_states.ndim != 2:
            raise ReasoningCriticTrainingError(
                "paired states must have shape [pairs, state_dim]"
            )
        if self.positive_states.shape != self.negative_states.shape:
            raise ReasoningCriticTrainingError(
                "positive and negative state shapes must match"
            )
        pairs = self.positive_states.shape[0]
        if self.context.shape[0] != pairs:
            raise ReasoningCriticTrainingError(
                "context pair count must match state pair count"
            )
        if len(self.task_ids) != pairs:
            raise ReasoningCriticTrainingError(
                "task_ids length must match pair count"
            )
        if (
            self.positive_step_indices.shape != (pairs,)
            or self.negative_step_indices.shape != (pairs,)
        ):
            raise ReasoningCriticTrainingError(
                "positive/negative step indices must have shape [pairs]"
            )
        if (
            self.positive_step_indices.dtype not in (torch.int32, torch.int64)
            or self.negative_step_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ReasoningCriticTrainingError(
                "positive/negative step indices must be integer typed"
            )
        if (
            self.positive_states.device != self.context.device
            or self.negative_states.device != self.context.device
            or self.positive_step_indices.device != self.context.device
            or self.negative_step_indices.device != self.context.device
        ):
            raise ReasoningCriticTrainingError(
                "all energy-pair tensors must share a device"
            )
        if pairs == 0:
            raise ReasoningCriticTrainingError(
                "energy pair batch cannot be empty"
            )


@dataclass(frozen=True)
class ReasoningCriticPairingConfig:
    max_trajectory_pairs_per_task: int = 32
    include_all_steps: bool = True
    context_atol: float = 1e-5
    context_rtol: float = 1e-5

    def __post_init__(self) -> None:
        if self.max_trajectory_pairs_per_task <= 0:
            raise ReasoningCriticTrainingError(
                "max_trajectory_pairs_per_task must be positive"
            )
        if self.context_atol < 0 or self.context_rtol < 0:
            raise ReasoningCriticTrainingError(
                "context tolerances must be non-negative"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_same_task_energy_pairs(
    trajectories: ReasoningTrajectoryBatch,
    config: ReasoningCriticPairingConfig = ReasoningCriticPairingConfig(),
) -> ReasoningEnergyPairBatch:
    """Pair verified success/failure trajectories from the same task/context.

    Pairing is deterministic in input order. Context equality is checked before
    pairing so a reused task id cannot accidentally compare unrelated prompts.
    """

    groups: dict[str, list[int]] = {}
    for index, task_id in enumerate(trajectories.task_ids):
        groups.setdefault(task_id, []).append(index)

    contexts: list[torch.Tensor] = []
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    pair_task_ids: list[str] = []
    positive_step_indices: list[int] = []
    negative_step_indices: list[int] = []

    for task_id, indices in groups.items():
        reference = trajectories.context[indices[0]]
        for index in indices[1:]:
            if not torch.allclose(
                reference.float(),
                trajectories.context[index].float(),
                atol=config.context_atol,
                rtol=config.context_rtol,
            ):
                raise ReasoningCriticTrainingError(
                    f"task {task_id!r} contains mismatched reasoning contexts"
                )

        success_indices = [
            index
            for index in indices
            if bool(trajectories.verified_success[index])
        ]
        failure_indices = [
            index
            for index in indices
            if not bool(trajectories.verified_success[index])
        ]
        if not success_indices or not failure_indices:
            continue

        trajectory_pairs = 0
        for positive_index in success_indices:
            for negative_index in failure_indices:
                if trajectory_pairs >= config.max_trajectory_pairs_per_task:
                    break
                positive_valid = torch.nonzero(
                    trajectories.effective_step_mask[positive_index],
                    as_tuple=False,
                ).flatten()
                negative_valid = torch.nonzero(
                    trajectories.effective_step_mask[negative_index],
                    as_tuple=False,
                ).flatten()
                if config.include_all_steps:
                    negative_set = {
                        int(step)
                        for step in negative_valid.tolist()
                    }
                    step_pairs = [
                        (int(step), int(step))
                        for step in positive_valid.tolist()
                        if int(step) in negative_set
                    ]
                else:
                    step_pairs = [
                        (
                            int(positive_valid[-1]),
                            int(negative_valid[-1]),
                        )
                    ]
                if not step_pairs:
                    raise ReasoningCriticTrainingError(
                        f"task {task_id!r} has no comparable reasoning steps"
                    )
                for positive_step, negative_step in step_pairs:
                    contexts.append(reference)
                    positives.append(
                        trajectories.state_trace[
                            positive_index,
                            positive_step,
                        ]
                    )
                    negatives.append(
                        trajectories.state_trace[
                            negative_index,
                            negative_step,
                        ]
                    )
                    pair_task_ids.append(task_id)
                    positive_step_indices.append(positive_step)
                    negative_step_indices.append(negative_step)
                trajectory_pairs += 1
            if trajectory_pairs >= config.max_trajectory_pairs_per_task:
                break

    if not positives:
        raise ReasoningCriticTrainingError(
            "no same-task success/failure trajectory pairs were available"
        )

    device = trajectories.context.device
    return ReasoningEnergyPairBatch(
        context=torch.stack(contexts, dim=0).detach(),
        positive_states=torch.stack(positives, dim=0).detach(),
        negative_states=torch.stack(negatives, dim=0).detach(),
        task_ids=tuple(pair_task_ids),
        positive_step_indices=torch.tensor(
            positive_step_indices,
            device=device,
            dtype=torch.long,
        ),
        negative_step_indices=torch.tensor(
            negative_step_indices,
            device=device,
            dtype=torch.long,
        ),
    )


@dataclass(frozen=True)
class ReasoningCriticTrainConfig:
    margin: float = 1.0
    temperature: float = 1.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.margin < 0:
            raise ReasoningCriticTrainingError(
                "margin must be non-negative"
            )
        if self.temperature <= 0:
            raise ReasoningCriticTrainingError(
                "temperature must be positive"
            )
        if self.max_grad_norm <= 0:
            raise ReasoningCriticTrainingError(
                "max_grad_norm must be positive"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningCriticExperimentConfig:
    pairing: ReasoningCriticPairingConfig = field(
        default_factory=ReasoningCriticPairingConfig
    )
    training: ReasoningCriticTrainConfig = field(
        default_factory=ReasoningCriticTrainConfig
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pairing": self.pairing.to_dict(),
            "training": self.training.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReasoningCriticStepMetrics:
    loss: float
    positive_energy: float
    negative_energy: float
    separation_rate: float
    grad_norm: float
    pairs: int


def reasoning_critic_pair_loss(
    critic: ReasoningEnergyCritic,
    pairs: ReasoningEnergyPairBatch,
    config: ReasoningCriticTrainConfig = ReasoningCriticTrainConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_energy = critic(
        pairs.positive_states,
        pairs.context,
    )
    negative_energy = critic(
        pairs.negative_states,
        pairs.context,
    )
    loss = energy_margin_ranking_loss(
        positive_energy,
        negative_energy,
        margin=config.margin,
        temperature=config.temperature,
    )
    return loss, positive_energy, negative_energy


def train_reasoning_critic_step(
    critic: ReasoningEnergyCritic,
    optimizer: IQOptimizer,
    pairs: ReasoningEnergyPairBatch,
    config: ReasoningCriticTrainConfig = ReasoningCriticTrainConfig(),
) -> ReasoningCriticStepMetrics:
    """Update only the standalone critic on detached verified trajectory pairs."""

    if any(
        tensor.requires_grad
        for tensor in (
            pairs.context,
            pairs.positive_states,
            pairs.negative_states,
        )
    ):
        raise ReasoningCriticTrainingError(
            "trajectory pair tensors must be detached before critic training"
        )

    optimizer.zero_grad(set_to_none=True)
    critic.train()
    loss, positive_energy, negative_energy = reasoning_critic_pair_loss(
        critic,
        pairs,
        config,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("reasoning critic loss is non-finite")
    loss.backward()

    bad_gradients = [
        name
        for name, parameter in critic.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
    ]
    if bad_gradients:
        raise RuntimeError(
            "non-finite reasoning critic gradients: "
            + ", ".join(bad_gradients)
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.parameters(),
        config.max_grad_norm,
    )
    if not bool(torch.isfinite(grad_norm)):
        raise RuntimeError(
            "reasoning critic gradient norm is non-finite"
        )
    optimizer.step()

    with torch.no_grad():
        separation = (
            positive_energy < negative_energy
        ).float().mean()

    return ReasoningCriticStepMetrics(
        loss=float(loss.detach()),
        positive_energy=float(positive_energy.detach().mean()),
        negative_energy=float(negative_energy.detach().mean()),
        separation_rate=float(separation),
        grad_norm=float(grad_norm),
        pairs=pairs.positive_states.shape[0],
    )
