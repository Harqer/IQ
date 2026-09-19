from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from .optimizer import IQOptimizer


class TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainStepConfig:
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0:
            raise TrainingError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise TrainingError("max_grad_norm must be positive")
        if self.precision not in {"fp32", "bf16"}:
            raise TrainingError("precision must be 'fp32' or 'bf16'")


@dataclass(frozen=True)
class TrainStepMetrics:
    loss: float
    grad_norm: float
    microbatches: int
    tokens: int


def _device_type(model: torch.nn.Module) -> str:
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise TrainingError("model has no parameters") from exc
    return parameter.device.type


def _validate_batch(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if "input_ids" not in batch:
        raise TrainingError("batch is missing input_ids")
    input_ids = batch["input_ids"]
    labels = batch.get("labels", input_ids)
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TrainingError("input_ids and labels must be tensors")
    if input_ids.shape != labels.shape:
        raise TrainingError("input_ids and labels must have identical shape")
    return input_ids, labels


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            bad.append(name)
    if bad:
        raise TrainingError(f"non-finite gradients: {', '.join(bad)}")


def train_step(
    model: torch.nn.Module,
    optimizer: IQOptimizer,
    microbatches: Iterable[Mapping[str, torch.Tensor]],
    config: TrainStepConfig = TrainStepConfig(),
) -> TrainStepMetrics:
    batches = list(microbatches)
    if len(batches) != config.gradient_accumulation_steps:
        raise TrainingError(
            f"expected {config.gradient_accumulation_steps} microbatches, got {len(batches)}"
        )

    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_sum = 0.0
    token_count = 0
    device_type = _device_type(model)
    use_bf16 = config.precision == "bf16"
    if use_bf16 and device_type not in {"cuda", "cpu"}:
        raise TrainingError(f"bf16 autocast is not supported by this training path on {device_type}")

    for batch in batches:
        input_ids, labels = _validate_batch(batch)
        token_count += int(input_ids.numel())
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
            output = model(input_ids, labels=labels)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise TrainingError("model produced a missing or non-finite loss")
            scaled_loss = output.loss / config.gradient_accumulation_steps
        scaled_loss.backward()
        loss_sum += float(output.loss.detach())

    _assert_finite_gradients(model)
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    if not bool(torch.isfinite(grad_norm_tensor)):
        raise TrainingError("gradient norm is non-finite")
    optimizer.step()

    return TrainStepMetrics(
        loss=loss_sum / len(batches),
        grad_norm=float(grad_norm_tensor),
        microbatches=len(batches),
        tokens=token_count,
    )
