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
            raise TrainingError(
                "gradient_accumulation_steps must be positive"
            )
        if self.max_grad_norm <= 0:
            raise TrainingError("max_grad_norm must be positive")
        if self.precision not in {"fp32", "bf16"}:
            raise TrainingError(
                "precision must be 'fp32' or 'bf16'"
            )


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


def _validate_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if "input_ids" not in batch:
        raise TrainingError("batch is missing input_ids")
    input_ids = batch["input_ids"]
    labels = batch.get("labels", input_ids)
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        labels,
        torch.Tensor,
    ):
        raise TrainingError(
            "input_ids and labels must be tensors"
        )
    if input_ids.shape != labels.shape:
        raise TrainingError(
            "input_ids and labels must have identical shape"
        )

    result = {
        "input_ids": input_ids,
        "labels": labels,
    }
    for name in (
        "attention_mask",
        "position_ids",
        "document_ids",
    ):
        value = batch.get(name)
        if value is None:
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != input_ids.shape
        ):
            raise TrainingError(
                f"{name} must be a tensor with the same shape as input_ids"
            )
        result[name] = value

    reasoning_context_lengths = batch.get("reasoning_context_lengths")
    if reasoning_context_lengths is not None:
        if (
            not isinstance(reasoning_context_lengths, torch.Tensor)
            or reasoning_context_lengths.shape != (input_ids.shape[0],)
            or reasoning_context_lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise TrainingError(
                "reasoning_context_lengths must be an integer tensor with shape [batch]"
            )
        result["reasoning_context_lengths"] = reasoning_context_lengths
    return result


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if (
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ):
            bad.append(name)
    if bad:
        raise TrainingError(
            f"non-finite gradients: {', '.join(bad)}"
        )


def train_step(
    model: torch.nn.Module,
    optimizer: IQOptimizer,
    microbatches: Iterable[Mapping[str, torch.Tensor]],
    config: TrainStepConfig = TrainStepConfig(),
) -> TrainStepMetrics:
    batches = list(microbatches)
    if len(batches) != config.gradient_accumulation_steps:
        raise TrainingError(
            f"expected {config.gradient_accumulation_steps} "
            f"microbatches, got {len(batches)}"
        )

    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_sum = 0.0
    token_count = 0
    device_type = _device_type(model)
    use_bf16 = config.precision == "bf16"
    if (
        use_bf16
        and device_type not in {"cuda", "cpu"}
    ):
        raise TrainingError(
            "bf16 autocast is not supported by this training "
            f"path on {device_type}"
        )

    for batch in batches:
        model_inputs = _validate_batch(batch)
        attention_mask = model_inputs.get("attention_mask")
        token_count += (
            int(attention_mask.to(dtype=torch.bool).sum())
            if attention_mask is not None
            else int(model_inputs["input_ids"].numel())
        )
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = model(**model_inputs)
            if (
                output.loss is None
                or not bool(torch.isfinite(output.loss))
            ):
                raise TrainingError(
                    "model produced a missing or non-finite loss"
                )
            scaled_loss = (
                output.loss
                / config.gradient_accumulation_steps
            )
        scaled_loss.backward()
        loss_sum += float(output.loss.detach())

    _assert_finite_gradients(model)
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.max_grad_norm,
    )
    if not bool(torch.isfinite(grad_norm_tensor)):
        raise TrainingError("gradient norm is non-finite")
    optimizer.step()

    return TrainStepMetrics(
        loss=loss_sum / len(batches),
        grad_norm=float(grad_norm_tensor),
        microbatches=len(batches),
        tokens=token_count,
    )
