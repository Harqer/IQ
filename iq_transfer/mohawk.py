from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F


class MohawkStage(str, Enum):
    MATRIX_ORIENTATION = "matrix_orientation"
    HIDDEN_ALIGNMENT = "hidden_alignment"
    END_TO_END_DISTILLATION = "end_to_end_distillation"


@dataclass(frozen=True)
class MohawkConfig:
    stage: MohawkStage
    temperature: float = 1.0
    matrix_weight: float = 1.0
    hidden_weight: float = 1.0
    logits_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        for name, value in (
            ("matrix_weight", self.matrix_weight),
            ("hidden_weight", self.hidden_weight),
            ("logits_weight", self.logits_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def matrix_orientation_loss(
    teacher_mixer: Tensor,
    student_mixer: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """MOHAWK Stage 1 Frobenius-distance objective.

    Inputs are comparable token-mixing operators with identical shape
    [..., query, key]. The caller is responsible for constructing a faithful
    Mamba-3 MIMO effective mixer rather than substituting hidden activations.
    """
    if teacher_mixer.shape != student_mixer.shape:
        raise ValueError(
            "teacher/student mixer shapes must match: "
            f"{tuple(teacher_mixer.shape)} != {tuple(student_mixer.shape)}"
        )
    delta = (student_mixer.float() - teacher_mixer.float()).square()
    if mask is not None:
        if mask.shape != teacher_mixer.shape[-2:]:
            raise ValueError("matrix mask must match final query/key dimensions")
        delta = delta * mask.to(device=delta.device, dtype=delta.dtype)
        denom = mask.sum().clamp_min(1).to(delta.dtype)
        leading = delta.numel() // (mask.numel())
        return delta.sum() / (denom * leading)
    return delta.mean()


def hidden_alignment_loss(
    teacher_hidden: Tensor,
    student_hidden: Tensor,
    *,
    token_mask: Tensor | None = None,
) -> Tensor:
    """MOHAWK Stage 2 per-token L2 hidden-state alignment."""
    if teacher_hidden.shape != student_hidden.shape:
        raise ValueError(
            "teacher/student hidden shapes must match: "
            f"{tuple(teacher_hidden.shape)} != {tuple(student_hidden.shape)}"
        )
    if teacher_hidden.ndim < 2:
        raise ValueError("hidden states must include token and feature dimensions")
    per_token = torch.linalg.vector_norm(
        student_hidden.float() - teacher_hidden.float(),
        ord=2,
        dim=-1,
    )
    if token_mask is None:
        return per_token.mean()
    if token_mask.shape != per_token.shape:
        raise ValueError("token_mask must match hidden states without feature dimension")
    weights = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1)


def logits_distillation_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    temperature: float = 1.0,
    token_mask: Tensor | None = None,
) -> Tensor:
    """MOHAWK Stage 3 teacher→student KL distillation."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher/student logits shapes must match: "
            f"{tuple(teacher_logits.shape)} != {tuple(student_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t = float(temperature)
    teacher_prob = F.softmax(teacher_logits.float() / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / t, dim=-1)
    per_token = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="none",
    ).sum(dim=-1) * (t * t)
    if token_mask is None:
        return per_token.mean()
    if token_mask.shape != per_token.shape:
        raise ValueError("token_mask must match logits without vocabulary dimension")
    weights = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1)


def mohawk_loss(
    config: MohawkConfig,
    *,
    teacher_mixer: Tensor | None = None,
    student_mixer: Tensor | None = None,
    teacher_hidden: Tensor | None = None,
    student_hidden: Tensor | None = None,
    teacher_logits: Tensor | None = None,
    student_logits: Tensor | None = None,
    token_mask: Tensor | None = None,
    matrix_mask: Tensor | None = None,
) -> Tensor:
    """Fail-closed dispatcher for the three MOHAWK stages."""
    if config.stage is MohawkStage.MATRIX_ORIENTATION:
        if teacher_mixer is None or student_mixer is None:
            raise ValueError("Stage 1 requires teacher_mixer and student_mixer")
        return config.matrix_weight * matrix_orientation_loss(
            teacher_mixer,
            student_mixer,
            mask=matrix_mask,
        )
    if config.stage is MohawkStage.HIDDEN_ALIGNMENT:
        if teacher_hidden is None or student_hidden is None:
            raise ValueError("Stage 2 requires teacher_hidden and student_hidden")
        return config.hidden_weight * hidden_alignment_loss(
            teacher_hidden,
            student_hidden,
            token_mask=token_mask,
        )
    if config.stage is MohawkStage.END_TO_END_DISTILLATION:
        if teacher_logits is None or student_logits is None:
            raise ValueError("Stage 3 requires teacher_logits and student_logits")
        return config.logits_weight * logits_distillation_loss(
            teacher_logits,
            student_logits,
            temperature=config.temperature,
            token_mask=token_mask,
        )
    raise ValueError(f"unsupported MOHAWK stage: {config.stage}")
