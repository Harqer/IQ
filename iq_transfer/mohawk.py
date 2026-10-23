"""MOHAWK objectives for GLM-5.3 → IQ Mamba-3 MIMO transfer.

Stage 1 accepts *measured* effective token-mixing operators only. In particular,
IQ's native Mamba-3 MIMO output activations are not a token-mixing matrix.
Stage 2 operates on byte-aligned, target-basis residual states. Stage 3
requires a verified identical output vocabulary; GLM-5.3 and Mamba-3 use
different tokenizers, so raw cross-model logit KL is intentionally rejected.
"""

from __future__ import annotations

import math
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
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for name in ("matrix_weight", "hidden_weight", "logits_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} has non-finite values")


def matrix_orientation_loss(
    teacher_mixer: Tensor,
    student_mixer: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Stage 1: squared Frobenius distance for comparable [*, T, T] operators."""
    if teacher_mixer.ndim < 2 or teacher_mixer.shape != student_mixer.shape:
        raise ValueError(
            "teacher/student mixer operators must have identical [*, T, T] shapes"
        )
    if teacher_mixer.shape[-1] != teacher_mixer.shape[-2]:
        raise ValueError("token-mixing matrices must be square")
    _require_finite("teacher_mixer", teacher_mixer)
    _require_finite("student_mixer", student_mixer)
    delta = (student_mixer.float() - teacher_mixer.float()).square()
    if mask is None:
        return delta.mean()
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(delta.shape[-2:]):
        raise ValueError("mixer mask must be boolean [query, key]")
    active = int(mask.sum())
    if active == 0:
        raise ValueError("mixer mask must select at least one position")
    weighted = delta * mask.to(device=delta.device, dtype=delta.dtype)
    return weighted.sum() / (active * (delta.numel() // mask.numel()))


def hidden_alignment_loss(
    teacher_hidden: Tensor,
    student_hidden: Tensor,
    *,
    token_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Stage 2: masked, scale-normalized L2 representation alignment.

    Inputs MUST already use the same UTF-8 byte-aligned token positions and
    IQ's 4096-wide target basis. This function cannot construct that alignment.
    """
    if teacher_hidden.ndim != 3 or teacher_hidden.shape != student_hidden.shape:
        raise ValueError("hidden states must match [batch, IQ-token, IQ-hidden]")
    _require_finite("teacher_hidden", teacher_hidden)
    _require_finite("student_hidden", student_hidden)
    if eps <= 0:
        raise ValueError("eps must be positive")
    squared = (student_hidden.float() - teacher_hidden.float()).square().mean(-1)
    reference = teacher_hidden.float().square().mean(-1)
    if token_mask is not None:
        if token_mask.dtype != torch.bool or token_mask.shape != squared.shape:
            raise ValueError("token_mask must be boolean [batch, IQ-token]")
        if not bool(token_mask.any()):
            raise ValueError("token_mask must select at least one token")
        squared = squared[token_mask]
        reference = reference[token_mask]
    return squared.mean() / reference.mean().clamp_min(eps)


def logits_distillation_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    teacher_tokenizer_sha256: str,
    student_tokenizer_sha256: str,
    temperature: float = 1.0,
    token_mask: Tensor | None = None,
) -> Tensor:
    """Stage 3: KL only in the exact same, verified token-ID vocabulary.

    GLM-5.3 → Mamba-3 native logits MUST NOT use this loss as-is. Their
    vocabularies differ; shared tensor shape would not establish token equivalence.
    """
    if (
        len(teacher_tokenizer_sha256) != 64
        or len(student_tokenizer_sha256) != 64
        or teacher_tokenizer_sha256 != student_tokenizer_sha256
    ):
        raise ValueError(
            "MOHAWK logits KL requires identical verified tokenizer SHA-256"
        )
    if teacher_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("logits must match [batch, IQ-token, shared-vocabulary]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    _require_finite("teacher_logits", teacher_logits)
    _require_finite("student_logits", student_logits)
    t = float(temperature)
    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logits.float() / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / t, dim=-1)
    per_token = (
        F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(-1) * t * t
    )
    if token_mask is None:
        return per_token.mean()
    if token_mask.dtype != torch.bool or token_mask.shape != per_token.shape:
        raise ValueError("token_mask must be boolean [batch, IQ-token]")
    if not bool(token_mask.any()):
        raise ValueError("token_mask must select at least one token")
    return per_token[token_mask].mean()


def mohawk_loss(
    config: MohawkConfig,
    *,
    teacher_mixer: Tensor | None = None,
    student_mixer: Tensor | None = None,
    teacher_hidden: Tensor | None = None,
    student_hidden: Tensor | None = None,
    teacher_logits: Tensor | None = None,
    student_logits: Tensor | None = None,
    teacher_tokenizer_sha256: str = "",
    student_tokenizer_sha256: str = "",
    token_mask: Tensor | None = None,
    matrix_mask: Tensor | None = None,
) -> Tensor:
    """Fail-closed objective dispatcher; no synthetic or silent fallback stages."""
    if config.stage is MohawkStage.MATRIX_ORIENTATION:
        if teacher_mixer is None or student_mixer is None:
            raise ValueError("Stage 1 requires real teacher_mixer and student_mixer")
        return config.matrix_weight * matrix_orientation_loss(
            teacher_mixer,
            student_mixer,
            mask=matrix_mask,
        )
    if config.stage is MohawkStage.HIDDEN_ALIGNMENT:
        if teacher_hidden is None or student_hidden is None:
            raise ValueError(
                "Stage 2 requires byte-aligned teacher_hidden and student_hidden"
            )
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
            teacher_tokenizer_sha256=teacher_tokenizer_sha256,
            student_tokenizer_sha256=student_tokenizer_sha256,
            temperature=config.temperature,
            token_mask=token_mask,
        )
    raise ValueError(f"unsupported MOHAWK stage: {config.stage}")
