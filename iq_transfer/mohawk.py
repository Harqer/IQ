from __future__ import annotations

"""Small, architecture-agnostic losses used by MOHAWK-style transfer.

This module intentionally contains no model-specific logic.  Stage 1 aligns sequence
mixing operators, Stage 2 aligns block outputs, and Stage 3 distills end-to-end
predictions.  Keeping the losses separate makes it possible to test a new IQ mixer
without changing the donor adapter.
"""

import torch
import torch.nn.functional as F


def normalized_frobenius_loss(
    student_matrix: torch.Tensor,
    teacher_matrix: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MOHAWK Stage-1 loss normalized by the teacher matrix norm.

    The original MOHAWK objective uses Frobenius distance.  Normalization makes
    comparisons across sequence lengths and layers easier while preserving the same
    optimum.
    """
    if student_matrix.shape != teacher_matrix.shape:
        raise ValueError(
            f"matrix shape mismatch: student={tuple(student_matrix.shape)} "
            f"teacher={tuple(teacher_matrix.shape)}"
        )
    diff = torch.linalg.matrix_norm(student_matrix - teacher_matrix, ord="fro", dim=(-2, -1))
    scale = torch.linalg.matrix_norm(teacher_matrix, ord="fro", dim=(-2, -1)).clamp_min(eps)
    return (diff / scale).mean()


def hidden_state_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    eps: float = 1e-8,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """MOHAWK Stage-2 block-output alignment loss.

    The normalized MSE avoids layer-scale differences dominating the objective.  A
    small cosine term preserves direction as well as magnitude.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"hidden shape mismatch: student={tuple(student_hidden.shape)} "
            f"teacher={tuple(teacher_hidden.shape)}"
        )
    mse = F.mse_loss(student_hidden.float(), teacher_hidden.float())
    scale = teacher_hidden.float().square().mean().clamp_min(eps)
    cosine = 1.0 - F.cosine_similarity(
        student_hidden.float(), teacher_hidden.float(), dim=-1, eps=eps
    ).mean()
    return mse / scale + cosine_weight * cosine


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """MOHAWK Stage-3 teacher-to-student distribution matching."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logit shape mismatch: student={tuple(student_logits.shape)} "
            f"teacher={tuple(teacher_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t = float(temperature)
    teacher_prob = F.softmax(teacher_logits.float() / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / t, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)
