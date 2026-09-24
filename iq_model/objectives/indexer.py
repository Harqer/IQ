from __future__ import annotations

import torch
from torch.nn import functional as F


class IndexerObjectiveError(ValueError):
    pass


def lightning_indexer_kl_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL distillation loss for Lightning-indexer warm-up.

    Scores are aligned to the same compressed-entry axis. Invalid/future entries
    are excluded before normalization. Hard top-k is not part of this objective,
    so gradients train the indexer score surface directly.
    """

    if student_scores.ndim != 2 or teacher_scores.ndim != 2:
        raise IndexerObjectiveError(
            "student_scores and teacher_scores must be rank-2 [tokens, entries]"
        )
    if student_scores.shape != teacher_scores.shape:
        raise IndexerObjectiveError(
            "student_scores and teacher_scores must have identical shape"
        )
    if valid_mask.shape != student_scores.shape or valid_mask.dtype is not torch.bool:
        raise IndexerObjectiveError(
            "valid_mask must be bool with the same shape as the score tensors"
        )
    if temperature <= 0:
        raise IndexerObjectiveError("temperature must be positive")

    row_valid = valid_mask.any(dim=-1)
    if not bool(row_valid.any()):
        raise IndexerObjectiveError(
            "indexer distillation batch has no query with a valid compressed entry"
        )

    neg_inf = torch.tensor(
        float("-inf"),
        device=student_scores.device,
        dtype=student_scores.dtype,
    )
    student = student_scores / temperature
    teacher = teacher_scores / temperature
    student = torch.where(valid_mask, student, neg_inf)
    teacher = torch.where(valid_mask, teacher, neg_inf)

    student_log_probs = F.log_softmax(student[row_valid].float(), dim=-1)
    teacher_probs = F.softmax(teacher[row_valid].float(), dim=-1)
    return (
        F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False,
        )
        * (temperature * temperature)
    )


def lightning_indexer_topk_recall(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    if topk <= 0:
        raise IndexerObjectiveError("topk must be positive")
    if (
        student_scores.shape != teacher_scores.shape
        or valid_mask.shape != student_scores.shape
    ):
        raise IndexerObjectiveError("score/mask shapes must match")
    recalls: list[torch.Tensor] = []
    for row in range(student_scores.shape[0]):
        valid = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        k = min(topk, int(valid.numel()))
        s = student_scores[row].index_select(0, valid)
        t = teacher_scores[row].index_select(0, valid)
        student_idx = valid.index_select(0, s.topk(k).indices)
        teacher_idx = valid.index_select(0, t.topk(k).indices)
        overlap = torch.isin(student_idx, teacher_idx).float().sum()
        recalls.append(overlap / float(k))
    if not recalls:
        raise IndexerObjectiveError("no valid rows for top-k recall")
    return torch.stack(recalls).mean()
