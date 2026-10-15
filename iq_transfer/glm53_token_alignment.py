"""Causal-safe teacher-token to student-token alignment across tokenizers.

GLM-5.3's 154880-token lexical space and IQ's Nemotron-based 131072-token
space do not have positional/token-ID parity. Fast tokenizer
return_offsets_mapping gives character spans in the SAME raw input text;
we may use only source tokens that have *completed* at the target token end
to prevent look-ahead. The mapped hidden representation is a distillation
target, NOT a functional conversion of GLM's attention operator.

Docs: https://huggingface.co/docs/transformers/main_classes/tokenizer
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


class GLMTokenAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class CausalSpanAlignment:
    mapped_teacher_output: torch.Tensor  # [B,IQ_tokens,GLM_hidden]
    supervision_mask: torch.Tensor  # [B,IQ_tokens]
    matched_source_tokens: torch.Tensor  # [B,IQ_tokens]
    coverage_ratio: float
    method: str = "completed_source_token_char_overlap_v1"


def _validate_offsets(offsets: torch.Tensor, *, name: str) -> None:
    if offsets.ndim != 3 or offsets.shape[-1] != 2 or offsets.dtype not in (
        torch.int32, torch.int64
    ):
        raise GLMTokenAlignmentError(
            f"{name} must contain int [batch,seq,2] character spans"
        )
    starts, ends = offsets.unbind(dim=-1)
    if bool((starts < 0).any()) or bool((ends < starts).any()):
        raise GLMTokenAlignmentError(f"{name} contains invalid character spans")
    # Special tokens may legally carry (0,0) and are unsupervised.
    for row in range(offsets.shape[0]):
        positions = offsets[row]
        completed = positions[:, 1] > positions[:, 0]
        regular = positions[completed]
        if regular.shape[0] > 1:
            if bool((regular[1:, 0] < regular[:-1, 0]).any()):
                raise GLMTokenAlignmentError(f"{name} spans are not ordered")
            if bool((regular[1:, 0] < regular[:-1, 1]).any()):
                raise GLMTokenAlignmentError(f"{name} spans overlap")


def align_teacher_by_char_spans(
    teacher_output: torch.Tensor,
    source_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    *,
    source_text_hashes: torch.Tensor,
    target_text_hashes: torch.Tensor,
) -> CausalSpanAlignment:
    """Map GLM outputs to IQ token positions without reading future text.

    Only GLM tokens overlapping the target token that have ended at or before
    the IQ token's character end contribute. Unmatched IQ tokens, whitespace
    gaps, BOS/EOS and incomplete source tokens are masked from loss.
    """
    _validate_offsets(source_offsets, name="GLM offsets")
    _validate_offsets(target_offsets, name="IQ offsets")
    if teacher_output.ndim != 3 or teacher_output.shape[:2] != source_offsets.shape[:2]:
        raise GLMTokenAlignmentError("source attention output and offsets differ")
    if source_offsets.shape[0] != target_offsets.shape[0] or target_offsets.shape[1] < 1:
        raise GLMTokenAlignmentError("paired tokenizations must share batch examples")
    if (source_text_hashes.dtype != torch.uint8 or
        target_text_hashes.dtype != torch.uint8 or
        source_text_hashes.shape != (teacher_output.shape[0], 32) or
        target_text_hashes.shape != (teacher_output.shape[0], 32)):
        raise GLMTokenAlignmentError("each raw text identity must be 32 SHA-256 bytes")
    if not torch.equal(source_text_hashes.cpu(), target_text_hashes.cpu()):
        raise GLMTokenAlignmentError("GLM and IQ captures do not share exact raw text")
    if not bool(torch.isfinite(teacher_output).all()):
        raise GLMTokenAlignmentError("source attention outputs contain nonfinite values")

    device = teacher_output.device
    src = source_offsets.to(device)
    dst = target_offsets.to(device)
    ss = src[:, :, 0].unsqueeze(1)
    se = src[:, :, 1].unsqueeze(1)
    ts = dst[:, :, 0].unsqueeze(-1)
    te = dst[:, :, 1].unsqueeze(-1)
    overlap = (torch.minimum(se, te) - torch.maximum(ss, ts)).clamp_min(0)
    # GLM source token must end by IQ's current token end to avoid
    # inadvertently teaching IQ to depend on unseen future characters.
    source_complete = se <= te
    source_nonempty = se > ss
    target_nonempty = te > ts
    weights = (
        overlap * (source_complete & source_nonempty & target_nonempty)
    ).to(dtype=torch.float32)
    denom = weights.sum(dim=-1)
    supervision = denom > 0
    if not bool(supervision.any()):
        raise GLMTokenAlignmentError(
            "no causally completed GLM tokens overlap an IQ token; "
            "paired tokenizer alignment is unsupported"
        )
    target = torch.bmm(
        weights, teacher_output.float()
    ) / denom.clamp_min(1).unsqueeze(-1)
    target = torch.where(
        supervision.unsqueeze(-1), target, torch.zeros_like(target)
    )
    count = (
        (weights > 0).sum(dim=-1).to(torch.int32)
    )
    return CausalSpanAlignment(
        mapped_teacher_output=target,
        supervision_mask=supervision,
        matched_source_tokens=count,
        coverage_ratio=float(supervision.float().mean().item()),
    )
