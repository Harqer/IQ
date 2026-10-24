"""Behavioral transfer from authenticated GLM teacher *text*, never cross-vocabulary logits.

The GLM and Mamba-3 vocabularies differ. This module supervises IQ on
teacher-generated UTF-8 continuations re-tokenized with IQ's own tokenizer.
Provenance checks prove artifact integrity, NOT that a remote teacher was truthful.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Iterable

import torch
from torch.nn import functional as F


class BehaviorTransferError(ValueError):
    pass


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_GLM_TEACHERS = frozenset({
    "zai-org/GLM-5.3-BF16", "zai-org/GLM-5.3", "glm-5.3",
})


@dataclass(frozen=True)
class TeacherExample:
    prompt: str
    response: str
    teacher_model: str
    teacher_revision: str
    request_id: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TeacherExample":
        if not isinstance(record, dict):
            raise BehaviorTransferError("teacher record must be a JSON object")
        needed = ("prompt", "response", "teacher_model", "teacher_revision", "request_id")
        if set(record) != set(needed):
            raise BehaviorTransferError(f"teacher record must have exactly {needed}")
        if not all(isinstance(record[key], str) and record[key].strip() for key in needed):
            raise BehaviorTransferError("all teacher record fields must be nonempty strings")
        if record["teacher_model"] not in SUPPORTED_GLM_TEACHERS:
            raise BehaviorTransferError("teacher source must be a declared GLM-5.3 serving model")
        if not _SHA40.fullmatch(record["teacher_revision"].lower()):
            raise BehaviorTransferError("teacher revision must be an immutable 40-character git SHA")
        if len(record["prompt"]) > 16000 or len(record["response"]) > 64000:
            raise BehaviorTransferError("teacher record exceeds configured text budget")
        return cls(**{key: record[key] for key in needed})

    def content_digest(self) -> str:
        content = {"prompt": self.prompt, "response": self.response}
        return sha256(json.dumps(
            content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()


def validate_teacher_examples(records: Iterable[dict[str, Any]]) -> list[TeacherExample]:
    examples = [TeacherExample.from_record(r) for r in records]
    if len(examples) < 2:
        raise BehaviorTransferError("need two or more real teacher examples")
    prompt_hashes: set[str] = set()
    ids: set[str] = set()
    revisions = {ex.teacher_revision.lower() for ex in examples}
    models = {ex.teacher_model for ex in examples}
    if len(revisions) != 1 or len(models) != 1:
        raise BehaviorTransferError("all records must share a pinned teacher revision and model identity")
    for ex in examples:
        digest = sha256(ex.prompt.encode("utf-8")).hexdigest()
        if digest in prompt_hashes:
            raise BehaviorTransferError("duplicate prompt would leak across splits")
        if ex.request_id in ids:
            raise BehaviorTransferError("duplicate remote request_id")
        prompt_hashes.add(digest)
        ids.add(ex.request_id)
    return examples


def build_iq_example(tokenizer: Any, example: TeacherExample, *, max_tokens: int) -> dict[str, torch.Tensor]:
    """Mask question tokens. Train on answer tokens and EOS using IQ IDs only."""
    if max_tokens < 4:
        raise BehaviorTransferError("max_tokens must be at least 4")
    question = "Question: " + example.prompt.strip() + "\nAnswer:"
    answer = " " + example.response.strip()
    q = tokenizer(question, add_special_tokens=False)["input_ids"]
    a = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not isinstance(q, list) or not isinstance(a, list) or not q or not a:
        raise BehaviorTransferError("tokenizer returned invalid/empty token IDs")
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise BehaviorTransferError("IQ tokenizer needs EOS")
    tokens = ([int(bos)] if bos is not None else []) + [int(x) for x in q]
    target = [int(x) for x in a] + [int(eos)]
    ids = tokens + target
    if len(ids) > max_tokens:
        raise BehaviorTransferError(
            f"example is {len(ids)} tokens > max_tokens={max_tokens}; "
            "reject instead of silently truncating the teacher's response"
        )
    if min(ids) < 0 or max(ids) >= len(tokenizer):
        raise BehaviorTransferError("tokenizer produced IDs outside IQ vocabulary")
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor([-100] * len(tokens) + target, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.bool),
    }


def collate_behavior_examples(examples: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    if not examples:
        raise BehaviorTransferError("cannot collate empty behavior batch")
    length = max(int(item["input_ids"].numel()) for item in examples)
    input_ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), length), dtype=torch.bool)
    for index, ex in enumerate(examples):
        n = ex["input_ids"].numel()
        if ex["labels"].shape != (n,) or ex["attention_mask"].shape != (n,):
            raise BehaviorTransferError("inconsistent IQ example shapes")
        input_ids[index, :n] = ex["input_ids"]
        labels[index, :n] = ex["labels"]
        mask[index, :n] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def causal_response_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Correct next-token CE for IQ's model(input_ids) logits."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise BehaviorTransferError("expected logits [B,T,V] and labels [B,T]")
    if logits.shape[1] < 2 or not bool((labels[:, 1:] != -100).any()):
        raise BehaviorTransferError("no supervised response tokens")
    if not bool(torch.isfinite(logits).all()):
        raise BehaviorTransferError("student logits are not finite")
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    if bool(((shifted_labels < 0) & (shifted_labels != -100)).any()):
        raise BehaviorTransferError("unexpected negative token labels")
    if bool((shifted_labels >= logits.shape[-1]).any()):
        raise BehaviorTransferError("response IDs exceed logits vocabulary")
    return F.cross_entropy(
        shifted_logits.reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=-100,
    )
