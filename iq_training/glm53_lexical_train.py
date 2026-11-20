"""GLM-5.3 lexical-only adaptation of a verified, weight-loaded full235 IQ.

This changes IN-MEMORY training parameters only. It never downloads HF weight
shards, writes to a bucket, trains a randomly initialized full235 backbone,
or certifies language quality. The source body must be loaded/verified by a
separate production loader before the production gate can be satisfied.

IQHybridForCausalLM uses separate embed_tokens and lm_head parameters.
Freeze all 235 physical layers plus norm/reasoning; update only lexical rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from iq_cloud.glm53_lexical import (
    GLM_PINNED_MAPPING_SHA256, GLM_REVISION, GLM_VOCAB, IQ_WIDTH,
)


class LexicalTrainingError(ValueError):
    """The training input or full235 provenance cannot be trusted."""


@dataclass(frozen=True)
class LexicalTrainingMetrics:
    total_loss: float
    causal_loss: float
    teacher_kl: float | None
    supervised_tokens: int
    trained_checkpoint_verified: bool = False
    generation_quality_verified: bool = False


def _require_ids(ids: Iterable[int], vocab_size: int) -> tuple[int, ...]:
    if type(vocab_size) is not int or vocab_size < 2:
        raise LexicalTrainingError("invalid lexical vocabulary capacity")
    try:
        values = tuple(ids)
    except TypeError as exc:
        raise LexicalTrainingError("defined token IDs must be iterable") from exc
    if (not values or len(values) != len(set(values))
            or any(type(i) is not int or i < 0 or i >= vocab_size for i in values)):
        raise LexicalTrainingError("defined token IDs must be unique and in range")
    return tuple(sorted(values))


def validate_train_holdout_bytes(
    training_samples: Iterable[bytes], heldout_samples: Iterable[bytes],
) -> dict[str, Any]:
    """Deterministic exact-text leakage guard; not a semantic near-dup check."""
    def checksums(samples: Iterable[bytes], name: str) -> set[str]:
        hashes: set[str] = set()
        for record in samples:
            if not isinstance(record, bytes) or not record:
                raise LexicalTrainingError(f"{name} samples must be nonempty UTF-8 bytes")
            try:
                record.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise LexicalTrainingError(f"{name} contains invalid UTF-8") from exc
            digest = sha256(record).hexdigest()
            if digest in hashes:
                raise LexicalTrainingError(f"duplicate {name} sample")
            hashes.add(digest)
        if not hashes:
            raise LexicalTrainingError(f"{name} split cannot be empty")
        return hashes
    train = checksums(training_samples, "training")
    heldout = checksums(heldout_samples, "heldout")
    if train & heldout:
        raise LexicalTrainingError("train/heldout exact-byte sample leakage")
    return {
        "training_samples": len(train),
        "heldout_samples": len(heldout),
        "exact_duplicate_leakage": False,
        "semantic_near_duplicate_leakage_checked": False,
        "heldout_from_calibration_verified": False,
    }


class IQ235GLM53LexicalObjective(nn.Module):
    """Run differentiable causal LM + optional teacher KD on the real IQ student.

    The full-sized path requires the pinned GLM token IDs and proof from an
    *external trusted loader* that the complete IQ235 pretrained checkpoint
    was actually installed. No such proof exists in architecture-only mode.
    A small-model mode is available solely for CPU regression tests.
    """

    def __init__(
        self,
        student: nn.Module,
        defined_token_ids: Iterable[int],
        *,
        tokenizer_mapping_sha256: str,
        teacher_weight: float = 0.5,
        temperature: float = 2.0,
        source_integrity: Mapping[str, Any] | None = None,
        teacher_model_revision: str | None = None,
        test_only_small_model: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(student, nn.Module):
            raise LexicalTrainingError("student must be a Torch module")
        if (not math.isfinite(teacher_weight) or not 0 <= teacher_weight <= 1
                or not math.isfinite(temperature) or temperature <= 0):
            raise LexicalTrainingError("invalid distillation weighting or temperature")
        if (not hasattr(student, "embed_tokens") or not hasattr(student, "lm_head")
                or not isinstance(student.embed_tokens, nn.Embedding)
                or not isinstance(student.lm_head, nn.Linear)):
            raise LexicalTrainingError("IQ needs a separate embedding and output head")
        embedding, head = student.embed_tokens.weight, student.lm_head.weight
        if embedding.data_ptr() == head.data_ptr():
            raise LexicalTrainingError("GLM-5.3 embedding and head must be untied")
        if head.requires_grad is False and embedding.requires_grad is False:
            # Frozen base weights can still be unfrozen, but a real loader
            # must have injected them before beginning adaptation.
            pass
        if embedding.ndim != 2 or head.shape != embedding.shape or head.device != embedding.device:
            raise LexicalTrainingError("embedding/head shape or device mismatch")
        if not test_only_small_model:
            cfg = getattr(student, "config", None)
            schedule = getattr(getattr(cfg, "schedule", None), "layers", None)
            model = getattr(cfg, "model", None)
            if (not isinstance(schedule, tuple) or len(schedule) != 235
                    or len(getattr(student, "layers", ())) != 235
                    or getattr(model, "vocab_size", None) != GLM_VOCAB
                    or getattr(model, "hidden_size", None) != IQ_WIDTH
                    or getattr(model, "tie_word_embeddings", None) is not False
                    or tuple(embedding.shape) != (GLM_VOCAB, IQ_WIDTH)):
                raise LexicalTrainingError("full235 untied GLM architecture required")
            if tokenizer_mapping_sha256 != GLM_PINNED_MAPPING_SHA256:
                raise LexicalTrainingError("pinned GLM tokenizer mapping digest mismatch")
            if teacher_weight and teacher_model_revision != GLM_REVISION:
                raise LexicalTrainingError(
                    "GLM teacher model must match the pinned full BF16 revision"
                )
            # A caller-supplied mapping is NOT trusted evidence. An actual
            # weight loader must prove and bind all loaded payload digests to
            # this model before a production path can be enabled. No such
            # loader is currently implemented in this branch.
            raise LexicalTrainingError(
                "IQ235 production lexical training blocked: there is no "
                "verified full-checkpoint loader/attestation; metadata and "
                "caller-supplied source_integrity flags cannot authorize it"
            )
        else:
            # TEST_ONLY is never a backdoor for a full-sized or pretrained IQ.
            # Production geometry is deliberately too large for this CPU
            # fixture path, even if a caller sets the flag explicitly.
            if (tokenizer_mapping_sha256 != "TEST_ONLY"
                    or embedding.shape[0] > 32
                    or embedding.shape[1] > 64
                    or len(getattr(student, "layers", ())) >= 235):
                raise LexicalTrainingError(
                    "TEST_ONLY restricted to tiny synthetic CPU fixtures"
                )
        defined = _require_ids(defined_token_ids, embedding.shape[0])
        if not test_only_small_model and len(defined) != 154856:
            raise LexicalTrainingError("all 154856 defined GLM IDs are required")
        if not test_only_small_model and 154820 not in defined:
            raise LexicalTrainingError("GLM EOS/pad ID must remain addressable")
        lookup = torch.full((embedding.shape[0],), -1, dtype=torch.long)
        lookup[list(defined)] = torch.arange(len(defined), dtype=torch.long)
        self.register_buffer("defined_ids", torch.tensor(defined, dtype=torch.long))
        self.register_buffer("dense_id_lookup", lookup)
        self.student = student
        self.teacher_weight = float(teacher_weight)
        self.temperature = float(temperature)
        self.test_only_small_model = test_only_small_model
        self.student.requires_grad_(False)
        self.student.embed_tokens.weight.requires_grad_(True)
        self.student.lm_head.weight.requires_grad_(True)
        actual_trainable = {
            name for name, p in self.student.named_parameters() if p.requires_grad
        }
        if actual_trainable != {"embed_tokens.weight", "lm_head.weight"}:
            raise LexicalTrainingError(
                f"nonlexical trainable parameters detected: {sorted(actual_trainable)}"
            )

    def _losses(
        self, input_ids: torch.Tensor, labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
        teacher_input_ids: torch.Tensor | None,
        **student_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        if (input_ids.ndim != 2 or input_ids.shape[1] < 2
                or input_ids.dtype not in (torch.int64, torch.int32)
                or labels.shape != input_ids.shape
                or labels.dtype not in (torch.int64, torch.int32)):
            raise LexicalTrainingError("causal GLM input IDs and labels must be integer [B,T>=2]")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape
            or attention_mask.dtype != torch.bool
        ):
            raise LexicalTrainingError("attention mask must be boolean [B,T]")
        lookup = self.dense_id_lookup.to(input_ids.device)
        if bool(((input_ids < 0) | (input_ids >= lookup.numel())).any()):
            raise LexicalTrainingError("input token ID outside GLM vocabulary")
        if bool((lookup[input_ids.long()] < 0).any()):
            raise LexicalTrainingError("input contains undefined GLM token ID")
        # Reject invalid labels and teacher streams BEFORE the expensive
        # 235-layer multi-GPU forward pass.
        expected = (*input_ids.shape, lookup.numel())
        shifted = labels[:, 1:].long()
        valid = shifted != -100
        if attention_mask is not None:
            valid = valid & attention_mask[:, 1:]
        if not bool(valid.any()):
            raise LexicalTrainingError("no supervised next-token GLM positions")
        requested = shifted[valid]
        if bool(((requested < 0) | (requested >= lookup.numel())).any()):
            raise LexicalTrainingError("supervised target outside vocabulary")
        dense = lookup[requested.to(lookup.device)]
        if bool((dense < 0).any()):
            raise LexicalTrainingError("supervised target is an undefined token")
        if self.teacher_weight:
            if (not isinstance(teacher_logits, torch.Tensor)
                    or tuple(teacher_logits.shape) != expected
                    or not bool(torch.isfinite(teacher_logits).all())):
                raise LexicalTrainingError(
                    "aligned finite GLM teacher logits are required for KD"
                )
            if (not isinstance(teacher_input_ids, torch.Tensor)
                    or teacher_input_ids.shape != input_ids.shape
                    or teacher_input_ids.dtype not in (torch.int32, torch.int64)
                    or teacher_input_ids.device != input_ids.device
                    or not torch.equal(teacher_input_ids, input_ids)):
                raise LexicalTrainingError(
                    "GLM teacher input token IDs must exactly equal student input IDs"
                )
            if teacher_logits.device != input_ids.device:
                raise LexicalTrainingError("student and teacher logits must share a device")
        output = self.student(
            input_ids, labels=None, attention_mask=attention_mask,
            **student_kwargs,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
            raise LexicalTrainingError("full IQ model did not produce GLM-sized logits")
        if not bool(torch.isfinite(logits).all()):
            raise LexicalTrainingError("student logits contain nonfinite values")
        ids = self.defined_ids.to(logits.device)
        compact = logits[:, :-1].float().index_select(-1, ids)
        token_logits = compact[valid]
        causal = F.cross_entropy(token_logits, dense.to(token_logits.device))
        teacher_kl: torch.Tensor | None = None
        if self.teacher_weight:
            teacher = teacher_logits[:, :-1].float().index_select(-1, ids)
            t = self.temperature
            teacher_logp = F.log_softmax(teacher[valid] / t, dim=-1).detach()
            student_logp = F.log_softmax(token_logits / t, dim=-1)
            teacher_kl = F.kl_div(
                student_logp, teacher_logp.exp(), reduction="batchmean"
            ) * (t * t)
            total = (1 - self.teacher_weight) * causal + self.teacher_weight * teacher_kl
        else:
            total = causal
        if not bool(torch.isfinite(total)):
            raise LexicalTrainingError("nonfinite lexical training loss")
        return total, causal, teacher_kl, int(valid.sum().item())

    def forward(
        self, input_ids: torch.Tensor, *, labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        teacher_input_ids: torch.Tensor | None = None,
        **student_kwargs: Any,
    ) -> tuple[torch.Tensor, LexicalTrainingMetrics]:
        total, causal, kl, tokens = self._losses(
            input_ids, labels, attention_mask, teacher_logits,
            teacher_input_ids, **student_kwargs,
        )
        metrics = LexicalTrainingMetrics(
            total_loss=float(total.detach()),
            causal_loss=float(causal.detach()),
            teacher_kl=None if kl is None else float(kl.detach()),
            supervised_tokens=tokens,
        )
        return total, metrics


def lexical_train_step(
    objective: IQ235GLM53LexicalObjective,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    *,
    max_grad_norm: float = 1.0,
) -> LexicalTrainingMetrics:
    """One explicit optimizer step; no automatic checkpoint writes or claims."""
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise LexicalTrainingError("gradient clip bound must be positive")
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = objective(**batch)
    loss.backward()
    trainable = [p for p in objective.parameters() if p.requires_grad]
    if not trainable or any(
        p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable
    ):
        raise LexicalTrainingError("missing or nonfinite lexical gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    if not bool(torch.isfinite(grad_norm)):
        raise LexicalTrainingError("nonfinite lexical gradient norm")
    optimizer.step()
    return metrics


def evaluate_lexical_holdout(
    objective: IQ235GLM53LexicalObjective,
    batch: Mapping[str, torch.Tensor],
) -> LexicalTrainingMetrics:
    """Read-only held-out loss, not approval for pretrained generation.

    The caller must first establish the held-out dataset manifest and prove
    teacher logits came from the pinned GLM-5.3 revision for the exact IDs.
    """
    was_training = objective.training
    try:
        objective.eval()
        with torch.no_grad():
            _loss, metrics = objective(**batch)
    finally:
        objective.train(was_training)
    return metrics
