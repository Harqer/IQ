"""GLM-5.3 teacher-logit distillation into the GLM-tokenized native Mamba-3 MIMO.

The teacher provides logits for the same GLM token IDs. Training updates the
student's lexical width adapters and optionally its native MIMO blocks; GLM
embedding/head parameters remain exact and frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from iq_model.glm_mamba3_student import GLMMamba3Student, GLMMambaError


class MambaGLMDistillationError(RuntimeError):
    pass


def token_batch_fingerprint(input_ids: Tensor) -> str:
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise MambaGLMDistillationError("teacher input IDs must be [batch,seq] integers")
    content = json.dumps(input_ids.detach().cpu().tolist(),
                         separators=(",", ":")).encode("utf-8")
    return sha256(content).hexdigest()


@dataclass(frozen=True)
class GLMLogitBatch:
    logits: Tensor
    input_ids_sha256: str
    tokenizer_mapping_sha256: str
    teacher_revision: str


@dataclass(frozen=True)
class DistillSettings:
    teacher_revision: str
    tokenizer_mapping_sha256: str
    kl_weight: float = 1.0
    ce_weight: float = 0.2
    temperature: float = 2.0
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for label, length in (("teacher_revision", 40),
                              ("tokenizer_mapping_sha256", 64)):
            value = getattr(self, label)
            if (len(value) != length
                    or any(x not in "0123456789abcdef" for x in value.lower())):
                raise ValueError(f"{label} must be a pinned {length}-hex digest")
        for key in ("kl_weight", "ce_weight", "temperature", "max_grad_norm"):
            x = getattr(self, key)
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.temperature <= 0 or self.max_grad_norm <= 0:
            raise ValueError("temperature and gradient bound must be positive")
        if self.kl_weight + self.ce_weight == 0:
            raise ValueError("at least one supervision term must be nonzero")



@torch.inference_mode()
def record_glm_teacher_last_token_logits(
    teacher_model: nn.Module,
    input_ids: Tensor,
    *,
    teacher_revision: str,
    tokenizer_mapping_sha256: str,
    expected_vocab: int = 154880,
) -> GLMLogitBatch:
    """Capture *real* official GLM teacher logits for the same GLM IDs.

    Teacher execution is caller-owned. The output is not a fabricated target
    distribution; it comes from the supplied GLM causal-LM forward.
    """
    if len(teacher_revision) != 40 or any(
        c not in "0123456789abcdef" for c in teacher_revision.lower()
    ):
        raise MambaGLMDistillationError("pin the GLM teacher source revision")
    if len(tokenizer_mapping_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in tokenizer_mapping_sha256.lower()
    ):
        raise MambaGLMDistillationError("pin the GLM tokenizer mapping fingerprint")
    config = getattr(teacher_model, "config", None)
    if (getattr(config, "model_type", None) != "glm_moe_dsa"
            or getattr(config, "vocab_size", None) != expected_vocab):
        raise MambaGLMDistillationError("teacher is not GLM-5.3 causal LM")
    identity = token_batch_fingerprint(input_ids)
    if (bool((input_ids < 0).any())
            or bool((input_ids >= expected_vocab).any())):
        raise MambaGLMDistillationError("teacher input IDs outside GLM vocabulary")
    teacher_model.eval()
    outputs = teacher_model(
        input_ids=input_ids, use_cache=False, logits_to_keep=1
    )
    logits = getattr(outputs, "logits", None)
    if (not isinstance(logits, Tensor) or logits.shape !=
            (input_ids.shape[0], 1, expected_vocab)):
        raise MambaGLMDistillationError("GLM teacher did not return expected final-token logits")
    if not bool(torch.isfinite(logits).all()):
        raise MambaGLMDistillationError("non-finite GLM teacher logits")
    return GLMLogitBatch(
        logits=logits[:, -1].detach(),
        input_ids_sha256=identity,
        tokenizer_mapping_sha256=tokenizer_mapping_sha256.lower(),
        teacher_revision=teacher_revision.lower(),
    )


def student_trainable_parameters(model: GLMMamba3Student) -> tuple[nn.Parameter, ...]:
    return tuple(p for p in model.parameters() if p.requires_grad)


def glm_to_mamba_distillation_step(
    student: GLMMamba3Student,
    *,
    input_ids: Tensor,
    next_token_ids: Tensor,
    teacher: GLMLogitBatch,
    optimizer: torch.optim.Optimizer,
    settings: DistillSettings,
) -> dict[str, float]:
    """One explicit optimizer step with independently supplied GLM teacher logits."""
    if not student.tokenizer_mapping_pinned:
        raise MambaGLMDistillationError("student tokenizer mapping must be independently pinned")
    if (student.mapping_sha256 != settings.tokenizer_mapping_sha256
            or teacher.tokenizer_mapping_sha256 != settings.tokenizer_mapping_sha256
            or teacher.teacher_revision.lower() != settings.teacher_revision.lower()):
        raise MambaGLMDistillationError("GLM teacher, tokenizer, or student revision mismatch")
    student.validate_input(input_ids)
    if teacher.input_ids_sha256 != token_batch_fingerprint(input_ids):
        raise MambaGLMDistillationError("GLM teacher logits do not match the exact input IDs")
    if (next_token_ids.shape != (input_ids.shape[0],)
            or next_token_ids.dtype not in (torch.int32, torch.int64)
            or bool((next_token_ids < 0).any())
            or bool((next_token_ids >= student.geometry.glm_vocab).any())):
        raise MambaGLMDistillationError("invalid GLM next-token targets")
    if (teacher.logits.shape != (input_ids.shape[0], student.geometry.glm_vocab)
            or not bool(torch.isfinite(teacher.logits).all())):
        raise MambaGLMDistillationError("invalid GLM teacher logits")
    if teacher.logits.device != input_ids.device or next_token_ids.device != input_ids.device:
        raise MambaGLMDistillationError("student inputs and teacher targets must share a device")
    params = student_trainable_parameters(student)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not params or len(ids) != len(set(ids)) or set(ids) != {id(p) for p in params}:
        raise MambaGLMDistillationError("optimizer parameters disagree with configured student training")
    if student.core.backbone.embedding.weight.requires_grad or student.core.lm_head.weight.requires_grad:
        raise MambaGLMDistillationError("original GLM embedding/head weights must remain frozen")

    student.train()
    optimizer.zero_grad(set_to_none=True)
    student_logits = student(input_ids, num_last_tokens=1).logits[:, -1, :].float()
    if student_logits.shape != teacher.logits.shape or not bool(torch.isfinite(student_logits).all()):
        raise MambaGLMDistillationError("student GLM-vocabulary logits are invalid")
    reference = teacher.logits.detach().float()
    temperature = settings.temperature
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(reference / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    ce = F.cross_entropy(student_logits, next_token_ids.long())
    loss = settings.kl_weight * kl + settings.ce_weight * ce
    if not bool(torch.isfinite(loss)):
        raise MambaGLMDistillationError("non-finite distillation loss")
    loss.backward()
    for param in params:
        if param.grad is None or not bool(torch.isfinite(param.grad).all()):
            raise MambaGLMDistillationError("missing or nonfinite student gradients")
    nn.utils.clip_grad_norm_(params, settings.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "teacher_kl": float(kl.detach()),
        "next_token_ce": float(ce.detach()),
        "trainable_parameters": float(sum(p.numel() for p in params)),
    }
