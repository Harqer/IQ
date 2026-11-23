"""One-step GLM-native teacher/logit distillation for the gated Mamba hybrid.

Training entry point only. It does NOT launch jobs, load donor checkpoints,
alter the legacy all256 bucket, or declare generation quality validated.

At a zero gate, teacher and student logits match, so teacher KL alone has
zero learning signal. An explicit next-token cross-entropy term supplies the
initial gradient for the gates; KL regularizes subsequent deviations.

Teacher targets MUST come from original GLM-5.3 logits in GLM token space and
must be bound to an independently pinned checkpoint and tokenizer revision.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from iq_model.glm_first_mamba import GLMFirstIntegrationError, GLMFirstMambaHybrid


class GLMFirstDistillationError(RuntimeError):
    pass


def input_ids_sha256(input_ids: Tensor) -> str:
    """Bind teacher logits to the exact GLM token-ID matrix and row ordering."""
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int64, torch.int32):
        raise GLMFirstDistillationError("source token IDs must be integer [batch, seq]")
    data = json.dumps(input_ids.detach().cpu().tolist(), separators=(",", ":"))
    return sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GLMTeacherLastTokenLogits:
    """Precomputed GLM teacher logits for the final prompt position of each row."""
    logits: Tensor  # [batch, GLM vocab]
    teacher_revision: str  # official immutable commit, not a nickname
    tokenizer_mapping_sha256: str
    input_ids_sha256: str  # exact prompt IDs and batch row order


@dataclass(frozen=True)
class GLMFirstDistillationConfig:
    teacher_revision: str
    tokenizer_mapping_sha256: str
    temperature: float = 2.0
    kl_weight: float = 1.0
    next_token_ce_weight: float = 1.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if len(self.teacher_revision) != 40 or any(
            x not in "0123456789abcdef" for x in self.teacher_revision.lower()
        ):
            raise ValueError("pin an exact 40-character teacher checkpoint revision")
        if len(self.tokenizer_mapping_sha256) != 64 or any(
            x not in "0123456789abcdef" for x in self.tokenizer_mapping_sha256.lower()
        ):
            raise ValueError("pin an exact 64-character GLM tokenizer mapping SHA-256")
        for key in ("temperature", "max_grad_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("kl_weight", "next_token_ce_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.next_token_ce_weight == 0:
            raise ValueError(
                "initial zero-gated adapters cannot learn from GLM teacher KL alone; "
                "require a positive next-token CE weight or a separate supervised objective"
            )


@dataclass(frozen=True)
class GLMFirstDistillationMetrics:
    loss: float
    kl: float
    next_token_ce: float
    trainable_parameters: int
    gate_abs_max: float


def _adapter_parameters(hybrid: GLMFirstMambaHybrid) -> tuple[nn.Parameter, ...]:
    return tuple(
        p for branch in hybrid.branches
        for p in (
            branch.gate,
            *branch.in_adapter.parameters(),
            *branch.out_adapter.parameters(),
        )
    )


def _optimizer_matches_adapters(
    hybrid: GLMFirstMambaHybrid,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    parameters = _adapter_parameters(hybrid)
    expected = {id(p) for p in parameters}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not actual or len(actual) != len(set(actual)) or set(actual) != expected:
        raise GLMFirstDistillationError(
            "optimizer must contain exactly Mamba residual gates and adapters; "
            "not original GLM or pretrained Mamba donor weights"
        )
    if any(not p.requires_grad for p in parameters):
        raise GLMFirstDistillationError("gates/adapters must remain trainable")
    if any(p.requires_grad for p in hybrid.glm_model.parameters()):
        raise GLMFirstDistillationError("GLM pretrained backbone must be frozen")
    if any(p.requires_grad for b in hybrid.branches for p in b.donor_block.parameters()):
        raise GLMFirstDistillationError("pretrained Mamba donor blocks must be frozen")
    return parameters


def glm_first_distillation_step(
    hybrid: GLMFirstMambaHybrid,
    *,
    input_ids: Tensor,
    next_token_ids: Tensor,
    teacher: GLMTeacherLastTokenLogits,
    optimizer: torch.optim.Optimizer,
    config: GLMFirstDistillationConfig,
) -> GLMFirstDistillationMetrics:
    """One explicit optimizer step; caller controls cost, batches and checkpoints.

    No internal gradient accumulation. Only the *final* prompt token's GLM
    distribution is compared. Logits are computed in float32 for KL/CE;
    no token-ID remapping is allowed.
    """
    params = _optimizer_matches_adapters(hybrid, optimizer)
    if (
        teacher.teacher_revision.lower() != config.teacher_revision.lower()
        or teacher.tokenizer_mapping_sha256.lower()
        != config.tokenizer_mapping_sha256.lower()
    ):
        raise GLMFirstDistillationError("teacher targets do not match pinned GLM/tokenizer revisions")
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise GLMFirstDistillationError("input_ids must be nonempty [batch,seq]")
    if input_ids.dtype not in (torch.int64, torch.int32):
        raise GLMFirstDistillationError("GLM input IDs must be integers")
    if teacher.input_ids_sha256 != input_ids_sha256(input_ids):
        raise GLMFirstDistillationError(
            "GLM teacher targets do not match the input token IDs and row order"
        )
    if next_token_ids.ndim != 1 or next_token_ids.shape[0] != input_ids.shape[0]:
        raise GLMFirstDistillationError("next_token_ids must be [batch]")
    if next_token_ids.dtype not in (torch.int64, torch.int32):
        raise GLMFirstDistillationError("GLM target token IDs must be integers")
    vocab = hybrid.contract.glm_vocab
    if (
        bool((input_ids < 0).any()) or bool((input_ids >= vocab).any())
        or bool((next_token_ids < 0).any()) or bool((next_token_ids >= vocab).any())
    ):
        raise GLMFirstDistillationError("input/target IDs fall outside GLM vocabulary")
    logits = teacher.logits
    if logits.ndim != 2 or logits.shape != (input_ids.shape[0], vocab):
        raise GLMFirstDistillationError("GLM teacher logits must be [batch,154880]")
    if not bool(torch.isfinite(logits).all()):
        raise GLMFirstDistillationError("teacher logits contain NaN/Inf")
    if logits.device != input_ids.device or next_token_ids.device != input_ids.device:
        raise GLMFirstDistillationError("GLM inputs, teacher logits and targets must share device")
    if not all(branch.enabled for branch in hybrid.branches):
        raise GLMFirstDistillationError("enable all Mamba branches explicitly before training")

    hybrid.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = hybrid(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    student_logits = getattr(outputs, "logits", None)
    if not isinstance(student_logits, Tensor) or student_logits.ndim != 3:
        raise GLMFirstDistillationError("hybrid model did not return causal LM logits")
    if student_logits.shape[0] != input_ids.shape[0] or student_logits.shape[-1] != vocab:
        raise GLMFirstDistillationError("student logits disagree with GLM vocabulary")
    final_logits = student_logits[:, -1, :].float()
    if not bool(torch.isfinite(final_logits).all()):
        raise GLMFirstDistillationError("hybrid logits contain NaN/Inf")
    detached_teacher = logits.detach().float()
    temperature = config.temperature
    kl = F.kl_div(
        F.log_softmax(final_logits / temperature, dim=-1),
        F.softmax(detached_teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    ce = F.cross_entropy(final_logits, next_token_ids.long())
    loss = config.kl_weight * kl + config.next_token_ce_weight * ce
    if not bool(torch.isfinite(loss)):
        raise GLMFirstDistillationError("GLM-first distillation loss is not finite")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise GLMFirstDistillationError("hybrid gates/adapters have missing or nonfinite gradients")
    nn.utils.clip_grad_norm_(params, config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return GLMFirstDistillationMetrics(
        loss=float(loss.detach()),
        kl=float(kl.detach()),
        next_token_ce=float(ce.detach()),
        trainable_parameters=sum(p.numel() for p in params),
        gate_abs_max=max(float(b.gate.detach().abs()) for b in hybrid.branches),
    )
