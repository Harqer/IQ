"""Differential GLM-first baseline parity check on an already-loaded teacher.

No source weights are downloaded. This only performs forward passes when
explicitly invoked with a loaded GLM teacher on a suitably provisioned host.
It requires Mamba branches to remain disabled; full hybrid/trained Mamba
quality and stateful decoding require separate tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .glm_first_mamba import GLMFirstIntegrationError, GLMFirstMambaHybrid


@dataclass(frozen=True)
class GLMReferenceParity:
    exact: bool
    max_abs_error: float
    mean_abs_error: float
    tokens: int
    logits_compared: int


def verify_zero_gate_reference_parity(
    teacher: nn.Module,
    hybrid: GLMFirstMambaHybrid,
    input_ids: torch.Tensor,
    *,
    use_cache: bool = False,
    logits_to_keep: int = 1,
) -> GLMReferenceParity:
    """Require equal logits from official GLM and the initially neutral overlay.

    The same teacher object is intentionally reused by the wrapper. This is
    the strict no-op integration gate, not a weight-transfer quality metric.
    """
    if hybrid.glm_model is not teacher:
        raise GLMFirstIntegrationError("hybrid must wrap the exact reference teacher")
    if any(b.enabled or bool(torch.count_nonzero(b.gate.detach()))
           for b in hybrid.branches):
        raise GLMFirstIntegrationError("baseline parity requires all branches disabled and zero-gated")
    if use_cache:
        raise GLMFirstIntegrationError("compare uncached GLM forwards first")
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise GLMFirstIntegrationError("input_ids must be a nonempty [batch, seq] integer matrix")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise GLMFirstIntegrationError("input_ids must contain integer GLM token IDs")
    if input_ids.min() < 0 or input_ids.max() >= hybrid.contract.glm_vocab:
        raise GLMFirstIntegrationError("token ID outside GLM vocabulary")

    old_teacher_training = teacher.training
    old_hybrid_training = hybrid.training
    try:
        teacher.eval()
        hybrid.eval()
        with torch.inference_mode():
            kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
            # logit slicing is supported by the official GLM-5.3 CausalLM but
            # also by small CPU compatibility fixtures.
            if logits_to_keep:
                kwargs["logits_to_keep"] = logits_to_keep
            baseline = teacher(**kwargs).logits
            candidate = hybrid(**kwargs).logits
            if baseline.shape != candidate.shape:
                raise GLMFirstIntegrationError("GLM baseline/hybrid logit shape differs")
            if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(candidate).all()):
                raise GLMFirstIntegrationError("nonfinite logits in teacher/hybrid comparison")
            diff = (baseline.float() - candidate.float()).abs()
            if not torch.equal(baseline, candidate):
                raise GLMFirstIntegrationError(
                    "zero-gated GLM-first overlay changed teacher logits "
                    f"(max_abs={float(diff.max()):.6g}); stop integration"
                )
            return GLMReferenceParity(
                exact=True,
                max_abs_error=float(diff.max()),
                mean_abs_error=float(diff.mean()),
                tokens=int(input_ids.numel()),
                logits_compared=int(diff.numel()),
            )
    finally:
        # Frozen GLM teacher is still held in eval mode by the hybrid wrapper;
        # restoring a training flag would violate the zero-gate reference policy.
        if old_hybrid_training and not hybrid._freeze_glm:
            hybrid.train(True)
        if old_teacher_training and not hybrid._freeze_glm:
            teacher.train(True)
