"""Mathematical and numerical no-learning audit for GLM-first Mamba-3.

The theorem is unconditional: a linear map R^6144 -> R^2048 has kernel
dimension >= 4096, so it cannot reproduce all possible GLM residual states
by itself. This does *not* destroy the GLM residual pathway if the Mamba
branch is only additive, but a nonzero branch may change the GLM function.

The optional probe deliberately performs NO optimization, weight mutation,
calibration or model downloads. Finite probe equality is not a proof of
equivalence on all prompts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from iq_model.glm_first_mamba import GLMFirstIntegrationError, GLMFirstMambaHybrid


@dataclass(frozen=True)
class WeightOnlyMathematicalAudit:
    glm_hidden: int
    mamba_hidden: int
    maximal_linear_rank: int
    minimum_nullity: int
    invertible_global_linear_adapter_possible: bool
    unchanged_glm_pathway_retained: bool
    zero_gate_exact_baseline_by_construction: bool
    mamba_donor_native_input_distribution_guaranteed: bool
    mamba_donor_native_interblock_connectivity_guaranteed: bool
    nonzero_branch_equivalence_proven: bool
    weight_only_replacement_certified: bool
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prove_weight_only_limits(
    glm_hidden: int = 6144,
    mamba_hidden: int = 2048,
) -> WeightOnlyMathematicalAudit:
    """Produce a structural proof; independent of trained parameter values."""
    if not isinstance(glm_hidden, int) or not isinstance(mamba_hidden, int):
        raise ValueError("hidden dimensions must be integers")
    if glm_hidden < 1 or mamba_hidden < 1:
        raise ValueError("hidden dimensions must be positive")
    maximal_rank = min(glm_hidden, mamba_hidden)
    minimum_nullity = glm_hidden - maximal_rank
    return WeightOnlyMathematicalAudit(
        glm_hidden=glm_hidden,
        mamba_hidden=mamba_hidden,
        maximal_linear_rank=maximal_rank,
        minimum_nullity=minimum_nullity,
        invertible_global_linear_adapter_possible=minimum_nullity == 0,
        unchanged_glm_pathway_retained=True,
        zero_gate_exact_baseline_by_construction=True,
        mamba_donor_native_input_distribution_guaranteed=False,
        mamba_donor_native_interblock_connectivity_guaranteed=False,
        nonzero_branch_equivalence_proven=False,
        weight_only_replacement_certified=False,
        conclusion=(
            "An R^%d -> R^%d adapter has nullity at least %d. "
            "The original GLM path is exactly retained when gates are zero; "
            "however, turning on the donor Mamba blocks introduces new "
            "transformations with unaligned hidden-state bases and broken "
            "pretrained donor-to-donor block connectivity. Weight copying "
            "alone does not certify an equivalent or beneficial recurrent "
            "replacement; an independent construction, calibration or "
            "measured training is required."
            % (glm_hidden, mamba_hidden, minimum_nullity)
        ),
    )


@dataclass(frozen=True)
class WeightOnlyProbe:
    exact_on_tested_inputs: bool
    max_abs_logit_change: float
    mean_abs_logit_change: float
    top1_agreement: float
    kl_teacher_to_candidate: float
    prompts_tested: int
    active_nonzero_branches: int
    global_equivalence_proven: bool = False
    weight_only_replacement_certified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def probe_untrained_hybrid(
    teacher: nn.Module,
    hybrid: GLMFirstMambaHybrid,
    input_ids: torch.Tensor,
) -> WeightOnlyProbe:
    """Read-only same-GLM-token-space comparison, no gradients or training.

    Uses non-cached, unpadded full-prefix forward; inputs must already be
    produced by the **pinned GLM** tokenizer. If the Mamba kernel/runtime is
    unavailable this fails rather than claiming success.
    """
    if hybrid.glm_model is not teacher:
        raise GLMFirstIntegrationError("candidate must wrap the identical GLM teacher")
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise GLMFirstIntegrationError("probe expects nonempty GLM [batch,seq] IDs")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise GLMFirstIntegrationError("probe token IDs must be integers")
    if (bool((input_ids < 0).any())
            or bool((input_ids >= hybrid.contract.glm_vocab).any())):
        raise GLMFirstIntegrationError("probe token ID outside GLM vocabulary")
    if teacher.training or hybrid.training:
        raise GLMFirstIntegrationError("probe models must be in eval() mode")

    active_nonzero = sum(
        int(b.enabled and bool(torch.count_nonzero(b.gate.detach())))
        for b in hybrid.branches
    )
    teacher_out = teacher(input_ids=input_ids, use_cache=False)
    candidate_out = hybrid(input_ids=input_ids, use_cache=False)

    teacher_logits = getattr(teacher_out, "logits", None)
    student_logits = getattr(candidate_out, "logits", None)
    if not isinstance(teacher_logits, torch.Tensor) or not isinstance(student_logits, torch.Tensor):
        raise GLMFirstIntegrationError("probe requires teacher and hybrid logits")
    if teacher_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise GLMFirstIntegrationError("teacher/hybrid logits have incompatible shapes")
    if teacher_logits.shape[-1] != hybrid.contract.glm_vocab:
        raise GLMFirstIntegrationError("probe tokenizer/output-head vocabulary mismatch")
    if not bool(torch.isfinite(teacher_logits).all()) or not bool(torch.isfinite(student_logits).all()):
        raise GLMFirstIntegrationError("teacher/hybrid logits contain nonfinite values")

    a = teacher_logits.float()
    b = student_logits.float()
    diff = (a - b).abs()
    # Calculate in fp32; zero-gate KL can have tiny floating-point roundoff.
    kl = F.kl_div(
        F.log_softmax(b, dim=-1),
        F.softmax(a, dim=-1),
        reduction="none",
    ).sum(-1).mean().clamp_min(0)
    return WeightOnlyProbe(
        exact_on_tested_inputs=bool(torch.equal(teacher_logits, student_logits)),
        max_abs_logit_change=float(diff.max()),
        mean_abs_logit_change=float(diff.mean()),
        top1_agreement=float(
            (teacher_logits.argmax(-1) == student_logits.argmax(-1)).float().mean()
        ),
        kl_teacher_to_candidate=float(kl),
        prompts_tested=int(input_ids.shape[0]),
        active_nonzero_branches=active_nonzero,
    )
