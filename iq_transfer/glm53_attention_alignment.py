"""Stagewise GLM MLA/DSA -> IQ CSA supervised attention alignment.

This trains actual IQ compressed attention parameters against captured official
GLM output on the *same input-token sequences*, with a pre-fit measured
6144->4096 coordinate map. It does not transplant GLM MLA weights 1:1 or
construct a pretrained IQ model. Source, train, and hold-out provenance are
validated before any parameter update.

Official teacher: transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaAttention
https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .transport import CoordinateMap


class GLM53AttentionAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class PairedAttentionBatch:
    """A single recorded and token-aligned source/target attention batch.

    Tensors are actual measurements, not a substitute for source activations.
    """
    iq_attention_input: torch.Tensor
    glm_attention_output: torch.Tensor
    token_ids: torch.Tensor
    dsa_indices: torch.Tensor

    def __post_init__(self) -> None:
        student = self.iq_attention_input
        teacher = self.glm_attention_output
        ids = self.token_ids
        indices = self.dsa_indices
        if student.ndim != 3 or teacher.ndim != 3:
            raise GLM53AttentionAlignmentError("paired hidden states must be [B,S,H]")
        if student.shape[:2] != teacher.shape[:2]:
            raise GLM53AttentionAlignmentError("teacher/student token axes differ")
        if not student.shape[0] or not student.shape[1]:
            raise GLM53AttentionAlignmentError("empty activation capture")
        if student.shape[-1] <= 0 or teacher.shape[-1] <= 0:
            raise GLM53AttentionAlignmentError("invalid hidden size")
        if ids.shape != student.shape[:2] or ids.dtype not in (torch.int32, torch.int64):
            raise GLM53AttentionAlignmentError("token_ids must be [B,S] int32/int64")
        if indices.ndim != 3 or indices.shape[:2] != ids.shape or indices.dtype not in (torch.int32, torch.int64):
            raise GLM53AttentionAlignmentError("DSA indices must be [B,S,K] integers")
        if indices.shape[-1] <= 0 or ((indices < 0) | (indices >= ids.shape[1])).any():
            raise GLM53AttentionAlignmentError("DSA index out of sequence bounds")
        if not student.is_floating_point() or not teacher.is_floating_point():
            raise GLM53AttentionAlignmentError("paired activations must be floating tensors")
        if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
            raise GLM53AttentionAlignmentError("paired activations contain nonfinite values")


def validate_pairing(
    train: PairedAttentionBatch,
    validation: PairedAttentionBatch,
    *,
    coordinate_map: CoordinateMap,
    student: nn.Module,
) -> None:
    if not hasattr(student, "config") or not hasattr(student.config, "hidden_size"):
        raise GLM53AttentionAlignmentError("student must have explicit attention config")
    shape = coordinate_map.matrix.shape
    if shape != (
        train.glm_attention_output.shape[-1],
        train.iq_attention_input.shape[-1],
    ):
        raise GLM53AttentionAlignmentError(
            "calibration map must transport GLM output width to IQ output width"
        )
    if validation.iq_attention_input.shape[-1] != train.iq_attention_input.shape[-1]:
        raise GLM53AttentionAlignmentError("hold-out IQ hidden width differs")
    if validation.glm_attention_output.shape[-1] != train.glm_attention_output.shape[-1]:
        raise GLM53AttentionAlignmentError("hold-out GLM hidden width differs")
    if student.config.hidden_size != train.iq_attention_input.shape[-1]:
        raise GLM53AttentionAlignmentError("student attention config does not match captured IQ width")
    if len(train.token_ids) == 0 or len(validation.token_ids) == 0:
        raise GLM53AttentionAlignmentError("no train/validation token sequences")
    # A held-out file containing the same token IDs as training is not
    # independent evidence. Use exact token sequence identity, not file name.
    train_rows = {tuple(row.tolist()) for row in train.token_ids.cpu()}
    val_rows = {tuple(row.tolist()) for row in validation.token_ids.cpu()}
    if train_rows.intersection(val_rows):
        raise GLM53AttentionAlignmentError(
            "training/held-out data share token sequences; leakage invalidates validation"
        )
    if coordinate_map.diagnostics is None or coordinate_map.diagnostics.validation_rmse is None:
        raise GLM53AttentionAlignmentError(
            "GLM->IQ coordinate map requires genuine held-out diagnostics"
        )
    if not math.isfinite(coordinate_map.diagnostics.validation_rmse):
        raise GLM53AttentionAlignmentError("calibration validation error is nonfinite")


def _teacher_target(
    data: PairedAttentionBatch, basis: torch.Tensor,
) -> torch.Tensor:
    return data.glm_attention_output.to(basis.device, torch.float32) @ basis


def evaluate_alignment(
    student: nn.Module, data: PairedAttentionBatch, *,
    basis: torch.Tensor,
) -> dict[str, float]:
    """Causal full-sequence IQ student vs mapped HF teacher output."""
    previous = student.training
    student.eval()
    try:
        with torch.no_grad():
            param = next(student.parameters())
            actual = student(data.iq_attention_input.to(
                device=param.device, dtype=param.dtype
            )).float()
            desired = _teacher_target(data, basis)
            if actual.shape != desired.shape:
                raise GLM53AttentionAlignmentError("IQ/GLM mapped output shapes differ")
            error = (actual - desired)
            mse = error.square().mean().item()
            baseline = desired.square().mean().item()
            relative = mse / max(baseline, 1e-12)
            cosine = torch.nn.functional.cosine_similarity(
                actual.reshape(-1, actual.shape[-1]),
                desired.reshape(-1, desired.shape[-1]), dim=-1,
                eps=1e-8,
            ).mean().item()
            if any(not math.isfinite(v) for v in (mse, relative, cosine)):
                raise GLM53AttentionAlignmentError("alignment metric is nonfinite")
            return {
                "mse": mse,
                "relative_mse": relative,
                "cosine": cosine,
            }
    finally:
        student.train(previous)


def evaluate_dsa_index_recall(
    student: nn.Module,
    data: PairedAttentionBatch,
) -> float:
    """Fraction of causal teacher top-k keys recovered by IQ's real indexer.

    Unlike plain output MSE, this exposes DSA selection disagreement. IQ and
    GLM may use different top-k counts; no exact-index parity is assumed.
    """
    if not hasattr(student, "indexer_scores"):
        raise GLM53AttentionAlignmentError(
            "IQ stage lacks an indexer_scores implementation"
        )
    device = next(student.parameters()).device
    x = data.iq_attention_input.to(device)
    previous = student.training
    student.eval()
    try:
        with torch.no_grad():
            segments = student.indexer_scores(x)
        if len(segments) != x.shape[0]:
            raise GLM53AttentionAlignmentError(
                "source/target index comparison supports only one unpadded segment per batch item"
            )
        recall_sum = 0.0
        count = 0
        for seg in segments:
            batch_index = seg.batch_index
            seq_positions = seg.token_indices
            if not torch.equal(
                seq_positions.cpu(),
                torch.arange(x.shape[1]),
            ):
                raise GLM53AttentionAlignmentError(
                    "packed/partial segments are not supported in GLM index comparison"
                )
            for q in range(x.shape[1]):
                teacher = {
                    int(v) for v in data.dsa_indices[batch_index, q].tolist()
                    if 0 <= int(v) <= q
                }
                student_keys = {
                    int(v) for v in seg.selected_indices[q].tolist()
                    if 0 <= int(v) <= q
                }
                if teacher:
                    recall_sum += len(teacher & student_keys) / len(teacher)
                    count += 1
        if not count:
            raise GLM53AttentionAlignmentError(
                "no causal GLM indexer selections to evaluate"
            )
        return recall_sum / count
    finally:
        student.train(previous)


def evaluate_causal_invariance(
    student: nn.Module,
    data: PairedAttentionBatch,
    *,
    tolerance: float = 1e-4,
) -> dict[str, float | bool]:
    """Measure whether changing future IQ tokens affects earlier attention.

    This does not establish packed-context/cache equivalence; it checks
    unpadded causal prefill under deliberately perturbed future activations.
    """
    if data.iq_attention_input.shape[1] < 2:
        raise GLM53AttentionAlignmentError("causal gate needs at least two tokens")
    if tolerance <= 0:
        raise GLM53AttentionAlignmentError("causal tolerance must be positive")
    parameter = next(student.parameters())
    input_data = data.iq_attention_input.to(
        device=parameter.device, dtype=parameter.dtype
    )
    previous = student.training
    student.eval()
    try:
        with torch.no_grad():
            original = student(input_data)
            discrepancy = 0.0
            for q in sorted({0, (input_data.shape[1] - 1) // 2}):
                modified = input_data.clone()
                modified[:, q + 1:] = modified[:, q + 1:] * -3.0 + 20.0
                after = student(modified)
                drift = (
                    original[:, :q + 1] - after[:, :q + 1]
                ).abs().max().item()
                discrepancy = max(discrepancy, drift)
        if not math.isfinite(discrepancy):
            raise GLM53AttentionAlignmentError("nonfinite causal leakage")
        return {
            "max_abs_future_leakage": discrepancy,
            "causal_pass": discrepancy <= tolerance,
        }
    finally:
        student.train(previous)


def fit_attention_from_paired_activations(
    student: nn.Module,
    train: PairedAttentionBatch,
    validation: PairedAttentionBatch,
    coordinate_map: CoordinateMap,
    *,
    steps: int,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> dict[str, object]:
    """Perform actual gradient descent; retain *best held-out* parameters.

    No arbitrary quality threshold is treated as pretrained GLM equivalence.
    """
    if type(steps) is not int or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise GLM53AttentionAlignmentError("positive training steps and learning rate required")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise GLM53AttentionAlignmentError("max_grad_norm must be positive")
    validate_pairing(train, validation, coordinate_map=coordinate_map, student=student)
    parameters = list(student.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise GLM53AttentionAlignmentError("IQ attention must be trainable")
    device = parameters[0].device
    dtype = parameters[0].dtype
    basis = torch.as_tensor(
        coordinate_map.matrix, dtype=torch.float32, device=device
    )
    train_input = train.iq_attention_input.to(device=device, dtype=dtype)
    train_target = _teacher_target(train, basis).detach()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    before = evaluate_alignment(student, validation, basis=basis)
    best = before
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in student.state_dict().items()
    }
    student.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = student(train_input).float()
        if output.shape != train_target.shape:
            raise GLM53AttentionAlignmentError("IQ attention output width mismatch")
        loss = torch.nn.functional.mse_loss(output, train_target)
        if not bool(torch.isfinite(loss)):
            raise GLM53AttentionAlignmentError("nonfinite distillation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        current = evaluate_alignment(student, validation, basis=basis)
        if current["mse"] < best["mse"]:
            best = current
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in student.state_dict().items()
            }
    # Revert a worse student rather than declaring an overfit as progress.
    student.load_state_dict(best_state, strict=True)
    student.eval()
    causal = evaluate_causal_invariance(student, validation)
    recall = evaluate_dsa_index_recall(student, validation)
    return {
        "heldout_dsa_topk_recall": recall,
        "heldout_student_causal": causal,
        "stage_training_steps": steps,
        "initial_holdout": before,
        "best_holdout": best,
        "holdout_improved": best["mse"] < before["mse"],
        "training_objective": "mapped_GLM_MLA_DSA_attention_output_MSE",
        "source_operator_equivalence": False,
        "full_model_pretrained": False,
    }
