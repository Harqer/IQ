"""MOHAWK Stage-2 training adapter for the existing IQ trainer.

This wraps the *actual* heterogeneous IQ student; no alternate SSM/mixer is
introduced. A dataset builder must first align GLM source states by UTF-8 byte
spans to IQ token positions, project them into IQ's hidden basis, and record
the alignment/calibration provenance. The adapter refuses missing targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from iq_model import HybridLayerType
from iq_transfer.mohawk import hidden_alignment_loss


class MohawkTrainingError(RuntimeError):
    pass


@dataclass
class MohawkStage2Output:
    loss: Tensor
    alignment_loss: Tensor


def _schedule_layers(student: nn.Module) -> tuple[HybridLayerType, ...]:
    config = getattr(student, "config", None)
    schedule = getattr(config, "schedule", None)
    layers = getattr(schedule, "layers", None)
    if not isinstance(layers, tuple) or not layers:
        raise MohawkTrainingError("student must expose an IQ HybridSchedule")
    if not hasattr(student, "layers") or len(student.layers) != len(layers):
        raise MohawkTrainingError("student schedule and physical layers do not match")
    return layers


def freeze_for_glm53_stage2(
    student: nn.Module,
    *,
    target_physical_layer: int,
    train_preceding_mamba: bool = False,
) -> tuple[int, ...]:
    """Freeze Mamba's lexical foundation; train the selected GLM context/MoE pair.

    If train_preceding_mamba is explicitly enabled, permit updates to the
    adjacent Mamba-3 MIMO block as a separate, measurable ablation.
    """
    layers = _schedule_layers(student)
    if not 1 <= target_physical_layer < len(layers):
        raise MohawkTrainingError("target_physical_layer is outside the IQ schedule")
    if (
        layers[target_physical_layer] is not HybridLayerType.MOE
        or layers[target_physical_layer - 1]
        not in {HybridLayerType.CSA, HybridLayerType.HCA}
    ):
        raise MohawkTrainingError("Stage 2 anchor must end in a CSA/HCA + MoE pair")
    selected = [target_physical_layer - 1, target_physical_layer]
    if train_preceding_mamba:
        mamba = target_physical_layer - 2
        if mamba < 0 or layers[mamba] is not HybridLayerType.MAMBA3:
            raise MohawkTrainingError("there is no adjacent Mamba-3 MIMO block to train")
        selected.insert(0, mamba)

    student.requires_grad_(False)
    for index in selected:
        student.layers[index].requires_grad_(True)
    if not any(p.requires_grad for p in student.parameters()):
        raise MohawkTrainingError("selected stages have no trainable parameters")
    return tuple(selected)


class IQGLM53MohawkStage2(nn.Module):
    """Train IQ physical residual states against genuinely aligned GLM states.

    The target corresponds to GLM's post-MLP residual output projected into the
    IQ target frame, at the source layer mapped to this MoE physical layer.
    No GLM token IDs or raw GLM logits are passed through the IQ tokenizer.
    """

    def __init__(
        self,
        student: nn.Module,
        *,
        source_glm_layer: int,
        target_physical_layer: int,
        alignment_manifest_sha256: str,
        train_preceding_mamba: bool = False,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if source_glm_layer < 0:
            raise MohawkTrainingError("source_glm_layer must be non-negative")
        if len(alignment_manifest_sha256) != 64 or any(
            digit not in "0123456789abcdef" for digit in alignment_manifest_sha256.lower()
        ):
            raise MohawkTrainingError("a real alignment manifest SHA-256 is required")
        if not math.isfinite(loss_weight) or loss_weight <= 0:
            raise MohawkTrainingError("loss_weight must be finite and positive")
        self.student = student
        self.source_glm_layer = int(source_glm_layer)
        self.target_physical_layer = int(target_physical_layer)
        self.alignment_manifest_sha256 = alignment_manifest_sha256.lower()
        self.loss_weight = float(loss_weight)
        self.trainable_layers = freeze_for_glm53_stage2(
            student,
            target_physical_layer=target_physical_layer,
            train_preceding_mamba=train_preceding_mamba,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        teacher_aligned_hidden: Tensor,
        teacher_alignment_mask: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        document_ids: Tensor | None = None,
        reasoning_context_lengths: Tensor | None = None,
    ) -> MohawkStage2Output:
        if not isinstance(teacher_aligned_hidden, Tensor):
            raise MohawkTrainingError("byte-aligned GLM teacher states are required")
        if input_ids.ndim != 2 or teacher_aligned_hidden.ndim != 3:
            raise MohawkTrainingError("input_ids [B,T] and teacher_aligned_hidden [B,T,H] are required")
        if tuple(teacher_aligned_hidden.shape[:2]) != tuple(input_ids.shape):
            raise MohawkTrainingError("GLM targets are not aligned to IQ token positions")
        if teacher_aligned_hidden.device != input_ids.device:
            raise MohawkTrainingError("IQ input and GLM targets must share a device")
        if not teacher_aligned_hidden.is_floating_point():
            raise MohawkTrainingError("GLM aligned targets must be floating-point")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise MohawkTrainingError("attention_mask must match input_ids")
            mask = attention_mask.to(dtype=torch.bool)
        else:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        if (
            not isinstance(teacher_alignment_mask, Tensor)
            or teacher_alignment_mask.dtype != torch.bool
            or teacher_alignment_mask.shape != input_ids.shape
        ):
            raise MohawkTrainingError("explicit causal teacher_alignment_mask is required")
        mask = mask & teacher_alignment_mask
        if not bool(mask.any()):
            raise MohawkTrainingError("no valid causally aligned tokens for hidden-state alignment")
        if labels is not None and labels.shape != input_ids.shape:
            raise MohawkTrainingError("labels must match input_ids shape")

        captures: list[Tensor] = []

        def capture(_layer: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            state = getattr(output, "hidden_states", output)
            if not isinstance(state, Tensor):
                raise MohawkTrainingError("selected IQ layer did not return hidden states")
            captures.append(state)

        handle = self.student.layers[self.target_physical_layer].register_forward_hook(capture)
        try:
            # Stage-2 represents a block alignment, not a language-modeling
            # objective. Labels are intentionally NOT forwarded to the student.
            self.student(
                input_ids,
                labels=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=document_ids,
                reasoning_context_lengths=reasoning_context_lengths,
            )
        finally:
            handle.remove()
        if len(captures) != 1:
            raise MohawkTrainingError("expected exactly one IQ physical-layer capture")
        student_hidden = captures[0]
        if student_hidden.shape != teacher_aligned_hidden.shape:
            raise MohawkTrainingError(
                f"teacher IQ-basis shape {tuple(teacher_aligned_hidden.shape)} "
                f"does not match student {tuple(student_hidden.shape)}"
            )
        alignment = hidden_alignment_loss(
            teacher_aligned_hidden.detach(),
            student_hidden,
            token_mask=mask,
        )
        return MohawkStage2Output(
            loss=self.loss_weight * alignment,
            alignment_loss=alignment,
        )
