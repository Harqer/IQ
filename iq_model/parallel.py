"""Parallel-branch fusion for IQ's Falcon-H1-inspired hybrid block.

The operator is a real concatenation/output projection, not a weighted average.
It is initialized to an exact Mamba-only residual branch to protect the
pretrained Mamba-3 MIMO function during initial GLM/NVIDIA attention transfer.
"""
from __future__ import annotations

import torch
from torch import nn


class ParallelFusionError(ValueError):
    pass


class ParallelBranchFusion(nn.Module):
    """Fuse Mamba-3 and attention outputs in the same hidden coordinate system.

    Given Mamba delta m and attention delta a, computes W@[m;a]. At construction
    W=[I,0] so existing pretrained Mamba behavior is preserved exactly before
    the newly inserted attention pathway is trained. The zero attention columns
    are trainable: gradients update W first, then attention in later steps.
    """

    def __init__(self, hidden_size: int, *, device=None, dtype=None) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ParallelFusionError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.projection = nn.Linear(
            2 * hidden_size, hidden_size, bias=False, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[:, :hidden_size].copy_(
                torch.eye(hidden_size, device=self.projection.weight.device,
                          dtype=self.projection.weight.dtype)
            )

    def forward(self, mamba_delta: torch.Tensor, attention_delta: torch.Tensor) -> torch.Tensor:
        if mamba_delta.shape != attention_delta.shape:
            raise ParallelFusionError(
                f"parallel branches must share shape: {mamba_delta.shape} != {attention_delta.shape}"
            )
        if mamba_delta.ndim != 3 or mamba_delta.shape[-1] != self.hidden_size:
            raise ParallelFusionError(
                "parallel branches must be [batch, tokens, hidden_size]"
            )
        return self.projection(torch.cat((mamba_delta, attention_delta), dim=-1))



class ExpertResidualGate(nn.Module):
    """Keep an uncalibrated expert residual neutral until weight alignment.

    Unlike bypassing the expert, its residual update is still computed, so the
    gate receives a useful gradient. Downstream training must decide when to
    open the gate; zero does not make random expert weights pretrained.
    """

    def __init__(self, *, device=None, dtype=None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    def forward(self, residual_input: torch.Tensor, expert_output: torch.Tensor) -> torch.Tensor:
        if residual_input.shape != expert_output.shape:
            raise ParallelFusionError("expert output and residual input must match")
        return residual_input + self.scale * (expert_output - residual_input)
