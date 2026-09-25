from __future__ import annotations

import torch
from torch import nn


class SiTUAndMul(nn.Module):
    """Sigmoid-Tanh Unit GLU used by Kimi K3.

    gate = beta * tanh(g / beta) * sigmoid(g)
    linear = linear_beta * tanh(u / linear_beta)
    output = gate * linear
    """

    def __init__(
        self,
        *,
        beta: float = 4.0,
        linear_beta: float | None = 25.0,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        if linear_beta is not None and linear_beta <= 0:
            raise ValueError("linear_beta must be positive when configured")
        self.beta = float(beta)
        self.linear_beta = (
            float(linear_beta) if linear_beta is not None else None
        )

    def forward(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError("SiTU gate and up tensors must have identical shapes")
        dtype = gate.dtype
        gate_f = gate.float()
        up_f = up.float()
        bounded_gate = (
            self.beta
            * torch.tanh(gate_f / self.beta)
            * torch.sigmoid(gate_f)
        )
        if self.linear_beta is not None:
            up_f = (
                self.linear_beta
                * torch.tanh(up_f / self.linear_beta)
            )
        return (bounded_gate * up_f).to(dtype)
