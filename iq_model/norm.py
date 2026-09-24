from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size <= 0 or eps <= 0:
            raise ValueError("hidden_size and eps must be positive")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class HeadRMSNorm(nn.Module):
    """RMS normalization over the per-head feature dimension.

    The same learned scale is shared across tokens and attention heads. This is
    the normalization needed by IQ's Q/K and compressed-KV attention paths.
    """

    def __init__(self, head_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if head_dim <= 0 or eps <= 0:
            raise ValueError("head_dim and eps must be positive")
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.numel():
            raise ValueError(
                f"head RMSNorm expected final dimension {self.weight.numel()}, got {x.shape[-1]}"
            )
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class UnweightedRMSNorm(nn.Module):
    """RMS normalization without a learned scale, used by V4 query heads."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(input_dtype)
