from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import UnweightedRMSNorm


class MHCError(RuntimeError):
    pass


@dataclass(frozen=True)
class MHCConfig:
    hidden_size: int
    streams: int = 4
    sinkhorn_iters: int = 20
    eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.streams <= 1:
            raise ValueError("mHC requires streams > 1")
        if self.sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")
        if self.eps <= 0 or self.rms_norm_eps <= 0 or self.initializer_range <= 0:
            raise ValueError("mHC eps/norm/init values must be positive")


@dataclass(frozen=True)
class MHCWeights:
    post: torch.Tensor
    comb: torch.Tensor
    collapsed: torch.Tensor


def sinkhorn_doubly_stochastic(
    matrix: torch.Tensor,
    *,
    iterations: int,
    eps: float,
) -> torch.Tensor:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("Sinkhorn input must end in a square matrix")
    if iterations <= 0 or eps <= 0:
        raise ValueError("iterations and eps must be positive")
    result = matrix.float()
    result = result / (result.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        result = result / (result.sum(dim=-1, keepdim=True) + eps)
        result = result / (result.sum(dim=-2, keepdim=True) + eps)
    return result


class ManifoldHyperConnection(nn.Module):
    """DeepSeek-V4-style manifold-constrained Hyper-Connection reference."""

    def __init__(self, config: MHCConfig) -> None:
        super().__init__()
        self.config = config
        n = config.streams
        weight_count = (2 + n) * n
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.fn = nn.Parameter(torch.empty(weight_count, n * config.hidden_size))
        self.base = nn.Parameter(torch.empty(weight_count))
        self.scale = nn.Parameter(torch.empty(3))
        nn.init.normal_(
            self.fn,
            mean=0.0,
            std=config.initializer_range,
        )
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, hidden_streams: torch.Tensor) -> MHCWeights:
        if hidden_streams.ndim != 4:
            raise ValueError(
                "mHC hidden_streams must have shape [batch, sequence, streams, hidden]"
            )
        if hidden_streams.shape[-2:] != (
            self.config.streams,
            self.config.hidden_size,
        ):
            raise ValueError(
                "mHC hidden_streams final dimensions do not match config"
            )

        batch, sequence = hidden_streams.shape[:2]
        n = self.config.streams
        flattened = hidden_streams.reshape(
            batch,
            sequence,
            n * self.config.hidden_size,
        ).float()
        flattened = self.input_norm(flattened)
        coefficients = F.linear(flattened, self.fn.float())

        pre_w, post_w, comb_w = coefficients.split(
            [n, n, n * n],
            dim=-1,
        )
        pre_b, post_b, comb_b = self.base.float().split(
            [n, n, n * n]
        )
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        comb_w = comb_w.view(batch, sequence, n, n)
        comb_b = comb_b.view(n, n)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.config.eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(
            comb_w * comb_scale + comb_b,
            dim=-1,
        ) + self.config.eps
        comb = sinkhorn_doubly_stochastic(
            comb,
            iterations=self.config.sinkhorn_iters,
            eps=self.config.eps,
        )

        collapsed = (
            pre.unsqueeze(-1) * hidden_streams.float()
        ).sum(dim=2).to(hidden_streams.dtype)
        return MHCWeights(
            post=post,
            comb=comb,
            collapsed=collapsed,
        )

    def merge(
        self,
        hidden_streams: torch.Tensor,
        block_output: torch.Tensor,
        weights: MHCWeights,
    ) -> torch.Tensor:
        if block_output.shape != hidden_streams.shape[:2] + (
            self.config.hidden_size,
        ):
            raise ValueError("mHC block_output shape mismatch")
        dtype = hidden_streams.dtype
        expanded = (
            weights.post.to(dtype).unsqueeze(-1)
            * block_output.unsqueeze(-2)
        )
        residual = torch.matmul(
            weights.comb.to(dtype).transpose(-1, -2),
            hidden_streams,
        )
        result = expanded + residual
        if not bool(torch.isfinite(result).all()):
            raise MHCError("mHC produced non-finite streams")
        return result


class MHCHead(nn.Module):
    """Final collapse from mHC streams back to one residual stream."""

    def __init__(self, config: MHCConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.fn = nn.Parameter(
            torch.empty(
                config.streams,
                config.streams * config.hidden_size,
            )
        )
        self.base = nn.Parameter(torch.empty(config.streams))
        self.scale = nn.Parameter(torch.empty(1))
        nn.init.normal_(
            self.fn,
            mean=0.0,
            std=config.initializer_range,
        )
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        if hidden_streams.ndim != 4:
            raise ValueError(
                "mHC head input must have shape [batch, sequence, streams, hidden]"
            )
        if hidden_streams.shape[-2:] != (
            self.config.streams,
            self.config.hidden_size,
        ):
            raise ValueError("mHC head input dimensions do not match config")
        flat = self.input_norm(hidden_streams.flatten(2).float())
        mixes = F.linear(flat, self.fn.float())
        pre = (
            torch.sigmoid(
                mixes * self.scale.float() + self.base.float()
            )
            + self.config.eps
        )
        return (
            pre.unsqueeze(-1) * hidden_streams.float()
        ).sum(dim=2).to(hidden_streams.dtype)


def expand_mhc_streams(
    hidden_states: torch.Tensor,
    *,
    streams: int,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, sequence, hidden]"
        )
    if streams <= 1:
        raise ValueError("streams must be > 1")
    return (
        hidden_states.unsqueeze(2)
        .expand(-1, -1, streams, -1)
        .contiguous()
    )
