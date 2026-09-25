from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AttnResError(RuntimeError):
    pass


@dataclass(frozen=True)
class BlockAttnResConfig:
    hidden_size: int
    block_size: int = 12
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.block_size <= 0:
            raise ValueError("hidden_size and block_size must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")


@dataclass
class BlockAttnResState:
    block_sources: torch.Tensor
    prefix_sum: torch.Tensor
    layer_index: int = 0


class AttentionResidualMixer(nn.Module):
    """Kimi-K3-style depth-wise residual retrieval.

    The learned vector scores RMS-normalized block sources plus the current
    within-block prefix sum, then softmax-selects a depth mixture.
    """

    def __init__(self, config: BlockAttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.score = nn.Parameter(torch.ones(config.hidden_size))

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_sources: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_sum.ndim != 3:
            raise ValueError(
                "prefix_sum must have shape [batch, sequence, hidden]"
            )
        if prefix_sum.shape[-1] != self.config.hidden_size:
            raise ValueError("prefix_sum hidden size mismatch")
        if block_sources.ndim != 4:
            raise ValueError(
                "block_sources must have shape [batch, sequence, blocks, hidden]"
            )
        if block_sources.shape[:2] != prefix_sum.shape[:2]:
            raise ValueError("block source batch/sequence mismatch")
        if block_sources.shape[-1] != self.config.hidden_size:
            raise ValueError("block source hidden size mismatch")

        values = torch.cat(
            [block_sources, prefix_sum.unsqueeze(2)],
            dim=2,
        )
        values_f = values.float()
        variance = values_f.pow(2).mean(
            dim=-1,
            keepdim=True,
        )
        keys = values_f * torch.rsqrt(
            variance + self.config.rms_norm_eps
        )
        scores = (
            keys * self.score.float()
        ).sum(dim=-1)
        probs = torch.softmax(scores, dim=-1).unsqueeze(-1)
        mixed = (probs * values_f).sum(dim=2)
        return mixed.to(prefix_sum.dtype)


class BlockAttentionResidual(nn.Module):
    """State machine for block Attention Residuals over IQ physical layers."""

    def __init__(self, config: BlockAttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.mixer = AttentionResidualMixer(config)

    def init_state(
        self,
        embeddings: torch.Tensor,
    ) -> BlockAttnResState:
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape [batch, sequence, hidden]"
            )
        empty = embeddings.new_zeros(
            embeddings.shape[0],
            embeddings.shape[1],
            0,
            embeddings.shape[2],
        )
        return BlockAttnResState(
            block_sources=empty,
            prefix_sum=embeddings,
            layer_index=0,
        )

    def read(
        self,
        state: BlockAttnResState,
    ) -> torch.Tensor:
        if state.block_sources.shape[2] == 0:
            return state.prefix_sum
        return self.mixer(
            state.prefix_sum,
            state.block_sources,
        )

    def advance(
        self,
        state: BlockAttnResState,
        layer_output: torch.Tensor,
    ) -> BlockAttnResState:
        if layer_output.shape != state.prefix_sum.shape:
            raise ValueError("AttnRes layer output shape mismatch")
        next_index = state.layer_index + 1
        prefix_sum = state.prefix_sum + layer_output
        block_sources = state.block_sources

        if next_index % self.config.block_size == 0:
            block_sources = torch.cat(
                [
                    block_sources,
                    prefix_sum.unsqueeze(2),
                ],
                dim=2,
            )
            prefix_sum = torch.zeros_like(prefix_sum)

        return BlockAttnResState(
            block_sources=block_sources,
            prefix_sum=prefix_sum,
            layer_index=next_index,
        )

    def finalize(
        self,
        state: BlockAttnResState,
    ) -> torch.Tensor:
        if state.block_sources.shape[2] == 0:
            return state.prefix_sum
        return self.mixer(
            state.prefix_sum,
            state.block_sources,
        )
