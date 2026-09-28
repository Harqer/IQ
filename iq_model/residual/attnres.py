from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AttnResError(RuntimeError):
    pass


@dataclass(frozen=True)
class BlockAttnResConfig:
    hidden_size: int
    num_layers: int
    block_size: int = 12
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0 or self.block_size <= 0:
            raise ValueError(
                "hidden_size, num_layers, and block_size must be positive"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")


@dataclass
class BlockAttnResState:
    block_sources: torch.Tensor
    prefix_sum: torch.Tensor
    prefix_active: bool
    layer_index: int = 0


class AttentionResidualMixer(nn.Module):
    """Learned depth-wise retrieval over completed blocks/current prefix."""

    def __init__(self, config: BlockAttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.score = nn.Parameter(torch.ones(config.hidden_size))

    def forward(
        self,
        block_sources: torch.Tensor,
        prefix_sum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_sources.ndim != 4:
            raise ValueError(
                "block_sources must have shape [batch, sequence, blocks, hidden]"
            )
        if block_sources.shape[-1] != self.config.hidden_size:
            raise ValueError("block source hidden size mismatch")
        if prefix_sum is not None:
            if prefix_sum.ndim != 3:
                raise ValueError(
                    "prefix_sum must have shape [batch, sequence, hidden]"
                )
            if block_sources.shape[:2] != prefix_sum.shape[:2]:
                raise ValueError("block source batch/sequence mismatch")
            if prefix_sum.shape[-1] != self.config.hidden_size:
                raise ValueError("prefix_sum hidden size mismatch")
            values = torch.cat(
                [block_sources, prefix_sum.unsqueeze(2)],
                dim=2,
            )
        else:
            values = block_sources

        if values.shape[2] == 0:
            raise AttnResError("AttnRes requires at least one depth source")

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
        return mixed.to(values.dtype)


class BlockAttentionResidual(nn.Module):
    """Kimi-K3-style Block Attention Residuals over IQ physical layers.

    The embedding is the first saved source. Layer deltas accumulate within a
    block. At each block boundary the completed prefix is saved and the next
    block starts with an empty prefix. Every physical layer has its own learned
    depth-retrieval vector; the final backbone collapse has another.
    """

    def __init__(self, config: BlockAttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.mixers = nn.ModuleList(
            AttentionResidualMixer(config)
            for _ in range(config.num_layers)
        )
        self.output_mixer = AttentionResidualMixer(config)

    def init_state(
        self,
        embeddings: torch.Tensor,
    ) -> BlockAttnResState:
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape [batch, sequence, hidden]"
            )
        if embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError("embedding hidden size mismatch")
        return BlockAttnResState(
            block_sources=embeddings.unsqueeze(2),
            prefix_sum=torch.zeros_like(embeddings),
            prefix_active=False,
            layer_index=0,
        )

    def read(
        self,
        state: BlockAttnResState,
    ) -> torch.Tensor:
        if state.layer_index >= self.config.num_layers:
            raise AttnResError(
                "cannot read a layer input after the configured layer count"
            )
        mixer = self.mixers[state.layer_index]
        return mixer(
            state.block_sources,
            state.prefix_sum if state.prefix_active else None,
        )

    def advance(
        self,
        state: BlockAttnResState,
        layer_delta: torch.Tensor,
    ) -> BlockAttnResState:
        if state.layer_index >= self.config.num_layers:
            raise AttnResError(
                "cannot advance beyond the configured layer count"
            )
        if layer_delta.shape != state.prefix_sum.shape:
            raise ValueError("AttnRes layer delta shape mismatch")

        prefix_sum = (
            state.prefix_sum + layer_delta
            if state.prefix_active
            else layer_delta
        )
        next_index = state.layer_index + 1
        block_sources = state.block_sources
        prefix_active = True

        if (
            next_index % self.config.block_size == 0
            and next_index < self.config.num_layers
        ):
            block_sources = torch.cat(
                [
                    block_sources,
                    prefix_sum.unsqueeze(2),
                ],
                dim=2,
            )
            prefix_sum = torch.zeros_like(prefix_sum)
            prefix_active = False

        return BlockAttnResState(
            block_sources=block_sources,
            prefix_sum=prefix_sum,
            prefix_active=prefix_active,
            layer_index=next_index,
        )

    def finalize(
        self,
        state: BlockAttnResState,
    ) -> torch.Tensor:
        if state.layer_index != self.config.num_layers:
            raise AttnResError(
                "finalize requires exactly num_layers advances"
            )
        return self.output_mixer(
            state.block_sources,
            state.prefix_sum if state.prefix_active else None,
        )
