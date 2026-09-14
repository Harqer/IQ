from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import IQBlock


@dataclass
class RecurrentCoreOutput:
    hidden_states: torch.Tensor
    pass_states: tuple[torch.Tensor, ...]


class RecurrentReasoningCore(nn.Module):
    """Fixed-depth recurrent reasoning engine for IQ v0.

    The same physical block stack is reused across multiple reasoning passes. v0 uses
    a fixed pass count on purpose; token-wise routing and learned halting are later
    experiments once recurrent stability and dense->loop transfer are established.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        *,
        hidden_size: int,
        passes: int,
        use_pass_embeddings: bool = True,
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("recurrent core requires at least one physical block")
        if passes <= 0:
            raise ValueError("passes must be positive")
        if delta_scale <= 0:
            raise ValueError("delta_scale must be positive")

        self.blocks = blocks
        self.passes = int(passes)
        self.delta_scale = float(delta_scale)

        if use_pass_embeddings:
            self.pass_embeddings = nn.Parameter(torch.zeros(self.passes, hidden_size))
        else:
            self.register_parameter("pass_embeddings", None)

    @property
    def physical_depth(self) -> int:
        return len(self.blocks)

    @property
    def effective_depth(self) -> int:
        return len(self.blocks) * self.passes

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        capture_passes: bool = False,
    ) -> RecurrentCoreOutput:
        pass_states: list[torch.Tensor] = []

        for pass_index in range(self.passes):
            if self.pass_embeddings is not None:
                hidden_states = hidden_states + self.pass_embeddings[pass_index].view(1, 1, -1)

            for block in self.blocks:
                if not isinstance(block, IQBlock):
                    raise TypeError(f"recurrent block must be IQBlock, got {type(block).__name__}")
                before = hidden_states
                candidate = block(
                    before,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                # delta_scale=1 is function-preserving for a transplanted block.
                # Alternative recurrence-aware scaling is an explicit ablation.
                hidden_states = before + self.delta_scale * (candidate - before)

            if capture_passes:
                pass_states.append(hidden_states)

        return RecurrentCoreOutput(hidden_states, tuple(pass_states))
