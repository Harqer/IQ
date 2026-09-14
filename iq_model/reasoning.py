from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import IQBlock
from .mixers import MixerContext


@dataclass
class RecurrentCoreOutput:
    hidden_states: torch.Tensor
    pass_states: tuple[torch.Tensor, ...]


class RecurrentReasoningCore(nn.Module):
    """Depth-recurrent reasoning engine over a heterogeneous physical block stack.

    Sequence recurrence (for example Gated DeltaNet state) and depth recurrence are
    intentionally separate.  The same eight physical blocks are revisited across
    passes, while each block keeps its own sequence-mixer family.
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

    @property
    def mixer_schedule(self) -> tuple[str, ...]:
        return tuple(block.mixer_kind for block in self.blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        context: MixerContext,
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
                candidate = block(before, context=context)
                # delta_scale=1 preserves the physical block function. Recurrent
                # residual stabilization remains an explicit later ablation.
                hidden_states = before + self.delta_scale * (candidate - before)

            if capture_passes:
                pass_states.append(hidden_states)

        return RecurrentCoreOutput(hidden_states, tuple(pass_states))
