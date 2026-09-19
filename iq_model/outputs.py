from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
