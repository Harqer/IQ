from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FutureLatentPredictor(nn.Module):
    """Predict a future/target representation from the current IQ representation.

    This is a JEPA-style architectural hook, not a token decoder. Target construction
    is intentionally owned by the training pipeline so experiments can compare
    future-segment, future-summary, recurrent-pass, and code-structure targets without
    changing the model backbone.
    """

    def __init__(self, hidden_size: int, predictor_dim: int = 0) -> None:
        super().__init__()
        inner = int(predictor_dim or hidden_size)
        self.hidden_size = int(hidden_size)
        self.predictor_dim = inner
        self.in_proj = nn.Linear(hidden_size, 2 * inner, bias=False)
        self.out_proj = nn.Linear(inner, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(hidden_states).chunk(2, dim=-1)
        prediction = self.out_proj(F.silu(gate) * value)
        return self.norm(prediction)
