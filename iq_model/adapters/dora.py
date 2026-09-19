from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """DoRA correction around an existing bias-free linear operator."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        *,
        alpha: float | None = None,
        dropout: float = 0.0,
        freeze_base: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("DoRA rank must be positive")
        if base.bias is not None:
            raise ValueError("DoRALinear currently requires a bias-free base linear layer")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("DoRA dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.eps = float(eps)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        with torch.no_grad():
            magnitude = base.weight.float().norm(dim=1).clamp_min(self.eps)
        self.magnitude = nn.Parameter(magnitude.to(dtype=base.weight.dtype, device=base.weight.device))
        if freeze_base:
            self.base.weight.requires_grad_(False)

    def direction_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base.weight + delta.to(dtype=self.base.weight.dtype)

    def effective_weight(self) -> torch.Tensor:
        direction = self.direction_weight()
        norm = direction.float().norm(dim=1, keepdim=True).clamp_min(self.eps)
        unit = direction.float() / norm
        return (unit * self.magnitude.float().unsqueeze(1)).to(direction.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout.p > 0:
            base_out = F.linear(x, self.base.weight)
            low_rank = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
            direction_out = base_out + low_rank
            direction = self.direction_weight()
            norm = direction.float().norm(dim=1).clamp_min(self.eps).to(direction.dtype)
            scale = (self.magnitude / norm).view(*([1] * (direction_out.ndim - 1)), -1)
            return direction_out * scale
        return F.linear(x, self.effective_weight())

    @torch.no_grad()
    def merge_into_base_(self) -> nn.Linear:
        self.base.weight.copy_(self.effective_weight())
        self.base.weight.requires_grad_(True)
        return self.base


def _resolve_parent_module(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid module path: {path!r}")
    current: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit():
            current = current[int(part)]
        else:
            child = getattr(current, part, None)
            if not isinstance(child, nn.Module):
                raise ValueError(f"module path {path!r} cannot resolve {part!r}")
            current = child
    return current, parts[-1]


def install_dora(
    model: nn.Module,
    module_paths: list[str] | tuple[str, ...],
    *,
    rank: int,
    alpha: float | None = None,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> tuple[str, ...]:
    if not module_paths:
        raise ValueError("at least one module path is required")
    if len(module_paths) != len(set(module_paths)):
        raise ValueError("DoRA module paths must be unique")

    staged: list[tuple[nn.Module, str, DoRALinear]] = []
    for path in module_paths:
        parent, name = _resolve_parent_module(model, path)
        child = getattr(parent, name, None)
        if not isinstance(child, nn.Linear):
            raise ValueError(f"DoRA target is not nn.Linear: {path!r}")
        staged.append(
            (
                parent,
                name,
                DoRALinear(
                    child,
                    rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                ),
            )
        )
    for parent, name, replacement in staged:
        setattr(parent, name, replacement)
    return tuple(module_paths)
