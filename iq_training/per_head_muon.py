from __future__ import annotations

"""Independent Q/K/V-head Muon, following Kimi K3 §2.5.

Each attention projection weight remains one serialized parameter, while the
Nesterov momentum/update is orthogonalized independently for each logical head.
The momentum convention, quintic Newton-Schulz coefficients and learning-rate
adjustment match PyTorch's public Muon algorithm. Optimizer state belongs to
the original full projection tensor; no weight surgery or extra parameters.
"""

from dataclasses import dataclass
from math import sqrt

import torch


@dataclass(frozen=True)
class HeadLayout:
    heads: int
    head_dim: int

    def validate(self, matrix: torch.Tensor) -> None:
        if matrix.ndim != 2 or self.heads < 1 or self.head_dim < 1:
            raise ValueError("per-head Muon needs a 2D projection and positive head shape")
        if matrix.shape[0] != self.heads * self.head_dim:
            raise ValueError(
                "projection output width differs from num_heads * head_dim"
            )


@torch.no_grad()
def newton_schulz_zeropower(
    matrix: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """PyTorch Muon's BF16 quintic Newton-Schulz for each isolated head."""
    if matrix.ndim != 2 or steps <= 0 or steps >= 100 or eps <= 0:
        raise ValueError("invalid Newton-Schulz matrix, iterations, or epsilon")
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError("non-finite per-head Muon momentum")
    # Mirror torch.optim._muon._zeropower_via_newtonschulz rather than
    # silently substituting FP32. Kimi K3 changes the head *partition*, not
    # the underlying Muon momentum and orthogonalization semantics.
    x = matrix.detach().to(torch.bfloat16, copy=True)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x.div_(x.norm().clamp(min=eps))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, gram_update, x, beta=a)
    if transpose:
        x = x.T
    return x


def head_adjusted_learning_rate(
    lr: float, shape: tuple[int, int], adjust: str | None
) -> float:
    rows, columns = shape
    if adjust is None or adjust == "original":
        factor = sqrt(max(1.0, rows / columns))
    elif adjust == "match_rms_adamw":
        factor = 0.2 * sqrt(max(rows, columns))
    elif adjust == "spectral_unclamped":
        factor = sqrt(rows / columns)
    else:
        raise ValueError("unsupported Muon learning rate adjustment")
    return lr * factor


class PerHeadMuon(torch.optim.Optimizer):
    """Muon states and quintic NS updates, independently per Q/K/V head."""

    def __init__(
        self,
        layouts: dict[torch.nn.Parameter, HeadLayout],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        adjust_lr_fn: str | None,
    ) -> None:
        if not layouts:
            raise ValueError("PerHeadMuon needs at least one recognized Q/K/V projection")
        if lr <= 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("invalid PerHeadMuon learning rate, decay or momentum")
        if ns_steps <= 0 or ns_steps >= 100:
            raise ValueError("PerHeadMuon requires 1..99 NS iterations")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw", "spectral_unclamped"):
            raise ValueError("unsupported Muon learning rate adjustment")
        groups = []
        for parameter, layout in layouts.items():
            layout.validate(parameter)
            groups.append(
                {
                    "params": [parameter],
                    "heads": layout.heads,
                    "head_dim": layout.head_dim,
                }
            )
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(groups, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        # torch.optim.Optimizer.load_state_dict otherwise replaces our group
        # head metadata from the serialized payload without validating it.
        incoming = state_dict.get("param_groups", [])
        if len(incoming) != len(self.param_groups):
            raise ValueError("per-head Muon checkpoint parameter-group count mismatch")
        for saved, current in zip(incoming, self.param_groups, strict=True):
            for field in ("heads", "head_dim"):
                if int(saved.get(field, -1)) != int(current[field]):
                    raise ValueError(
                        f"per-head Muon checkpoint {field} layout changed"
                    )
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("PerHeadMuon does not support closures")
        # Preflight every parameter before any update or momentum mutation.
        for group in self.param_groups:
            param = group["params"][0]
            HeadLayout(int(group["heads"]), int(group["head_dim"])).validate(param)
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse or torch.is_complex(param):
                raise ValueError("per-head Muon requires real, dense gradients")
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("non-finite Q/K/V gradient")
        for group in self.param_groups:
            param = group["params"][0]
            grad = param.grad
            if grad is None:
                continue
            heads, width = int(group["heads"]), int(group["head_dim"])
            HeadLayout(heads, width).validate(param)
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buffer = state["momentum_buffer"]
            # Same momentum convention as torch.optim.Muon:
            # B <- lerp(B, g, 1-momentum), U <- lerp(g, B, momentum).
            buffer.lerp_(grad, 1.0 - group["momentum"])
            raw = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
            projected = raw.reshape(heads, width, param.shape[1])
            update_parts = []
            for part in projected.unbind(dim=0):
                update = newton_schulz_zeropower(part, steps=group["ns_steps"])
                adjusted = head_adjusted_learning_rate(
                    group["lr"],
                    (part.shape[0], part.shape[1]),
                    group["adjust_lr_fn"],
                )
                update_parts.append(update * adjusted)
            update = torch.stack(update_parts, dim=0).reshape_as(param)
            param.mul_(1.0 - group["lr"] * group["weight_decay"])
            param.add_(update.to(param.dtype), alpha=-1.0)
        return None
