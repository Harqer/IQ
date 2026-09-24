from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("rotary dim must be a positive even integer")
        if max_position_embeddings <= 0 or base <= 0:
            raise ValueError("max_position_embeddings and base must be positive")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.dim = int(dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        if position_ids.numel() and (
            int(position_ids.min()) < 0
            or int(position_ids.max()) >= self.max_position_embeddings
        ):
            raise ValueError("position_ids exceed configured maximum")
        freqs = (
            position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1)
            * self.inv_freq.to(device=device)
        )
        angles = torch.cat((freqs, freqs), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("rotary slice must have an even final dimension")
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rotate_interleaved_pairs(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("interleaved rotary slice must have an even final dimension")
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    *,
    rotary_at_end: bool = False,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE to one attention tensor.

    Dense Transformer anchors rotate the leading rotary dimensions.
    CSA/HCA rotate the trailing partial-RoPE dimensions and can apply the
    inverse rotation to the attention output.
    """

    if x.ndim != 4:
        raise ValueError("x must have shape [batch, heads, sequence, head_dim]")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    if cos.ndim != 3 or sin.ndim != 3 or cos.shape != sin.shape:
        raise ValueError("cos and sin must have matching [batch, sequence, rotary_dim] shapes")
    if cos.shape[-1] != rotary_dim:
        raise ValueError("cos/sin final dimension must equal rotary_dim")
    if cos.shape[0] != x.shape[0] or cos.shape[1] != x.shape[2]:
        raise ValueError("cos/sin batch and sequence dimensions must match x")

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    if rotary_at_end:
        passthrough = x[..., :-rotary_dim]
        rotated = x[..., -rotary_dim:]
    else:
        rotated = x[..., :rotary_dim]
        passthrough = x[..., rotary_dim:]

    sign = -1.0 if inverse else 1.0
    rotate_fn = rotate_interleaved_pairs if rotary_at_end else rotate_half
    rotated = (rotated * cos) + (sign * rotate_fn(rotated) * sin)
    if rotary_at_end:
        return torch.cat((passthrough, rotated), dim=-1)
    return torch.cat((rotated, passthrough), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, sequence, head_dim]")
    return (
        apply_rotary_to_tensor(q, cos, sin, rotary_dim),
        apply_rotary_to_tensor(k, cos, sin, rotary_dim),
    )


def apply_partial_rotary_at_end(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    return apply_rotary_to_tensor(
        x,
        cos,
        sin,
        rotary_dim,
        rotary_at_end=True,
    )


def apply_inverse_partial_rotary_at_end(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    return apply_rotary_to_tensor(
        x,
        cos,
        sin,
        rotary_dim,
        rotary_at_end=True,
        inverse=True,
    )
