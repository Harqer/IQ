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
        if position_ids.numel() and (int(position_ids.min()) < 0 or int(position_ids.max()) >= self.max_position_embeddings):
            raise ValueError("position_ids exceed configured maximum")
        freqs = position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1) * self.inv_freq.to(device=device)
        angles = torch.cat((freqs, freqs), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, sequence, head_dim]")
    if rotary_dim > q.shape[-1] or rotary_dim > k.shape[-1]:
        raise ValueError("rotary_dim exceeds attention head dimension")
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)
