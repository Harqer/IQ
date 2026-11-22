"""Experimental CPU-only packed INT4 projection; no architecture changes.

Groupwise symmetric INT4, FP32 scales, exact zero-column removal.
Reference implementation: bounded row-tile dequantization, NOT an optimized kernel.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PackedInt4Linear(nn.Module):
    """Shape-preserving replacement for an inference-only nn.Linear.

    The original dense weight is not retained. `packed`, `scales`, `indices`,
    and optional `bias` are stored in persistent buffers for state_dict use.
    """
    def __init__(self, in_features: int, out_features: int, group_size: int,
                 packed: torch.Tensor, scales: torch.Tensor,
                 indices: torch.Tensor, bias: torch.Tensor | None = None,
                 tile_rows: int = 64) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.tile_rows = int(tile_rows)
        if self.group_size <= 0 or self.group_size % 2 != 0:
            raise ValueError('group_size must be positive and even')
        if self.tile_rows <= 0:
            raise ValueError('tile_rows must be positive')
        self.register_buffer('packed', packed.contiguous())
        self.register_buffer('scales', scales.contiguous())
        self.register_buffer('indices', indices.contiguous())
        self.register_buffer('bias', bias.contiguous() if bias is not None else None)

    @property
    def active_features(self) -> int:
        return int(self.indices.numel())

    @classmethod
    @torch.no_grad()
    def from_weight(cls, weight: torch.Tensor, *, bias: torch.Tensor | None = None,
                    group_size: int = 32, tile_rows: int = 64):
        if weight.device.type != 'cpu' or weight.ndim != 2:
            raise ValueError('Expected a 2D CPU weight tensor')
        rows, columns = weight.shape
        if bias is not None and tuple(bias.shape) != (rows,):
            raise ValueError('Bias dimensions do not match weight')
        if group_size <= 0 or group_size % 2:
            raise ValueError('group_size must be positive and even')
        # Lossless structural zero-column detection. Retain input positions.
        active = torch.zeros((columns,), dtype=torch.bool)
        for start in range(0, rows, tile_rows):
            block = weight[start:start + tile_rows]
            if not bool(torch.isfinite(block).all()):
                raise ValueError('Nonfinite source weights')
            active |= torch.any(block != 0, dim=0)
        indices = torch.nonzero(active).flatten().to(torch.int32)
        n = indices.numel()
        groups = (n + group_size - 1) // group_size
        padded = groups * group_size
        packed = torch.empty((rows, padded // 2), dtype=torch.uint8)
        scales = torch.empty((rows, groups), dtype=torch.float32)
        if n == 0:
            packed.zero_()
            scales.zero_()
        else:
            col_idx = indices.to(torch.long)
            for start in range(0, rows, tile_rows):
                stop = min(rows, start + tile_rows)
                block = weight[start:stop].index_select(1, col_idx).float()
                if padded != n:
                    block = F.pad(block, (0, padded - n))
                grouped = block.reshape(stop - start, groups, group_size)
                maxabs = grouped.abs().amax(dim=-1)
                scale = maxabs / 7.0
                safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                q = (grouped / safe.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int16)
                q = (q + 8).reshape(stop - start, padded)
                packed[start:stop] = (q[:, ::2] | (q[:, 1::2] << 4)).to(torch.uint8)
                scales[start:stop] = scale
        return cls(columns, rows, group_size, packed, scales,
                   indices, bias.detach().clone() if bias is not None else None,
                   tile_rows=tile_rows)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != 'cpu' or x.shape[-1] != self.in_features:
            raise ValueError('Expected CPU input with original in_features')
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        # Preserve the original input dimension at the module boundary.
        if self.active_features == 0:
            result = flat.new_zeros((flat.shape[0], self.out_features))
            if self.bias is not None:
                result = result + self.bias.to(result.dtype)
            return result.reshape(*shape, self.out_features)
        active_x = flat.index_select(-1, self.indices.to(torch.long)).float()
        n = self.active_features
        padded = self.packed.shape[1] * 2
        out = torch.empty((active_x.shape[0], self.out_features),
                          dtype=torch.float32, device='cpu')
        for start in range(0, self.out_features, self.tile_rows):
            stop = min(start + self.tile_rows, self.out_features)
            p = self.packed[start:stop]
            # Low nibble is earlier input element.
            q = torch.stack((p & 15, (p >> 4) & 15), dim=-1)
            q = q.reshape(stop - start, padded).to(torch.float32) - 8.0
            expanded_scales = self.scales[start:stop].repeat_interleave(
                self.group_size, dim=-1)
            dense_tile = (q * expanded_scales)[:, :n]
            out[:, start:stop] = F.linear(active_x, dense_tile)
        if self.bias is not None:
            out += self.bias.float()
        return out.to(dtype=x.dtype).reshape(*shape, self.out_features)

    @property
    def packed_nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales, self.indices)
                   if t is not None) + (0 if self.bias is None else self.bias.numel() * self.bias.element_size())
