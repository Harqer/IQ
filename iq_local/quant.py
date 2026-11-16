"""Group-64 symmetric int4 quantization for cold expert shards."""

from __future__ import annotations

from typing import NamedTuple

import torch

GROUP = 64


class PackedExpert(NamedTuple):
    packed: torch.Tensor
    scales: torch.Tensor


def quantize_int4(tensor: torch.Tensor) -> PackedExpert:
    if tensor.ndim != 2:
        raise ValueError("quantize_int4 expects a 2-D tensor")
    rows, cols = tensor.shape
    if cols % GROUP != 0:
        raise ValueError(f"last dim {cols} not divisible by group {GROUP}")
    x = tensor.detach().to(torch.float32)
    grouped = x.reshape(rows, cols // GROUP, GROUP)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = (max_abs / 7.0).squeeze(-1)  # [rows, cols // GROUP]
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp_(-8, 7).to(torch.int8)
    low = (q[..., 0::2] & 0x0F).to(torch.uint8)
    high = (q[..., 1::2] & 0x0F).to(torch.uint8)
    packed = low | (high << 4)
    return PackedExpert(packed.reshape(rows, cols // 2), scales)


def dequantize_int4(
    packed: PackedExpert, out_shape: tuple[int, int], out_dtype: torch.dtype
) -> torch.Tensor:
    rows, cols = out_shape
    if cols % GROUP != 0:
        raise ValueError(f"last dim {cols} not divisible by group {GROUP}")
    packed3d = packed.packed.reshape(rows, cols // GROUP, GROUP // 2)
    low = (packed3d & 0x0F).to(torch.int8)
    high = ((packed3d >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    q = torch.stack((low, high), dim=-1).reshape(rows, cols // GROUP, GROUP)
    values = q.to(torch.float32) * packed.scales.to(torch.float32).unsqueeze(-1)
    return values.reshape(rows, cols).to(out_dtype)
