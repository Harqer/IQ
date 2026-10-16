from __future__ import annotations

from math import ceil
from pathlib import Path

import torch

from .checkpoint import SafetensorsSource
from .donor import DonorError


class GLM53DequantizedSource:
    """Lazy FP8 E4M3/128x128 -> BF16 donor source for the GLM-5.3 checkpoint.

    Keeps the original checkpoint shards, index, and FP32 block scales intact.
    Only tensors requested by the transfer executor are materialized.
    """

    BLOCK = 128

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._raw = SafetensorsSource(checkpoint_dir)
        raw_keys = frozenset(self._raw.keys())
        self._keys = tuple(k for k in self._raw.keys() if not k.endswith(".weight_scale_inv"))
        self._scales: dict[str, str] = {}

        for key in raw_keys:
            if not key.endswith(".weight_scale_inv"):
                continue
            weight_key = key[: -len("_scale_inv")]
            if weight_key not in raw_keys:
                raise DonorError(f"orphan GLM-5.3 FP8 scale: {key}")
            weight_shape = self._raw.shape(weight_key)
            if len(weight_shape) != 2:
                raise DonorError(f"FP8 weight must be a matrix: {weight_key}: {weight_shape}")
            expected = tuple(ceil(dim / self.BLOCK) for dim in weight_shape)
            actual = self._raw.shape(key)
            if actual != expected:
                raise DonorError(
                    f"FP8 scale geometry mismatch: {key}: got {actual}, expected {expected} "
                    f"for {weight_key} {weight_shape}"
                )
            self._scales[weight_key] = key

    @property
    def quantized_weight_count(self) -> int:
        return len(self._scales)

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def shape(self, key: str) -> tuple[int, ...]:
        if key not in self._keys:
            raise DonorError(f"unknown GLM-5.3 parameter: {key}")
        return self._raw.shape(key)

    def checkpoint_files(self) -> tuple[Path, ...]:
        return self._raw.checkpoint_files()

    def get(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise DonorError(f"unknown GLM-5.3 parameter: {key}")
        weight = self._raw.get(key)
        scale_key = self._scales.get(key)
        if weight.dtype == torch.float8_e4m3fn:
            if scale_key is None:
                raise DonorError(f"FP8 tensor missing block scales: {key}")
            scale = self._raw.get(scale_key)
            if scale.dtype != torch.float32:
                raise DonorError(f"FP8 scales must be FP32: {scale_key}: {scale.dtype}")
            if not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise DonorError(f"FP8 scales must be positive and finite: {scale_key}")
            rows, cols = weight.shape
            out = torch.empty((rows, cols), dtype=torch.bfloat16, device="cpu")
            # Dequantize one 128-row tile at a time, avoiding another
            # full-size FP32 copy of each multi-gigabyte donor matrix.
            for block_row in range(scale.shape[0]):
                start = block_row * self.BLOCK
                stop = min(start + self.BLOCK, rows)
                expanded = scale[block_row].repeat_interleave(self.BLOCK)[:cols]
                out[start:stop] = (
                    weight[start:stop].to(torch.float32) * expanded.unsqueeze(0)
                ).to(torch.bfloat16)
            return out
        if scale_key is not None:
            raise DonorError(f"FP8 scale companion found for non-FP8 tensor: {key}")
        if weight.dtype in (torch.float8_e5m2, torch.float8_e4m3fnuz):
            raise DonorError(f"unsupported GLM-5.3 FP8 format: {key}: {weight.dtype}")
        return weight
