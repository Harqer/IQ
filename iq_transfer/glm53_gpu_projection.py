"""CUDA-preferred bounded matrix products for GLM->IQ WARM expert transport.

CPU preserves historical NumPy float64 algebra. GPU computes in FP32,
which represents BF16 donor values exactly but introduces FP32 matmul rounding.
Every transport result is checked for finiteness before checkpoint encoding.
"""
from __future__ import annotations

import os
import numpy as np
import torch


class ProjectionComputeError(RuntimeError):
    pass


class ProjectionOps:
    def __init__(self, mode: str | None = None, *, row_block: int = 512):
        mode = mode or os.environ.get("IQ_PROJECTION_DEVICE", "auto")
        if mode not in ("auto", "cpu", "cuda"):
            raise ProjectionComputeError("IQ_PROJECTION_DEVICE must be auto/cpu/cuda")
        if mode == "cuda" and not torch.cuda.is_available():
            raise ProjectionComputeError("CUDA required but not available")
        self.cuda = (mode != "cpu" and torch.cuda.is_available())
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.row_block = row_block
        if row_block <= 0:
            raise ProjectionComputeError("row_block must be positive")
        self.cache: dict[tuple[int,tuple[int,...],tuple[int,...]],torch.Tensor] = {}

    def _to_device(self, a: np.ndarray, cache: bool = False) -> torch.Tensor:
        ar = np.asarray(a)
        key=(int(ar.__array_interface__["data"][0]),tuple(ar.shape),tuple(ar.strides))
        if cache and key in self.cache:
            return self.cache[key]
        t=torch.as_tensor(np.ascontiguousarray(ar,dtype=np.float32)).to(self.device)
        if cache:
            self.cache[key]=t
        return t

    def matmul(self, left: np.ndarray, right: np.ndarray, *, cache_left=False, cache_right=False) -> np.ndarray:
        if not self.cuda:
            return np.asarray(left,dtype=np.float64) @ np.asarray(right,dtype=np.float64)
        a=np.asarray(left)
        b=np.asarray(right)
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
            raise ProjectionComputeError("matrix product shape mismatch")
        with torch.no_grad():
            a_gpu=self._to_device(a, cache=cache_left)
            b_gpu=self._to_device(b, cache=cache_right)
            blocks=[]
            for row in range(0,a.shape[0],self.row_block):
                chunk=a_gpu[row:row+self.row_block] @ b_gpu
                blocks.append(chunk.cpu())
            result=torch.cat(blocks).numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise ProjectionComputeError("nonfinite GPU projection")
        return result

    def close(self)->None:
        self.cache.clear()
