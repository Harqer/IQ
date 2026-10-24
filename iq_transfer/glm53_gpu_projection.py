"""GPU-first, memory-bounded GLM->IQ matrix products.

The CPU path preserves NumPy float64 reference semantics; GPU computes FP32
matrix products with TF32 disabled. GPU OOM shrinks the row tile and retries,
NEVER silently falling back to CPU (which may burn hours).
"""
from __future__ import annotations

import os
import numpy as np
import torch


class ProjectionComputeError(RuntimeError):
    pass


class ProjectionOps:
    def __init__(self, mode: str | None = None, *, row_block: int = 512,
                 min_row_block: int = 16):
        mode = mode or os.environ.get("IQ_PROJECTION_DEVICE", "auto")
        if mode not in ("auto", "cpu", "cuda"):
            raise ProjectionComputeError("IQ_PROJECTION_DEVICE must be auto/cpu/cuda")
        if mode == "cuda" and not torch.cuda.is_available():
            raise ProjectionComputeError("CUDA explicitly required but unavailable")
        self.cuda = mode != "cpu" and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if min_row_block < 1 or row_block < 1 or (self.cuda and row_block < min_row_block):
            raise ProjectionComputeError("invalid GPU tile sizes")
        self.row_block, self.min_row_block = row_block, min_row_block
        # Keep the original array alive with cached GPU data to avoid pointer
        # reuse accidentally returning a previous stage's projection basis.
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, torch.Tensor]] = {}
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = False

    def _to_device(self, array: np.ndarray, *, cache: bool = False) -> torch.Tensor:
        arr = np.asarray(array)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ProjectionComputeError("expected finite rank-2 matrix")
        key = id(array)
        if cache and key in self.cache:
            original, snapshot, tensor = self.cache[key]
            if original is array and np.array_equal(snapshot, arr):
                return tensor
        value = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        result = value.to(self.device)
        if cache:
            self.cache[key] = (array, np.array(arr, copy=True), result)
        return result

    def matmul(self, left: np.ndarray, right: np.ndarray, *,
               cache_left: bool = False, cache_right: bool = False) -> np.ndarray:
        a = np.asarray(left)
        b = np.asarray(right)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ProjectionComputeError("matrix product shape mismatch")
        if not self.cuda:
            return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
        if cache_left:
            # Caching the full left matrix defeats bounded source transfer.
            cache_left = False
        out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
        with torch.inference_mode():
            rhs = self._to_device(right, cache=cache_right)
            row = 0
            block = self.row_block
            while row < a.shape[0]:
                stop = min(row + block, a.shape[0])
                try:
                    lhs = self._to_device(a[row:stop])
                    partial = lhs @ rhs
                    out[row:stop] = partial.cpu().double().numpy()
                    del partial, lhs
                    row = stop
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    if block <= self.min_row_block:
                        raise ProjectionComputeError(
                            f"CUDA OOM at minimum tile {block}; checkpoint and restart on larger GPU"
                        ) from exc
                    block = max(self.min_row_block, block // 2)
                    print(f"GLM53_CUDA_OOM_REDUCED_ROW_TILE={block}", flush=True)
        if not np.isfinite(out).all():
            raise ProjectionComputeError("nonfinite GPU projection")
        return out

    def check_orthonormal(self, matrix: np.ndarray, *, atol: float = 1e-4) -> bool:
        p = np.asarray(matrix)
        if p.ndim != 2 or p.shape[0] < p.shape[1]:
            return False
        if not self.cuda:
            raise ProjectionComputeError("GPU basis verification requested on CPU")
        with torch.inference_mode():
            # No cache: checking mutated calibration maps must not reuse a stale view.
            gpu = self._to_device(matrix, cache=False)
            for j in range(0, p.shape[1], 128):
                k = min(j + 128, p.shape[1])
                gram = gpu.T @ gpu[:, j:k]
                col = torch.arange(j, k, device=self.device)
                gram[col, torch.arange(k - j, device=self.device)] -= 1.
                if not bool(torch.isfinite(gram).all()) or float(gram.abs().max()) > atol:
                    return False
        return True

    def close(self) -> None:
        self.cache.clear()
        if self.cuda:
            torch.cuda.empty_cache()
