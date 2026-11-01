"""Bounded one-batch-ahead GLM sparse expert I/O for the actual transfer.

Remote prefetch uses a separate Hugging Face HTTP Session and its own validated
safetensors headers. One worker can fetch at most one future batch while CUDA
transforms and commits the current batch. Nothing is ever prefetched for saved
(resumed) experts. There are no extra validation-only donor reads.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
from typing import Any

import torch

from .glm53_gpu_batch import ExpertSpec
from .glm53_gpu_projection import ProjectionComputeError
from .http_range_source import HubSafetensorsSource


class StagedBatchSource:
    """Expose *only* tensors that were fetched for a specific planned batch."""

    def __init__(self, staged: dict[str, torch.Tensor]):
        self._staged = staged

    def get(self, key: str) -> torch.Tensor:
        try:
            return self._staged[key]
        except KeyError as exc:
            raise ProjectionComputeError(
                f"prefetched batch unexpectedly lacks {key}"
            ) from exc


class RoutedBatchPrefetch:
    """One in-flight I/O batch; isolated HTTP connection and bounded host RAM."""

    def __init__(self, source: Any, *, donor_root: str):
        mode = os.environ.get("IQ_MOE_GPU_PREFETCH", "1")
        if mode not in ("0", "1"):
            raise ProjectionComputeError("IQ_MOE_GPU_PREFETCH must be 0 or 1")
        self.enabled = mode == "1" and isinstance(source, HubSafetensorsSource)
        self.donor_root = donor_root
        self._pool: ThreadPoolExecutor | None = None
        self._worker_source: HubSafetensorsSource | None = None
        # Current + one future batch only. Never allow an unbounded queue of
        # source tensor downloads consuming host RAM.
        self._outstanding: set[Future] = set()
        if self.enabled:
            # Do not share requests.Session or its mutable header cache
            # with the main thread. Immutable repository revision is copied.
            self._worker_source = HubSafetensorsSource(
                source.model_dir,
                repo_id=source.repo_id,
                revision=source.revision,
                token=source.token,
                max_request_bytes=source.max_request_bytes,
                max_tensor_bytes=source.max_tensor_bytes,
            )
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iq-glm-range")

    def submit(self, specs: list[ExpertSpec]) -> Future | None:
        if not specs or not self.enabled:
            return None
        assert self._pool is not None
        if len(self._outstanding) >= 2:
            raise ProjectionComputeError(
                "HTTP prefetch queue exceeds current+one-ahead batch limit"
            )
        future = self._pool.submit(self._load, tuple(specs))
        self._outstanding.add(future)
        return future

    def _load(self, specs: tuple[ExpertSpec, ...]) -> dict[str, torch.Tensor]:
        assert self._worker_source is not None
        staged: dict[str, torch.Tensor] = {}
        for spec in specs:
            for kind in ("gate", "up", "down"):
                key = f"{self.donor_root}.experts.{spec.donor_id}.{kind}_proj.weight"
                staged[key] = self._worker_source.get(key)
        return staged

    def source_for(self, future: Future | None, fallback: Any) -> Any:
        if future is None:
            return fallback
        if future not in self._outstanding:
            raise ProjectionComputeError("unknown or already consumed prefetch future")
        try:
            return StagedBatchSource(future.result())
        finally:
            self._outstanding.remove(future)

    def close(self) -> None:
        if self._pool is not None:
            for future in self._outstanding:
                future.cancel()
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._outstanding.clear()
        if self._worker_source is not None:
            self._worker_source.close()
            self._worker_source = None

    def __enter__(self) -> "RoutedBatchPrefetch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
