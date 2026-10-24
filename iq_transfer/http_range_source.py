"""Fail-closed, bounded HTTP Range access to indexed Hugging Face safetensors.

This loader never downloads complete remote shards. Full tensor materialization
is capped deliberately; transforms requiring larger tensors must become
chunk-compatible before using this source on memory-constrained machines.
"""
from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import get_token, hf_hub_url

from scripts.probe_glm53_remote_ranges import (
    MAX_HEADER,
    RangeProbeError,
    fetch_exact_range,
    probe_shard,
)

from .checkpoint import SafetensorsSource
from .donor import DonorError

# Safetensors dtype name -> (torch dtype attribute, bytes per element).
_DTYPES: dict[str, tuple[str, int]] = {
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
    "U8": ("uint8", 1),
    "BOOL": ("bool", 1),
}


class HubSafetensorsSource(SafetensorsSource):
    """Use existing local shards or HTTP 206 *tensor* ranges, never full shards.

    max_tensor_bytes caps returned tensors and row gatherings. max_request_bytes
    caps each HTTP request; during materialization transient copies may raise
    process memory above max_tensor_bytes. It is not a whole-process RAM limit.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        repo_id: str,
        revision: str,
        token: str | None = None,
        max_request_bytes: int = 4 * 1024 * 1024,
        max_tensor_bytes: int = 128 * 1024 * 1024,
        session: Any | None = None,
    ) -> None:
        super().__init__(model_dir)
        if not repo_id or not revision:
            raise DonorError("remote checkpoint repo and pinned revision are required")
        if not 1 <= max_request_bytes <= MAX_HEADER:
            raise DonorError("invalid max_request_bytes (must be 1..MAX_HEADER)")
        if not 1 <= max_tensor_bytes:
            raise DonorError("max_tensor_bytes must be positive")
        self.repo_id = repo_id
        self.revision = revision
        self.token = token if token is not None else get_token()
        self.max_request_bytes = max_request_bytes
        self.max_tensor_bytes = max_tensor_bytes
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        # shard -> (validated header, complete shard size, tensor-data start)
        self._headers: dict[str, tuple[dict, int, int]] = {}
        # Retained for compatibility with preexisting local temporary-shard
        # cleanup. Range reads never add paths to this collection.
        self._ephemeral_shards: set[Path] = set()
        self._shard_last_layer: dict[str, int] = {}
        for key, shard in self._weight_map.items():
            parts = key.split(".")
            last = -1
            if len(parts) > 2 and parts[:2] == ["model", "layers"]:
                try:
                    last = int(parts[2])
                except ValueError:
                    pass
            self._shard_last_layer[shard] = max(
                last, self._shard_last_layer.get(shard, -1)
            )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def clear_ephemeral_shards(self, *, completed_layer: int | None = None) -> None:
        """Remove only explicitly tracked old temporary files, never local inputs.

        New HTTP range requests do not create or register shard files.
        """
        for path in tuple(self._ephemeral_shards):
            if completed_layer is not None and (
                self._shard_last_layer.get(path.name, -1) > completed_layer
            ):
                continue
            try:
                path.unlink(missing_ok=True)
            finally:
                self._ephemeral_shards.discard(path)

    def _shard(self, key: str) -> str:
        try:
            shard = self._weight_map[key]
        except KeyError as exc:
            raise DonorError(f"unknown checkpoint tensor: {key}") from exc
        if (
            not shard.endswith(".safetensors")
            or Path(shard).name != shard
            or "\\" in shard
            or shard in (".", "..")
        ):
            raise DonorError(f"unsafe checkpoint shard name for {key}")
        return shard

    def _existing_local(self, key: str) -> bool:
        return (self.model_dir / self._shard(key)).is_file()

    def _metadata(self, shard: str) -> tuple[dict, int, int]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        try:
            value = probe_shard(
                self._session,
                hf_hub_url(self.repo_id, shard, revision=self.revision),
                token=self.token,
            )
        except (RangeProbeError, requests.RequestException, ValueError) as exc:
            raise DonorError(f"remote shard metadata rejected for {shard}: {exc}") from exc
        self._headers[shard] = value
        return value

    def _entry(self, key: str) -> tuple[str, tuple[int, ...], str, int, int]:
        shard = self._shard(key)
        header, file_bytes, data_start = self._metadata(shard)
        entry = header.get(key)
        if not isinstance(entry, dict):
            raise DonorError(f"tensor {key} missing from official shard header")
        name = entry.get("dtype")
        if name not in _DTYPES:
            raise DonorError(f"unsupported safetensors dtype {name!r}: {key}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise DonorError(f"invalid tensor shape in shard header: {key}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise DonorError(f"invalid tensor offsets: {key}")
        begin, end = offsets
        if type(begin) is not int or type(end) is not int:
            raise DonorError(f"noninteger tensor offsets: {key}")
        expected = prod(shape) * _DTYPES[name][1]
        if end - begin != expected or begin < 0 or data_start + end > file_bytes:
            raise DonorError(f"tensor shape/dtype/byte-length mismatch: {key}")
        return shard, tuple(shape), name, data_start + begin, expected

    @staticmethod
    def _torch_dtype(name: str):
        try:
            import torch
        except ImportError as exc:
            raise DonorError("PyTorch required to materialize a remote tensor") from exc
        return getattr(torch, _DTYPES[name][0])

    def _read(self, shard: str, start: int, size: int) -> bytearray:
        if size > self.max_tensor_bytes:
            raise DonorError(
                f"tensor/read of {size} bytes exceeds max_tensor_bytes="
                f"{self.max_tensor_bytes}; chunk-compatible transform required"
            )
        result = bytearray(size)
        url = hf_hub_url(self.repo_id, shard, revision=self.revision)
        _, total, _ = self._metadata(shard)
        if start < 0 or start + size > total:
            raise DonorError("remote tensor byte range lies outside shard")
        for offset in range(0, size, self.max_request_bytes):
            length = min(self.max_request_bytes, size - offset)
            try:
                payload, remote_total = fetch_exact_range(
                    self._session, url, start + offset, start + offset + length - 1,
                    token=self.token,
                )
            except (RangeProbeError, requests.RequestException, ValueError) as exc:
                raise DonorError(f"remote tensor-range request rejected: {exc}") from exc
            if remote_total != total:
                raise DonorError("shard size changed between header and tensor reads")
            result[offset:offset + length] = payload
        return result

    def shape(self, key: str) -> tuple[int, ...]:
        if self._existing_local(key):
            return super().shape(key)
        return self._entry(key)[1]

    def get(self, key: str) -> Any:
        if self._existing_local(key):
            return super().get(key)
        import torch
        shard, shape, name, start, size = self._entry(key)
        dtype = self._torch_dtype(name)
        if not size:
            return torch.empty(shape, dtype=dtype)
        payload = self._read(shard, start, size)
        # Clone detaches the returned tensor from the temporary network buffer.
        return torch.frombuffer(payload, dtype=dtype).clone().reshape(shape)

    def get_rows(self, key: str, row_indices: Any) -> Any:
        if self._existing_local(key):
            return super().get_rows(key, row_indices)
        import torch
        shard, shape, name, start, _ = self._entry(key)
        if len(shape) != 2:
            raise DonorError(f"get_rows requires a rank-2 tensor: {key}")
        indices = torch.as_tensor(row_indices, dtype=torch.long, device="cpu").reshape(-1)
        dtype = self._torch_dtype(name)
        row_bytes = shape[1] * _DTYPES[name][1]
        if indices.numel() and (bool((indices < 0).any()) or int(indices.max()) >= shape[0]):
            raise DonorError("remote tensor row index out of bounds")
        requested_bytes = indices.numel() * row_bytes
        if requested_bytes > self.max_tensor_bytes:
            raise DonorError(
                f"row gathering requires {requested_bytes} bytes, exceeds "
                f"max_tensor_bytes={self.max_tensor_bytes}"
            )
        if not indices.numel():
            return torch.empty((0, shape[1]), dtype=dtype)
        unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
        gathered = torch.empty((len(unique), shape[1]), dtype=dtype)
        i = 0
        while i < len(unique):
            j = i + 1
            while j < len(unique) and int(unique[j]) == int(unique[j - 1]) + 1:
                j += 1
            first = int(unique[i])
            span_bytes = (j - i) * row_bytes
            # Bound large contiguous runs; _read itself sends bounded requests.
            if span_bytes > self.max_tensor_bytes:
                raise DonorError("contiguous row run exceeds memory cap")
            if span_bytes:
                payload = self._read(shard, start + first * row_bytes, span_bytes)
                gathered[i:j] = torch.frombuffer(payload, dtype=dtype).clone().reshape(
                    j - i, shape[1]
                )
            i = j
        return gathered[inverse].contiguous()
