from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json


class BatchArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenBatchArtifact:
    batches: tuple[dict[str, Any], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise BatchArtifactError(f"unsupported token-batch schema: {self.schema_version}")
        if not self.batches:
            raise BatchArtifactError("token-batch artifact must contain at least one batch")
        _validate_batches(self.batches)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise BatchArtifactError("PyTorch is required for token-batch artifacts") from exc
    return torch


def _validate_batches(batches: Sequence[Mapping[str, Any]]) -> None:
    torch = _require_torch()
    allowed = {"input_ids", "attention_mask", "position_ids"}
    for index, batch in enumerate(batches):
        unknown = sorted(set(batch) - allowed)
        if unknown:
            raise BatchArtifactError(f"batch {index} contains unsupported fields: {', '.join(unknown)}")
        ids = batch.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise BatchArtifactError(f"batch {index} input_ids must be a rank-2 tensor")
        if ids.dtype not in (torch.int32, torch.int64):
            raise BatchArtifactError(f"batch {index} input_ids must be integer typed")
        for optional in ("attention_mask", "position_ids"):
            value = batch.get(optional)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != ids.shape:
                raise BatchArtifactError(f"batch {index} {optional} must match input_ids shape")
            if value.dtype not in (torch.int32, torch.int64, torch.bool):
                raise BatchArtifactError(f"batch {index} {optional} must be integer/bool typed")


def save_token_batches(batches: Sequence[Mapping[str, Any]], path: str | Path) -> tuple[Path, Path]:
    torch = _require_torch()
    batch_tuple = tuple(dict(batch) for batch in batches)
    _validate_batches(batch_tuple)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise BatchArtifactError("safetensors is required to persist token batches") from exc

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = base.with_suffix(".safetensors")
    metadata_path = base.with_suffix(".json")
    tensors: dict[str, Any] = {}
    fields: list[list[str]] = []
    for i, batch in enumerate(batch_tuple):
        names = sorted(batch)
        fields.append(names)
        for name in names:
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise BatchArtifactError(f"batch {i} field {name} is not a tensor")
            tensors[f"batch/{i:06d}/{name}"] = value.detach().contiguous().cpu()
    save_file(tensors, str(tensor_path))
    metadata = {"schema_version": 1, "batch_count": len(batch_tuple), "fields": fields}
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return tensor_path, metadata_path


def load_token_batches(path: str | Path) -> TokenBatchArtifact:
    _require_torch()
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise BatchArtifactError("safetensors is required to load token batches") from exc

    base = Path(path)
    tensor_path = base.with_suffix(".safetensors")
    metadata_path = base.with_suffix(".json")
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise BatchArtifactError(f"token-batch artifact is incomplete: {base}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchArtifactError("token-batch metadata is invalid") from exc
    if metadata.get("schema_version") != 1:
        raise BatchArtifactError(f"unsupported token-batch schema: {metadata.get('schema_version')}")
    batch_count = int(metadata.get("batch_count", -1))
    fields = metadata.get("fields")
    if batch_count <= 0 or not isinstance(fields, list) or len(fields) != batch_count:
        raise BatchArtifactError("token-batch metadata has invalid batch_count/fields")

    tensors = load_file(str(tensor_path), device="cpu")
    expected: set[str] = set()
    batches: list[dict[str, Any]] = []
    for i in range(batch_count):
        names = fields[i]
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise BatchArtifactError(f"batch {i} field list is invalid")
        batch: dict[str, Any] = {}
        for name in names:
            if name not in {"input_ids", "attention_mask", "position_ids"}:
                raise BatchArtifactError(f"batch {i} contains unsupported field {name!r}")
            key = f"batch/{i:06d}/{name}"
            expected.add(key)
            if key not in tensors:
                raise BatchArtifactError(f"token-batch tensor is missing: {key}")
            batch[name] = tensors[key]
        batches.append(batch)
    if set(tensors) != expected:
        extra = sorted(set(tensors) - expected)
        raise BatchArtifactError(f"token-batch tensor/index mismatch; extra={extra}")
    return TokenBatchArtifact(tuple(batches))
