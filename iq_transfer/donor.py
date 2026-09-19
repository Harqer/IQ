from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
import json


class DonorError(RuntimeError):
    pass


@dataclass(frozen=True)
class DonorConfig:
    model_name: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int | None = None
    dtype: str | None = None

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise DonorError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DonorConfig":
        required = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise DonorError(f"missing donor config fields: {', '.join(missing)}")
        return cls(
            model_name=str(data.get("_name_or_path") or data.get("name_or_path") or "unknown"),
            model_type=str(data.get("model_type") or "unknown"),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            vocab_size=int(data["vocab_size"]) if data.get("vocab_size") is not None else None,
            dtype=str(data.get("torch_dtype")) if data.get("torch_dtype") is not None else None,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DonorConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))


@dataclass(frozen=True)
class LayerRef:
    index: int
    path: str


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> None:
        if self.errors:
            raise DonorError("; ".join(self.errors))


@dataclass(frozen=True)
class OperatorRef:
    layer: int
    role: str
    source_key: str
    shape: tuple[int, ...]
    row_start: int | None = None
    row_stop: int | None = None

    def materialize(self, source: "TensorSource") -> Any:
        tensor = source.get(self.source_key)
        if self.row_start is None:
            return tensor
        return tensor[self.row_start:self.row_stop]


class TensorSource(Protocol):
    def keys(self) -> Sequence[str]:
        ...

    def shape(self, key: str) -> tuple[int, ...]:
        ...

    def get(self, key: str) -> Any:
        ...


class MappingTensorSource:
    def __init__(self, tensors: Mapping[str, Any]) -> None:
        self._tensors = tensors

    def keys(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def shape(self, key: str) -> tuple[int, ...]:
        value = self._tensors[key]
        return tuple(int(x) for x in value.shape)

    def get(self, key: str) -> Any:
        return self._tensors[key]


class DonorInspector(Protocol):
    config: DonorConfig

    def layers(self) -> tuple[LayerRef, ...]:
        ...

    def operators(self, source: TensorSource) -> tuple[OperatorRef, ...]:
        ...

    def validate_checkpoint(self, source: TensorSource) -> ValidationReport:
        ...
