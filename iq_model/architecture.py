from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable, Mapping
import json


class ArchitectureError(ValueError):
    pass


class HybridLayerType(str, Enum):
    MAMBA3 = "mamba3"
    MOE = "moe"
    CSA = "csa"
    HCA = "hca"
    DENSE_ATTENTION = "dense_attention"

    @property
    def is_attention(self) -> bool:
        return self in {
            HybridLayerType.CSA,
            HybridLayerType.HCA,
            HybridLayerType.DENSE_ATTENTION,
        }


@dataclass(frozen=True)
class HybridSchedule:
    """Explicit heterogeneous IQ backbone schedule.

    The schedule contains physical backbone operators only: sequence mixing,
    expert computation, and context attention. Reasoning recurrence, critics,
    and halting live outside the backbone schedule so they cannot be mistaken
    for token-mixing layers.
    """

    layers: tuple[HybridLayerType, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ArchitectureError("hybrid schedule must contain at least one layer")
        mamba = self.count(HybridLayerType.MAMBA3)
        if mamba == 0:
            raise ArchitectureError("hybrid schedule requires Mamba-3 layers")
        attention = sum(1 for layer in self.layers if layer.is_attention)
        if mamba <= attention:
            raise ArchitectureError(
                "IQ requires a Mamba-dominant schedule: Mamba-3 layer count must exceed attention layer count"
            )

    @classmethod
    def from_tokens(cls, tokens: Iterable[str | HybridLayerType]) -> "HybridSchedule":
        parsed: list[HybridLayerType] = []
        aliases = {
            "m": HybridLayerType.MAMBA3,
            "mamba": HybridLayerType.MAMBA3,
            "mamba3": HybridLayerType.MAMBA3,
            "e": HybridLayerType.MOE,
            "moe": HybridLayerType.MOE,
            "c": HybridLayerType.CSA,
            "csa": HybridLayerType.CSA,
            "h": HybridLayerType.HCA,
            "hca": HybridLayerType.HCA,
            "a": HybridLayerType.DENSE_ATTENTION,
            "dense": HybridLayerType.DENSE_ATTENTION,
            "dense_attention": HybridLayerType.DENSE_ATTENTION,
        }
        for token in tokens:
            if isinstance(token, HybridLayerType):
                parsed.append(token)
                continue
            key = str(token).strip().lower()
            if key in {"x", "executive"}:
                raise ArchitectureError(
                    "reasoning control is outside HybridSchedule; configure recurrence, critic, and halting separately"
                )
            try:
                parsed.append(aliases[key])
            except KeyError as exc:
                raise ArchitectureError(f"unknown hybrid layer token: {token!r}") from exc
        return cls(tuple(parsed))

    @classmethod
    def parse(cls, pattern: str) -> "HybridSchedule":
        normalized = pattern.replace(",", " ").replace("->", " ")
        return cls.from_tokens(part for part in normalized.split() if part)

    def count(self, layer_type: HybridLayerType) -> int:
        return sum(layer is layer_type for layer in self.layers)

    def positions(self, layer_type: HybridLayerType) -> tuple[int, ...]:
        return tuple(index for index, layer in enumerate(self.layers) if layer is layer_type)

    @property
    def attention_positions(self) -> tuple[int, ...]:
        return tuple(index for index, layer in enumerate(self.layers) if layer.is_attention)

    def to_tokens(self) -> tuple[str, ...]:
        return tuple(layer.value for layer in self.layers)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "layers": list(self.to_tokens()),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HybridSchedule":
        if int(data.get("schema_version", -1)) != 1:
            raise ArchitectureError(
                f"unsupported hybrid-schedule schema: {data.get('schema_version')!r}"
            )
        layers = data.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ArchitectureError("hybrid-schedule layers must be a non-empty list")
        return cls.from_tokens(str(layer) for layer in layers)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()
