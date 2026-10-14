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
    PARALLEL_CSA = "parallel_csa"
    PARALLEL_HCA = "parallel_hca"

    @property
    def is_attention(self) -> bool:
        return self in {
            HybridLayerType.CSA,
            HybridLayerType.HCA,
            HybridLayerType.DENSE_ATTENTION,
            HybridLayerType.PARALLEL_CSA,
            HybridLayerType.PARALLEL_HCA,
        }

    @property
    def is_parallel(self) -> bool:
        return self in {
            HybridLayerType.PARALLEL_CSA,
            HybridLayerType.PARALLEL_HCA,
        }

    @property
    def has_mamba(self) -> bool:
        return self is HybridLayerType.MAMBA3 or self.is_parallel

    @property
    def has_moe(self) -> bool:
        return self is HybridLayerType.MOE or self.is_parallel


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
        if not self.mamba_positions:
            raise ArchitectureError("hybrid schedule requires Mamba-3 layers")
        # Existing serial schedules retain their Mamba-dominance contract.
        # Fused layers each contain both branches, so no artificial dominance
        # constraint is imposed on a parallel schedule.
        attention = len(self.attention_positions)
        if not self.parallel_positions and len(self.mamba_positions) <= attention:
            raise ArchitectureError(
                "IQ serial schedule requires more Mamba slots than attention slots"
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
            "parallel_csa": HybridLayerType.PARALLEL_CSA,
            "parallel_hca": HybridLayerType.PARALLEL_HCA,
            "pc": HybridLayerType.PARALLEL_CSA,
            "ph": HybridLayerType.PARALLEL_HCA,
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
    def mamba_positions(self) -> tuple[int, ...]:
        return tuple(index for index, layer in enumerate(self.layers) if layer.has_mamba)

    @property
    def moe_positions(self) -> tuple[int, ...]:
        return tuple(index for index, layer in enumerate(self.layers) if layer.has_moe)

    @property
    def parallel_positions(self) -> tuple[int, ...]:
        return tuple(index for index, layer in enumerate(self.layers) if layer.is_parallel)

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
