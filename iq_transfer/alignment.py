from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np


class AlignmentError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class ByteSpan:
    start: int
    end: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise AlignmentError(f"invalid byte span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, order=True)
class TokenByteSpan:
    token_index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.token_index < 0:
            raise AlignmentError("token_index must be non-negative")
        if self.start < 0 or self.end <= self.start:
            raise AlignmentError(f"invalid token byte span [{self.start}, {self.end})")


@dataclass(frozen=True)
class PairedSpanActivations:
    spans: tuple[ByteSpan, ...]
    source: np.ndarray
    target: np.ndarray


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_token_spans(token_spans: Iterable[TokenByteSpan], token_count: int) -> tuple[TokenByteSpan, ...]:
    spans = tuple(sorted(token_spans, key=lambda x: (x.start, x.end, x.token_index)))
    if len(spans) != token_count:
        raise AlignmentError(f"expected one byte span per token: got {len(spans)} spans for {token_count} tokens")
    indices = sorted(x.token_index for x in spans)
    if indices != list(range(token_count)):
        raise AlignmentError("token byte spans must cover token indices exactly once")
    previous_end = 0
    for i, span in enumerate(spans):
        if i and span.start < previous_end:
            raise AlignmentError("token byte spans must not overlap")
        previous_end = span.end
    return spans


def pool_activations_by_byte_spans(
    activations: Any,
    token_spans: Iterable[TokenByteSpan],
    canonical_spans: Iterable[ByteSpan],
    *,
    require_full_coverage: bool = True,
) -> np.ndarray:
    x = _numpy(activations)
    if x.ndim != 2:
        raise AlignmentError("activations must be rank-2 [tokens, features]")
    if not np.isfinite(x).all():
        raise AlignmentError("activations contain non-finite values")
    tokens = _validate_token_spans(token_spans, x.shape[0])
    spans = tuple(canonical_spans)
    if not spans:
        raise AlignmentError("at least one canonical byte span is required")

    pooled = np.empty((len(spans), x.shape[1]), dtype=np.float64)
    for row, span in enumerate(spans):
        total_overlap = 0
        weighted = np.zeros(x.shape[1], dtype=np.float64)
        for token in tokens:
            overlap = max(0, min(span.end, token.end) - max(span.start, token.start))
            if overlap:
                weighted += x[token.token_index] * overlap
                total_overlap += overlap
        if total_overlap == 0:
            raise AlignmentError(f"canonical span [{span.start}, {span.end}) has no token coverage")
        if require_full_coverage and total_overlap != span.length:
            raise AlignmentError(
                f"canonical span [{span.start}, {span.end}) coverage is {total_overlap}/{span.length} bytes"
            )
        pooled[row] = weighted / total_overlap
    return pooled


def align_token_activations_by_bytes(
    source_activations: Any,
    source_token_spans: Iterable[TokenByteSpan],
    target_activations: Any,
    target_token_spans: Iterable[TokenByteSpan],
    canonical_spans: Iterable[ByteSpan],
    *,
    require_full_coverage: bool = True,
) -> PairedSpanActivations:
    spans = tuple(canonical_spans)
    source = pool_activations_by_byte_spans(
        source_activations, source_token_spans, spans, require_full_coverage=require_full_coverage
    )
    target = pool_activations_by_byte_spans(
        target_activations, target_token_spans, spans, require_full_coverage=require_full_coverage
    )
    return PairedSpanActivations(spans=spans, source=source, target=target)
