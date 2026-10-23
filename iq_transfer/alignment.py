from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .transport import CoordinateMap
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


def _validate_token_spans(
    token_spans: Iterable[TokenByteSpan], token_count: int
) -> tuple[TokenByteSpan, ...]:
    spans = tuple(sorted(token_spans, key=lambda x: (x.start, x.end, x.token_index)))
    if len(spans) != token_count:
        raise AlignmentError(
            f"expected one byte span per token: got {len(spans)} spans for {token_count} tokens"
        )
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
            raise AlignmentError(
                f"canonical span [{span.start}, {span.end}) has no token coverage"
            )
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
        source_activations,
        source_token_spans,
        spans,
        require_full_coverage=require_full_coverage,
    )
    target = pool_activations_by_byte_spans(
        target_activations,
        target_token_spans,
        spans,
        require_full_coverage=require_full_coverage,
    )
    return PairedSpanActivations(spans=spans, source=source, target=target)


@dataclass(frozen=True)
class CausalTokenAlignment:
    """GLM source hidden projected into IQ's vocabulary-position and hidden frame."""

    teacher_aligned_hidden: np.ndarray
    valid_token_mask: np.ndarray


def align_glm_states_to_mamba_tokens(
    source_hidden: Any,
    source_token_spans: Iterable[TokenByteSpan],
    target_token_spans: Iterable[TokenByteSpan],
    *,
    source_document_sha256: str,
    target_document_sha256: str,
    coordinate_map: CoordinateMap,
) -> CausalTokenAlignment:
    """Align one real, unpacked document without observing future GLM tokens.

    A Mamba token ending at byte b may only use the most recent *complete*
    GLM token ending at or before b. If none exists, the target position is
    masked rather than filled with an invented or future source state.

    Source and target offsets must cover exactly the same UTF-8 document bytes.
    The caller must capture post-MLP GLM residual states at causal token prefixes
    and fit the source-to-IQ output map on independent calibration records.
    """
    from .transport import CoordinateMap

    if not isinstance(coordinate_map, CoordinateMap):
        raise AlignmentError("a fitted GLM-to-IQ CoordinateMap is required")
    if (
        len(source_document_sha256) != 64
        or len(target_document_sha256) != 64
        or source_document_sha256.lower() != target_document_sha256.lower()
        or any(c not in "0123456789abcdef" for c in source_document_sha256.lower())
    ):
        raise AlignmentError(
            "source and target must identify the same SHA-256 UTF-8 document"
        )
    if (
        coordinate_map.diagnostics is None
        or coordinate_map.diagnostics.validation_rmse is None
        or not np.isfinite(coordinate_map.diagnostics.validation_rmse)
    ):
        raise AlignmentError(
            "the output coordinate map must have held-out validation metrics"
        )
    if (
        "residual_out" not in coordinate_map.source_space
        or "residual_out" not in coordinate_map.target_space
    ):
        raise AlignmentError(
            "Stage 2 requires an explicit residual_out-to-residual_out map"
        )

    source = _numpy(source_hidden)
    if source.ndim != 2 or source.shape[0] == 0 or not np.isfinite(source).all():
        raise AlignmentError("source_hidden must be finite [GLM-token, GLM-hidden]")
    source_spans = _validate_token_spans(source_token_spans, source.shape[0])
    target_spans_unsorted = tuple(target_token_spans)
    if not target_spans_unsorted:
        raise AlignmentError("target tokens cannot be empty")
    target_spans = _validate_token_spans(
        target_spans_unsorted, len(target_spans_unsorted)
    )

    def full_coverage(spans: tuple[TokenByteSpan, ...]) -> int:
        previous = 0
        for span in spans:
            if span.start != previous:
                raise AlignmentError(
                    "source/target spans must continuously cover UTF-8 bytes"
                )
            previous = span.end
        return previous

    if full_coverage(source_spans) != full_coverage(target_spans):
        raise AlignmentError(
            "source and target tokenizations cover different byte lengths"
        )
    matrix = np.asarray(coordinate_map.matrix, dtype=np.float64)
    if matrix.shape[0] != source.shape[1]:
        raise AlignmentError("source hidden width does not match GLM-to-IQ map")
    if not np.isfinite(matrix).all():
        raise AlignmentError("coordinate map must be finite")

    # Sorted, non-overlapping token spans have monotonically increasing ends.
    source_ends = np.asarray([s.end for s in source_spans], dtype=np.int64)
    result = np.zeros((len(target_spans), matrix.shape[1]), dtype=np.float32)
    valid = np.zeros(len(target_spans), dtype=np.bool_)
    projected = source @ matrix
    for target in target_spans:
        source_position = int(
            np.searchsorted(source_ends, target.end, side="right") - 1
        )
        if source_position < 0:
            continue
        result[target.token_index] = projected[
            source_spans[source_position].token_index
        ]
        valid[target.token_index] = True
    if not bool(valid.any()):
        raise AlignmentError("no causal GLM token prefixes are available for IQ tokens")
    if not np.isfinite(result).all():
        raise AlignmentError("aligned GLM hidden states are non-finite")
    return CausalTokenAlignment(result, valid)
