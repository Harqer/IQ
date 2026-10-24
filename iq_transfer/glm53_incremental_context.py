"""Bounded row-range transport of a GLM attention input projection.

This handles the q_a input-only transform in streaming compiles. The larger
MLA q_b/kv_b/O refactor is still the original full-tensor implementation.
"""
from __future__ import annotations

import numpy as np
import torch

from .checkpoint import SafetensorsSource
from .glm53_incremental_moe import _convert, _finite, _inverse
from .transport import CoordinateMap


class GLM53ContextStreamingError(RuntimeError):
    pass


def transport_input_projection_rows(
    *,
    source: SafetensorsSource,
    key: str,
    residual_map: CoordinateMap,
    chunk_rows: int = 128,
    max_output_bytes: int = 128 * 1024 * 1024,
) -> torch.Tensor:
    """Transport W @ pinv(P).T without materializing the donor W.

    The result itself is materialized as a BF16 tensor, capped separately
    from the network input chunks. This is not a whole-process RAM limit.
    """
    if chunk_rows < 1 or max_output_bytes < 1:
        raise GLM53ContextStreamingError("row count and output budget must be positive")
    shape = source.shape(key)
    if len(shape) != 2:
        raise GLM53ContextStreamingError("attention input matrix must have rank 2")
    rows, donor_width = shape
    if rows <= 0 or donor_width <= 0:
        raise GLM53ContextStreamingError("attention input matrix cannot be empty")
    if residual_map.matrix.shape[0] != donor_width:
        raise GLM53ContextStreamingError("attention input map donor width mismatch")
    target_width = residual_map.matrix.shape[1]
    if rows * target_width * 2 > max_output_bytes:
        raise GLM53ContextStreamingError(
            "attention input projection output exceeds allocated tensor budget"
        )
    inv = _inverse(residual_map)
    output = torch.empty((rows, target_width), dtype=torch.bfloat16)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = _finite(
            source.get_rows(key, range(start, end)),
            (end - start, donor_width),
            key,
        )
        # The old compiler transports the whole matrix in float64. Running
        # the same operation by row block preserves each row independently.
        transformed = chunk @ inv
        output[start:end] = _convert(transformed)
    return output
