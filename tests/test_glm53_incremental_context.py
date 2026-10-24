"""CPU-only row-streamed GLM context projection parity and budget tests."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from iq_transfer.glm53_compile import _transport_input_only
from iq_transfer.glm53_incremental_context import (
    GLM53ContextStreamingError,
    transport_input_projection_rows,
)
from iq_transfer.transport import CoordinateMap


class _Source:
    def __init__(self, value: torch.Tensor):
        self.value = value
        self.calls: list[list[int]] = []

    def shape(self, _key):
        return tuple(self.value.shape)

    def get(self, _key):
        raise AssertionError("full donor tensor materialization forbidden")

    def get_rows(self, _key, indices):
        rows = list(indices)
        self.calls.append(rows)
        return self.value[rows]


class TestIncrementalContext(unittest.TestCase):
    def test_row_streamed_q_a_matches_original_projection(self):
        rng = np.random.default_rng(51)
        for map_matrix in (
            np.eye(7, 4),
            rng.normal(size=(7, 4)),
        ):
            with self.subTest(kind="orthogonal" if map_matrix[0, 0] == 1 else "dense"):
                source_value = torch.tensor(rng.normal(size=(9, 7)), dtype=torch.bfloat16)
                source = _Source(source_value)
                map_ = CoordinateMap(map_matrix, 1e-6)
                reference = _transport_input_only(source_value, map_)
                result = transport_input_projection_rows(
                    source=source, key="q_a",
                    residual_map=map_, chunk_rows=3,
                )
                self.assertEqual(result.dtype, source_value.dtype)
                self.assertEqual(result.shape, (9, 4))
                self.assertTrue(torch.allclose(result.float(), reference.float()))
                self.assertEqual(source.calls, [list(range(i, i + 3)) for i in (0, 3, 6)])

    def test_output_budget_rejects_before_any_row_fetch(self):
        source = _Source(torch.zeros((9, 7), dtype=torch.bfloat16))
        with self.assertRaisesRegex(GLM53ContextStreamingError, "output exceeds"):
            transport_input_projection_rows(
                source=source,
                key="q_a",
                residual_map=CoordinateMap(np.eye(7, 4), 1e-6),
                max_output_bytes=4,
            )
        self.assertEqual(source.calls, [])

    def test_reject_mismatched_source_width(self):
        source = _Source(torch.zeros((9, 7), dtype=torch.bfloat16))
        with self.assertRaisesRegex(GLM53ContextStreamingError, "donor width"):
            transport_input_projection_rows(
                source=source,
                key="q_a",
                residual_map=CoordinateMap(np.eye(6, 4), 1e-6),
            )


if __name__ == "__main__":
    unittest.main()
