"""CPU-only MOHAWK Stage-2 artifact tests; never synthesize deployment evidence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from iq_transfer.alignment import AlignmentError
from iq_transfer.stage2_artifacts import Stage2ArtifactError, write_stage2_example
from iq_transfer.transport import CoordinateMap, MapDiagnostics


REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


def validated_map(*, verified=True):
    mat = np.zeros((2, 4096), dtype=np.float64)
    mat[0, 0] = 1
    mat[1, 1] = 1
    return CoordinateMap(
        matrix=mat, ridge=1e-3,
        source_space="glm53.layer.3.residual_out",
        target_space="iq.layer.5.residual_out",
        diagnostics=MapDiagnostics(
            sample_count=10,
            source_features=2,
            target_features=4096,
            effective_rank=2,
            condition_number=2.0,
            fit_rmse=0.1,
            validation_rmse=0.2 if verified else None,
        ),
    )


class TestStage2Artifacts(unittest.TestCase):
    def kwargs(self):
        return {
            "document_bytes": b"abcd",
            "glm_states": np.array([[1., 0.], [0., 2.]], dtype=np.float32),
            "glm_token_spans": [(0, 2), (2, 4)],
            "iq_token_spans": [(0, 1), (1, 3), (3, 4)],
            "iq_token_ids": [10, 11, 12],
            "residual_out_map": validated_map(),
            "source_glm_layer": 3,
            "target_iq_layer": 5,
            "teacher_revision": REV,
            "split": "train",
        }

    def test_real_causal_alignment_masks_future_source_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_stage2_example(
                Path(tmp) / "aligned", **self.kwargs()
            )
            tensors = load_file(str(Path(tmp) / "aligned" / "alignment.safetensors"))
            self.assertEqual(manifest["causally_aligned_tokens"], 2)
            mask = tensors["teacher_alignment_mask"].tolist()
            self.assertEqual(mask, [[False, True, True]])
            hidden = tensors["teacher_aligned_hidden"][0]
            self.assertAlmostEqual(float(hidden[1, 0]), 1.0, places=5)
            self.assertAlmostEqual(float(hidden[2, 1]), 2.0, places=5)
            self.assertFalse(manifest["quality_verified"])

    def test_unvalidated_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = self.kwargs()
            options["residual_out_map"] = validated_map(verified=False)
            with self.assertRaises(AlignmentError):
                write_stage2_example(Path(tmp) / "aligned", **options)

    def test_wrong_stage_identity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = self.kwargs()
            options["source_glm_layer"] = 4
            with self.assertRaisesRegex(Stage2ArtifactError, "expected measured map"):
                write_stage2_example(Path(tmp) / "aligned", **options)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aligned"
            write_stage2_example(path, **self.kwargs())
            with self.assertRaisesRegex(Stage2ArtifactError, "exists"):
                write_stage2_example(path, **self.kwargs())


if __name__ == "__main__":
    unittest.main()
