"""No-cloud-cost tests of full-transfer, multi-expert CUDA batching."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file

from iq_transfer.glm53_gpu_batch import ExpertSpec, batch_capacity, device_bases
from iq_transfer.glm53_gpu_projection import ProjectionOps, ProjectionComputeError
from iq_transfer.glm53_incremental_moe import save_incremental_moe_stage
from tests.test_glm53_incremental_moe import _donor


class BatchCapacityTests(unittest.TestCase):
    def test_bounded_real_work_without_dropping_experts(self):
        # GLM-5.3 BF16 routed weights: each expert has three rectangular matrices.
        self.assertEqual(batch_capacity(
            source_width=6144, intermediate_width=2048,
            expert_count=256, maximum=16), 7)
        self.assertEqual(batch_capacity(
            source_width=6144, intermediate_width=2048,
            expert_count=3, maximum=16), 3)

    def test_invalid_capacity_fails_closed(self):
        with self.assertRaises(ProjectionComputeError):
            batch_capacity(source_width=6144, intermediate_width=2048,
                           expert_count=256, maximum=17)
        with self.assertRaises(ProjectionComputeError):
            batch_capacity(source_width=6144, intermediate_width=2048,
                           expert_count=256, maximum=4, bytes_limit=1024)
        with patch.dict(os.environ, {"IQ_MOE_GPU_BATCH_EXPERTS": "bad"}):
            with self.assertRaises(ProjectionComputeError):
                batch_capacity(source_width=4, intermediate_width=3,
                               expert_count=2)

    def test_gpu_bases_not_silently_run_on_cpu(self):
        ops = ProjectionOps(mode="cpu")
        with self.assertRaises(ProjectionComputeError):
            device_bases(ops, input_inverse=np.eye(4),
                         output_map=np.eye(4), source_width=4)

    def test_expert_spec_is_immutable(self):
        item = ExpertSpec(1, 7)
        self.assertEqual((item.slot, item.donor_id), (1, 7))


@unittest.skipUnless(torch.cuda.is_available(), "real CUDA device required")
class GPUIntegrationTests(unittest.TestCase):
    def test_full_sparse_stage_batched_cuda_matches_cpu_and_resumes(self):
        source, stage, config, inspector = _donor(sparse=True, bf16=True)
        with tempfile.TemporaryDirectory() as d:
            cpu_dir = Path(d) / "cpu"
            gpu_dir = Path(d) / "gpu"
            cpu_dir.mkdir()
            gpu_dir.mkdir()
            with patch.dict(os.environ, {"IQ_PROJECTION_DEVICE": "cpu"}):
                cpu_index = {}
                save_incremental_moe_stage(
                    source=source, inspector=inspector, stage=stage,
                    config=config, output_dir=cpu_dir, weight_map=cpu_index,
                )
            source.read_keys.clear()
            with patch.dict(os.environ, {
                "IQ_PROJECTION_DEVICE": "cuda",
                "IQ_MOE_GPU_BATCH_EXPERTS": "2",
            }):
                gpu_index = {}
                save_incremental_moe_stage(
                    source=source, inspector=inspector, stage=stage,
                    config=config, output_dir=gpu_dir, weight_map=gpu_index,
                )
                self.assertEqual(gpu_index, cpu_index)
                self.assertEqual(len(gpu_index), 15)
                for key, shard in cpu_index.items():
                    want = load_file(str(cpu_dir / shard))[key]
                    got = load_file(str(gpu_dir / gpu_index[key]))[key]
                    torch.testing.assert_close(
                        got.float(), want.float(), rtol=0.015, atol=0.015,
                        msg=lambda msg: f"{key}: {msg}",
                    )
                self.assertFalse(any(".experts.6." in x for x in source.read_keys))
                source.read_keys.clear()
                resume_index = {}
                save_incremental_moe_stage(
                    source=source, inspector=inspector, stage=stage,
                    config=config, output_dir=gpu_dir, weight_map=resume_index,
                    resume_existing=True,
                )
                self.assertEqual(resume_index, gpu_index)
                self.assertEqual(source.read_keys, [])

    def test_batch_oom_halves_without_losing_expert_slots(self):
        source, stage, config, inspector = _donor(sparse=True, bf16=True)
        import iq_transfer.glm53_gpu_batch as gpu_batch
        original = gpu_batch.project_routed_batch
        attempts = []

        def maybe_oom(**kwargs):
            count = len(kwargs["specs"])
            attempts.append(count)
            if count > 1:
                raise torch.cuda.OutOfMemoryError("fixture: batch too big")
            return original(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {
                "IQ_PROJECTION_DEVICE": "cuda",
                "IQ_MOE_GPU_BATCH_EXPERTS": "2",
            }), patch.object(gpu_batch, "project_routed_batch", side_effect=maybe_oom):
                index = {}
                save_incremental_moe_stage(
                    source=source, inspector=inspector, stage=stage,
                    config=config, output_dir=Path(tmp), weight_map=index,
                )
                self.assertEqual(attempts, [2, 1, 1])
                self.assertEqual(len(index), 15)


if __name__ == "__main__":
    unittest.main()
