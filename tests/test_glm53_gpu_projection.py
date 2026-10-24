import unittest
import numpy as np
import torch
from iq_transfer.glm53_gpu_projection import ProjectionOps, ProjectionComputeError


class ProjectionOpsTests(unittest.TestCase):
    def test_cpu_preserves_reference_product(self):
        rng = np.random.default_rng(137)
        a = rng.normal(size=(19,37)).astype(np.float64)
        b = rng.normal(size=(37,11)).astype(np.float64)
        ops = ProjectionOps(mode="cpu", row_block=8)
        np.testing.assert_allclose(ops.matmul(a,b,cache_right=True),a@b,rtol=0,atol=0)
        ops.close()

    def test_invalid_device_rejected(self):
        with self.assertRaises(ProjectionComputeError):
            ProjectionOps(mode="tpu")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU worker required")
    def test_cuda_bf16_checkpoint_parity(self):
        rng=np.random.default_rng(94)
        a=rng.normal(size=(257,1024)).astype(np.float32)
        b=rng.normal(size=(1024,512)).astype(np.float32)
        a=torch.from_numpy(a).to(torch.bfloat16).float().numpy()
        b=torch.from_numpy(b).to(torch.bfloat16).float().numpy()
        ops=ProjectionOps(mode="cuda",row_block=64)
        target=ops.matmul(a,b,cache_right=True)
        expected=a.astype(np.float64)@b.astype(np.float64)
        # Reduced FP32 accumulation tolerance; BF16 final checkpoint would be coarser.
        np.testing.assert_allclose(target,expected,rtol=2e-3,atol=0.0003)
        np.testing.assert_allclose(ops.matmul(a,b,cache_right=True),target,rtol=0,atol=0)
        ops.close()


if __name__ == "__main__":
    unittest.main()
