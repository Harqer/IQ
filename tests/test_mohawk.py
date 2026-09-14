from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class MohawkTests(unittest.TestCase):
    def test_normalized_frobenius_zero_for_identical_matrices(self):
        import torch
        from iq_transfer.mohawk import normalized_frobenius_loss

        x = torch.eye(4).view(1, 1, 4, 4)
        self.assertEqual(float(normalized_frobenius_loss(x, x)), 0.0)

    def test_linear_mixer_is_causal_and_row_normalized(self):
        import torch
        from iq_transfer.linear_mixer import IQLinearAttentionMixer

        torch.manual_seed(1)
        mixer = IQLinearAttentionMixer(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        hidden = torch.randn(2, 5, 8)
        matrix = mixer.mixing_matrix(hidden)

        self.assertEqual(tuple(matrix.shape), (2, 2, 5, 5))
        upper = torch.triu(matrix, diagonal=1)
        self.assertTrue(torch.allclose(upper, torch.zeros_like(upper), atol=1e-7))
        row_sums = matrix.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6))

    def test_phi_weight_initialization_splits_fused_qkv_exactly(self):
        import torch
        from iq_transfer.linear_mixer import IQLinearAttentionMixer

        mixer = IQLinearAttentionMixer(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        fused = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
        out = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        mixer.initialize_from_phi4(fused, out)

        self.assertTrue(torch.equal(mixer.q_proj.weight, fused[:8]))
        self.assertTrue(torch.equal(mixer.k_proj.weight, fused[8:12]))
        self.assertTrue(torch.equal(mixer.v_proj.weight, fused[12:16]))
        self.assertTrue(torch.equal(mixer.o_proj.weight, out))

    def test_stage2_alignment_prefers_matching_hidden_states(self):
        import torch
        from iq_transfer.mohawk import hidden_state_alignment_loss

        torch.manual_seed(2)
        teacher = torch.randn(2, 7, 8)
        same = hidden_state_alignment_loss(teacher, teacher)
        noisy = hidden_state_alignment_loss(teacher + 0.5 * torch.randn_like(teacher), teacher)
        self.assertLess(float(same), float(noisy))


if __name__ == "__main__":
    unittest.main()
