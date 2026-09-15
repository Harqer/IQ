from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MAMBA_AVAILABLE = importlib.util.find_spec("mamba_ssm") is not None
FLA_AVAILABLE = importlib.util.find_spec("fla") is not None


@unittest.skipUnless(
    TORCH_AVAILABLE and MAMBA_AVAILABLE and FLA_AVAILABLE,
    "torch, mamba_ssm and FLA are required",
)
class MambaV2CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for Mamba-3 MIMO/NSA tests")

    def tiny_configs(self):
        from transformers.models.phi3.configuration_phi3 import Phi3Config
        from iq_model import IQArchitectureConfig

        iq = IQArchitectureConfig(
            prelude_layers=1,
            recurrent_layers=2,
            recurrent_passes=2,
            coda_layers=1,
            core_mixer_schedule=("mamba3_mimo", "nsa"),
            mamba3_state_size=64,
            mamba3_head_dim=64,
            mamba3_expand=2,
            mamba3_mimo_rank=4,
            mamba3_chunk_size=16,
            nsa_block_size=16,
            nsa_block_count=2,
            nsa_window_size=32,
            use_latent_predictor=True,
        )
        phi = Phi3Config(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=iq.effective_depth,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            original_max_position_embeddings=128,
            attention_dropout=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
        phi._attn_implementation = "eager"
        return phi, iq

    def test_mamba3_mimo_returns_model_width(self):
        import torch
        from iq_model.mixers import MixerContext, build_mixer

        phi, iq = self.tiny_configs()
        mixer = build_mixer("mamba3_mimo", phi, iq, layer_idx=0).cuda().eval()
        hidden = torch.randn(1, 64, phi.hidden_size, device="cuda")
        context = MixerContext(None, None, None, None)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = mixer(hidden, context=context)

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertTrue(torch.isfinite(output).all())

    def test_mamba_reference_rejects_zero_padding(self):
        import torch
        from iq_model.mixers import MixerContext, build_mixer

        phi, iq = self.tiny_configs()
        mixer = build_mixer("mamba3_mimo", phi, iq, layer_idx=0).cuda().eval()
        hidden = torch.randn(1, 64, phi.hidden_size, device="cuda")
        mask = torch.ones(1, 64, device="cuda", dtype=torch.long)
        mask[:, -4:] = 0

        with self.assertRaises(ValueError):
            mixer(hidden, context=MixerContext(mask, None, None, None))

    def test_full_mamba_nsa_recurrent_forward_backward_is_finite(self):
        import torch
        from iq_model import IQRecurrentPhiModel

        torch.manual_seed(17)
        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq).cuda().train()
        ids = torch.randint(0, phi.vocab_size, (1, 64), device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(ids, capture_core_passes=True)
            loss = output.logits.float().square().mean()
            if output.latent_prediction is not None:
                loss = loss + 0.01 * output.latent_prediction.float().square().mean()

        self.assertEqual(len(output.core_pass_states), iq.recurrent_passes)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertIsNotNone(output.latent_prediction)
        self.assertTrue(torch.isfinite(output.latent_prediction).all())
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        finite_gradients = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            self.assertTrue(torch.isfinite(parameter.grad).all())
            finite_gradients += 1
        self.assertGreater(finite_gradients, 0)


if __name__ == "__main__":
    unittest.main()
