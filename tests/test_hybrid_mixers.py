from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
FLA_AVAILABLE = importlib.util.find_spec("fla") is not None


@unittest.skipUnless(TORCH_AVAILABLE and FLA_AVAILABLE, "torch and FLA are required")
class HybridMixerConstructionTests(unittest.TestCase):
    def test_default_phi4_gdn_width_is_exact(self):
        from iq_model import IQArchitectureConfig

        cfg = IQArchitectureConfig()
        key_width = round(3072 * cfg.gdn_key_width_ratio)
        self.assertEqual(key_width, 2304)
        self.assertEqual(key_width % cfg.gdn_head_dim, 0)
        self.assertEqual(key_width // cfg.gdn_head_dim, 18)


@unittest.skipUnless(TORCH_AVAILABLE and FLA_AVAILABLE, "torch and FLA are required")
class HybridMixerCudaSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for FLA kernel smoke tests")

    def tiny_configs(self):
        from transformers.models.phi3.configuration_phi3 import Phi3Config
        from iq_model import IQArchitectureConfig

        iq = IQArchitectureConfig(
            prelude_layers=1,
            recurrent_layers=2,
            recurrent_passes=2,
            coda_layers=1,
            core_mixer_schedule=("gated_deltanet", "nsa"),
            gdn_key_width_ratio=0.75,
            gdn_head_dim=64,
            nsa_block_size=16,
            nsa_block_count=2,
            nsa_window_size=32,
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
        return phi, iq

    def test_gated_deltanet_and_nsa_return_model_width(self):
        import torch
        from iq_model.mixers import MixerContext, build_mixer

        phi, iq = self.tiny_configs()
        hidden = torch.randn(1, 64, phi.hidden_size, device="cuda", dtype=torch.float16)
        padding = torch.ones(1, 64, device="cuda", dtype=torch.long)
        context = MixerContext(
            padding_mask=padding,
            causal_mask=None,
            position_ids=None,
            position_embeddings=None,
        )

        for layer_idx, kind in enumerate(("gated_deltanet", "nsa")):
            mixer = build_mixer(kind, phi, iq, layer_idx=layer_idx).cuda().half().eval()
            with torch.no_grad():
                output = mixer(hidden, context=context)
            self.assertEqual(tuple(output.shape), tuple(hidden.shape))
            self.assertTrue(torch.isfinite(output).all())

    def test_full_hybrid_recurrent_forward_backward_is_finite(self):
        import torch
        from iq_model import IQRecurrentPhiModel

        torch.manual_seed(13)
        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq).cuda().half().train()
        ids = torch.randint(0, phi.vocab_size, (1, 64), device="cuda")

        output = model(ids, capture_core_passes=True)
        self.assertEqual(len(output.core_pass_states), iq.recurrent_passes)
        self.assertTrue(torch.isfinite(output.logits).all())

        loss = output.logits.float().square().mean()
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
