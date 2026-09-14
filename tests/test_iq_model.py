from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE, "torch and transformers are required")
class IQArchitectureTests(unittest.TestCase):
    def tiny_configs(self):
        from transformers.models.phi3.configuration_phi3 import Phi3Config
        from iq_model import IQArchitectureConfig

        iq = IQArchitectureConfig(
            prelude_layers=1,
            recurrent_layers=2,
            recurrent_passes=2,
            coda_layers=1,
        )
        phi = Phi3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=iq.effective_depth,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            original_max_position_embeddings=64,
            attention_dropout=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
        phi._attn_implementation = "eager"
        return phi, iq

    def test_default_topology_matches_phi4_mini_depth(self):
        from iq_model import IQArchitectureConfig

        cfg = IQArchitectureConfig()
        self.assertEqual(cfg.physical_layers, 16)
        self.assertEqual(cfg.effective_depth, 32)
        self.assertEqual(cfg.teacher_layer_for_prelude(3), 3)
        self.assertEqual(cfg.teacher_layer_for_core(0, 0), 4)
        self.assertEqual(cfg.teacher_layer_for_core(1, 0), 12)
        self.assertEqual(cfg.teacher_layer_for_core(2, 7), 27)
        self.assertEqual(cfg.teacher_layer_for_coda(0), 28)
        self.assertEqual(cfg.teacher_layer_for_coda(3), 31)

    def test_model_uses_physical_recurrent_core_not_duplicated_passes(self):
        from iq_model import IQRecurrentPhiModel

        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq)
        self.assertEqual(model.physical_depth, 4)
        self.assertEqual(model.effective_depth, 6)
        self.assertEqual(len(model.prelude), 1)
        self.assertEqual(model.reasoning_core.physical_depth, 2)
        self.assertEqual(model.reasoning_core.effective_depth, 4)
        self.assertEqual(len(model.coda), 1)
        self.assertEqual(tuple(model.reasoning_core.pass_embeddings.shape), (2, 32))

    def test_ffn_is_explicit_phi_compatible_swiglu(self):
        from iq_model import IQRecurrentPhiModel
        from iq_model.components import PhiCompatibleSwiGLU

        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq)
        ffn = model.reasoning_core.blocks[0].feed_forward
        self.assertIsInstance(ffn, PhiCompatibleSwiGLU)
        self.assertEqual(tuple(ffn.gate_up_proj.weight.shape), (2 * phi.intermediate_size, phi.hidden_size))
        self.assertEqual(tuple(ffn.down_proj.weight.shape), (phi.hidden_size, phi.intermediate_size))

    def test_forward_runs_all_recurrent_passes(self):
        import torch
        from iq_model import IQRecurrentPhiModel

        torch.manual_seed(4)
        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq).eval()
        ids = torch.randint(0, phi.vocab_size, (2, 11))

        with torch.no_grad():
            output = model(ids, capture_core_passes=True)

        self.assertEqual(tuple(output.logits.shape), (2, 11, phi.vocab_size))
        self.assertEqual(tuple(output.hidden_states.shape), (2, 11, phi.hidden_size))
        self.assertEqual(len(output.core_pass_states), iq.recurrent_passes)
        for state in output.core_pass_states:
            self.assertEqual(tuple(state.shape), (2, 11, phi.hidden_size))

    def test_pass_embeddings_are_zero_initialized_for_transfer_safety(self):
        import torch
        from iq_model import IQRecurrentPhiModel

        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq)
        embeddings = model.reasoning_core.pass_embeddings
        self.assertTrue(torch.equal(embeddings, torch.zeros_like(embeddings)))

    def test_unsupported_research_features_fail_explicitly(self):
        from iq_model import IQArchitectureConfig, IQRecurrentPhiModel

        phi, _ = self.tiny_configs()
        with self.assertRaises(NotImplementedError):
            IQRecurrentPhiModel(
                phi,
                IQArchitectureConfig(
                    prelude_layers=1,
                    recurrent_layers=2,
                    recurrent_passes=2,
                    coda_layers=1,
                    latent_slots=4,
                ),
            )


if __name__ == "__main__":
    unittest.main()
