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

        # CPU unit tests use the Phi-only control so they do not require GPU kernels.
        iq = IQArchitectureConfig(
            prelude_layers=1,
            recurrent_layers=2,
            recurrent_passes=2,
            coda_layers=1,
            core_mixer_schedule=("phi_gqa", "phi_gqa"),
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

    def test_default_topology_and_mamba_schedule(self):
        from iq_model import IQArchitectureConfig

        cfg = IQArchitectureConfig()
        self.assertEqual(cfg.physical_layers, 16)
        self.assertEqual(cfg.effective_depth, 32)
        self.assertEqual(
            cfg.core_mixer_schedule,
            (
                "mamba3_mimo",
                "mamba3_mimo",
                "nsa",
                "mamba3_mimo",
                "mamba3_mimo",
                "nsa",
                "mamba3_mimo",
                "nsa",
            ),
        )
        self.assertEqual(cfg.core_mixer_schedule.count("mamba3_mimo"), 5)
        self.assertEqual(cfg.core_mixer_schedule.count("nsa"), 3)
        self.assertTrue(cfg.use_latent_predictor)
        self.assertEqual(cfg.teacher_layer_for_core(0, 0), 4)
        self.assertEqual(cfg.teacher_layer_for_core(1, 0), 12)
        self.assertEqual(cfg.teacher_layer_for_core(2, 7), 27)

    def test_mamba3_reference_geometry(self):
        from iq_model import IQArchitectureConfig

        cfg = IQArchitectureConfig()
        self.assertEqual(cfg.mamba3_state_size, 128)
        self.assertEqual(cfg.mamba3_head_dim, 64)
        self.assertEqual(cfg.mamba3_mimo_rank, 4)
        self.assertEqual(cfg.mamba3_chunk_size * cfg.mamba3_mimo_rank, 64)
        self.assertEqual(cfg.mamba3_rope_fraction, 0.5)

    def test_dense_to_recurrent_layout_groups_three_teacher_depths_per_core_block(self):
        from iq_model import DenseToRecurrentLayout, IQArchitectureConfig

        layout = DenseToRecurrentLayout.from_config(IQArchitectureConfig())
        self.assertEqual(layout.prelude, ((0, 0), (1, 1), (2, 2), (3, 3)))
        self.assertEqual(layout.teacher_layers_for_core_block(0), (4, 12, 20))
        self.assertEqual(layout.teacher_layers_for_core_block(7), (11, 19, 27))
        self.assertEqual(layout.coda, ((0, 28), (1, 29), (2, 30), (3, 31)))

    def test_model_uses_physical_recurrent_core_not_duplicated_passes(self):
        from iq_model import IQRecurrentPhiModel

        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq)
        self.assertEqual(model.physical_depth, 4)
        self.assertEqual(model.effective_depth, 6)
        self.assertEqual(len(model.prelude), 1)
        self.assertEqual(model.reasoning_core.physical_depth, 2)
        self.assertEqual(model.reasoning_core.effective_depth, 4)
        self.assertEqual(model.reasoning_core.mixer_schedule, ("phi_gqa", "phi_gqa"))
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

    def test_explicit_gqa_matches_canonical_phi_eager_attention(self):
        import torch
        from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3RotaryEmbedding
        from iq_model.mixers import MixerContext, PhiCompatibleGQA

        torch.manual_seed(9)
        phi, _ = self.tiny_configs()
        reference = Phi3Attention(phi, layer_idx=0).eval()
        candidate = PhiCompatibleGQA(phi, layer_idx=0).eval()
        candidate.load_state_dict(reference.state_dict(), strict=True)

        hidden = torch.randn(2, 7, phi.hidden_size)
        position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
        rotary = Phi3RotaryEmbedding(phi)
        position_embeddings = rotary(hidden, position_ids)
        mask_value = torch.finfo(hidden.dtype).min
        mask = torch.full((7, 7), mask_value, dtype=hidden.dtype)
        mask = torch.triu(mask, diagonal=1).view(1, 1, 7, 7).expand(2, 1, 7, 7)

        with torch.no_grad():
            ref_out, _ = reference(
                hidden_states=hidden,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                past_key_values=None,
            )
            cand_out = candidate(
                hidden,
                context=MixerContext(
                    padding_mask=None,
                    causal_mask=mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                ),
            )

        self.assertTrue(torch.allclose(cand_out, ref_out, atol=1e-6, rtol=1e-5))

    def test_forward_runs_all_recurrent_passes_and_latent_predictor(self):
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
        self.assertEqual(tuple(output.latent_prediction.shape), (2, 11, phi.hidden_size))
        self.assertEqual(len(output.core_pass_states), iq.recurrent_passes)

    def test_pass_embeddings_are_zero_initialized_for_transfer_safety(self):
        import torch
        from iq_model import IQRecurrentPhiModel

        phi, iq = self.tiny_configs()
        model = IQRecurrentPhiModel(phi, iq)
        embeddings = model.reasoning_core.pass_embeddings
        self.assertTrue(torch.equal(embeddings, torch.zeros_like(embeddings)))

    def test_latent_nsa_cannot_alias_plain_nsa(self):
        from iq_model.mixers import UnsupportedMixerComposition, build_mixer

        phi, iq = self.tiny_configs()
        with self.assertRaises(UnsupportedMixerComposition):
            build_mixer("latent_nsa", phi, iq, layer_idx=0)

    def test_path_inside_nsa_fails_until_joint_operator_is_validated(self):
        from iq_model import IQArchitectureConfig, IQRecurrentPhiModel

        phi, _ = self.tiny_configs()
        cfg = IQArchitectureConfig(
            prelude_layers=1,
            recurrent_layers=2,
            recurrent_passes=2,
            coda_layers=1,
            core_mixer_schedule=("nsa", "phi_gqa"),
            attention_position_strategy="path",
        )
        with self.assertRaises(NotImplementedError):
            IQRecurrentPhiModel(phi, cfg)

    def test_keystone_monitor_collects_cross_task_activity(self):
        import torch
        from iq_model.keystone import KeystoneActivationMonitor

        monitor = KeystoneActivationMonitor(intermediate_size=4)
        monitor.set_task("code")
        monitor.observe(torch.tensor([[[1.0, -2.0, 0.0, 4.0]]]))
        monitor.set_task("math")
        monitor.observe(torch.tensor([[[3.0, 0.0, -2.0, 2.0]]]))
        score = monitor.cross_task_mean_abs()
        self.assertTrue(torch.allclose(score, torch.tensor([2.0, 1.0, 1.0, 3.0])))

    def test_unsupported_latent_workspace_fails_explicitly(self):
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
                    core_mixer_schedule=("phi_gqa", "phi_gqa"),
                    latent_slots=4,
                ),
            )


if __name__ == "__main__":
    unittest.main()
