"""End-to-end CPU execution of IQ's *actual* 80-layer hybrid forward graph.

No pretrained model, GLM tokenizer download, GPU, SISO substitution, or mocked
CSA/HCA/MoE. Only the existing CUDA Mamba MIMO kernel is replaced *explicitly*
by IQ's pure Torch rank-4 MIMO mathematical oracle in the test-only CPU mode.
"""
from __future__ import annotations

import unittest

import torch

from iq_model import (
    BlockAttnResConfig,
    CompressedContextConfig,
    HybridLayerType,
    HybridModelError,
    HybridSchedule,
    IQHybridConfig,
    IQHybridForCausalLM,
    IQModelConfig,
    Mamba3MIMOConfig,
    ReasoningRecurrenceConfig,
    RoutedMoEConfig,
    StableLatentMoEConfig,
    validate_canonical_hybrid_backbone,
)
from iq_model.hybrid import (
    CompressedContextResidualLayer,
    Mamba3ResidualLayer,
)
from iq_model.mlp.latent_moe import StableLatentMoELayer
from iq_transfer.complete_transplant import canonical_complete_schedule


def tiny_canonical_config(*, schedule=None, reasoning=True):
    if schedule is None:
        schedule = canonical_complete_schedule()
    layers = len(schedule.layers)
    model = IQModelConfig(
        vocab_size=97,
        hidden_size=16,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        rope_theta=10000.0,
        residual_dropout=0.0,
        attention_dropout=0.0,
    )
    config = IQHybridConfig(
        model=model,
        schedule=schedule,
        mamba3=Mamba3MIMOConfig(
            d_model=16,
            num_layers=schedule.count(HybridLayerType.MAMBA3),
            d_state=8,
            headdim=8,
            mimo_rank=4,
            expand=2.0,
            rope_fraction=0.5,
            chunk_size=16,
            outproj_norm=False,
        ),
        moe=RoutedMoEConfig(
            hidden_size=16,
            expert_intermediate_size=16,
            num_experts=4,
            top_k=2,
            shared_expert_intermediate_size=16,
        ),
        moe_variant="stable_latent",
        stable_moe=StableLatentMoEConfig(
            hidden_size=16,
            latent_size=8,
            expert_intermediate_size=16,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        ),
        attnres=BlockAttnResConfig(
            hidden_size=16, num_layers=layers, block_size=12,
        ),
        compressed_context=CompressedContextConfig(
            hidden_size=16,
            num_attention_heads=4,
            head_dim=8,
            q_lora_rank=8,
            partial_rotary_dim=4,
            max_position_embeddings=64,
            sliding_window=3,
            csa_compress_rate=2,
            hca_compress_rate=4,
            o_groups=2,
            o_lora_rank=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            compress_rope_theta=10000.0,
        ),
        reasoning=(
            ReasoningRecurrenceConfig(
                hidden_size=16,
                state_dim=8,
                transition_hidden_dim=16,
                max_steps=2,
                min_steps=1,
                dropout=0.0,
            )
            if reasoning else None
        ),
        energy_critic=None,
    )
    validate_canonical_hybrid_backbone(config)
    return config


class RealIQFullHybridCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, *, schedule=None, reasoning=True):
        torch.manual_seed(2026)
        return IQHybridForCausalLM(
            tiny_canonical_config(schedule=schedule, reasoning=reasoning),
            dtype=torch.float32,
            device="cpu",
            test_only_cpu_reference=True,
        )

    def test_all_80_real_hybrid_layers_execute_with_loss_and_gradients(self):
        """All canonical types execute in their actual positions, including MoE."""
        model = self.make_model()
        self.assertEqual(len(model.layers), 80)
        self.assertEqual(
            [model.config.schedule.count(kind) for kind in (
                HybridLayerType.MAMBA3, HybridLayerType.CSA,
                HybridLayerType.HCA, HybridLayerType.MOE,
            )],
            [32,12,12,24],
        )
        self.assertIsNotNone(model.attnres)
        self.assertIsNotNone(model.reasoning)
        positions = {
            kind: [i for i,k in enumerate(model.config.schedule.layers) if k is kind]
            for kind in (HybridLayerType.MAMBA3, HybridLayerType.CSA,
                         HybridLayerType.HCA, HybridLayerType.MOE)
        }
        called = []
        handles = []
        for index, layer in enumerate(model.layers):
            def record(_mod, _args, _out, index=index):
                called.append(index)
            handles.append(layer.register_forward_hook(record))
        try:
            ids = torch.tensor([[4,5,6,7]],dtype=torch.long)
            out = model(
                ids,
                labels=ids,
                reasoning_context_lengths=torch.tensor([2]),
                return_hidden_states=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(called,list(range(80)), "Every scheduled layer must execute")
        self.assertEqual(out.logits.shape,(1,4,97))
        self.assertTrue(bool(torch.isfinite(out.logits).all()))
        self.assertIsNotNone(out.loss)
        self.assertTrue(bool(torch.isfinite(out.loss)))
        self.assertEqual(len(out.expert_counts),24)
        for counts in out.expert_counts:
            self.assertEqual(int(counts.sum()),4*2)
        self.assertIsNotNone(out.reasoning_state)
        self.assertEqual(out.hidden_states.shape,(1,4,16))

        # Assert the required components participate in the actual
        # language loss, not merely that their constructors exist.
        out.loss.backward()
        for kind in positions:
            for index in (positions[kind][0], positions[kind][-1]):
                layer = model.layers[index]
                self.assertIsNotNone(next(layer.parameters()).grad,
                                     f"no gradient in {kind.value} layer {index}")
                any_nonzero = any(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    and bool((p.grad.abs()>0).any())
                    for p in layer.parameters()
                )
                self.assertTrue(any_nonzero, f"zero gradient in {kind.value} layer {index}")

    def test_actual_hybrid_is_causal_across_mamba_csa_hca_and_moe(self):
        """Changing future tokens must not alter earlier outputs."""
        schedule = HybridSchedule.parse("M C E M H E M C E M")
        model = self.make_model(schedule=schedule,reasoning=False).eval()
        a = torch.tensor([[11,12,13,14]])
        b = torch.tensor([[11,12,97-1,97-2]])
        with torch.no_grad():
            y_a = model(a).logits
            y_b = model(b).logits
        torch.testing.assert_close(y_a[:,:2],y_b[:,:2],rtol=1e-5,atol=1e-5)

    def test_document_boundaries_do_not_leak_across_required_layers(self):
        """Packed recurrence resets, CSA/HCA restrict attention, MoE is per-token."""
        schedule = HybridSchedule.parse("M C E M H E M C E M")
        model = self.make_model(schedule=schedule,reasoning=False).eval()
        packed = torch.tensor([[3,4,21,22]],dtype=torch.long)
        docs = torch.tensor([[0,0,1,1]],dtype=torch.long)
        with torch.no_grad():
            combined = model(packed,document_ids=docs).logits
            solo = model(packed[:,2:]).logits
        torch.testing.assert_close(combined[:,2:],solo,rtol=2e-5,atol=2e-5)

    def test_scheduled_csa_hca_and_moe_are_not_skipped(self):
        schedule = HybridSchedule.parse("M C E M H E M")
        model = self.make_model(schedule=schedule,reasoning=False).eval()
        self.assertIsInstance(model.layers[1],CompressedContextResidualLayer)
        self.assertIsInstance(model.layers[2],StableLatentMoELayer)
        self.assertIsInstance(model.layers[4],CompressedContextResidualLayer)
        self.assertEqual(model.layers[1].attention.mode,"csa")
        self.assertEqual(model.layers[4].attention.mode,"hca")
        with torch.no_grad():
            sample = torch.tensor([[2,7,12,18,22]],dtype=torch.long)
            original = model(sample).logits
            # Zero actual CSA/HCA attention projection and real MoE weights;
            # the graph must be sensitive to changing these components.
            layer = model.layers[4]
            layer.attention.output.out_proj.weight.zero_()
            layer = model.layers[2]
            for parameter in layer.moe.parameters():
                parameter.zero_()
            changed = model(sample).logits
        self.assertGreater(
            float((original-changed).abs().max()),1e-9,
            "CSA/HCA/MoE intervention had no effect on IQ logits",
        )

    def test_cpu_reference_requires_explicit_opt_in(self):
        cfg=tiny_canonical_config(
            schedule=HybridSchedule.parse("M C E M H E M"),reasoning=False
        )
        with self.assertRaisesRegex(HybridModelError,"production CUDA"):
            IQHybridForCausalLM(cfg,device="cpu",dtype=torch.float32)
        with self.assertRaisesRegex(HybridModelError,"requires CPU"):
            IQHybridForCausalLM(
                cfg,device="cuda",dtype=torch.float32,
                test_only_cpu_reference=True,
            )


if __name__ == "__main__":
    unittest.main()
