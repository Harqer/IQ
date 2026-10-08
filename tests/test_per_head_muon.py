from __future__ import annotations

import unittest

import torch
from torch import nn

from iq_model import (
    CompressedContextConfig,
    CompressedSparseContextAttention,
    HeavilyCompressedContextAttention,
    IQForCausalLM,
    IQModelConfig,
)
from iq_model.attention.context_dense import DenseContextAttention
from iq_training import OptimizerConfig, build_optimizer, classify_parameters
from iq_training.per_head_muon import (
    HeadLayout,
    PerHeadMuon,
    head_adjusted_learning_rate,
    newton_schulz_zeropower,
)


class PerHeadMuonTests(unittest.TestCase):
    def test_independent_head_updates_match_explicit_newton_schulz(self):
        torch.manual_seed(140)
        original = torch.randn(6, 5)
        grad = torch.randn_like(original)
        parameter = nn.Parameter(original.clone())
        optimizer = PerHeadMuon(
            {parameter: HeadLayout(3, 2)},
            lr=0.02, momentum=0.95, nesterov=True,
            ns_steps=5, adjust_lr_fn="match_rms_adamw",
            weight_decay=0.1,
        )
        parameter.grad = grad.clone()
        optimizer.step()
        # Independent upstream Muon recurrence, per original Q/K/V head.
        buffer = torch.zeros_like(grad).lerp(grad, 0.05)
        raw = grad.lerp(buffer, 0.95)
        expected = original * (1 - 0.02 * 0.1)
        updates = []
        for head in range(3):
            block = raw[head * 2:(head + 1) * 2]
            updates.append(
                newton_schulz_zeropower(block, steps=5).float()
                * (0.02 * 0.2 * max(2, 5)**0.5)
            )
        expected -= torch.cat(updates)
        torch.testing.assert_close(parameter.detach(), expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer)

    def test_single_head_matches_official_torch_muon_for_two_updates(self):
        # With only one head, headwise Muon must collapse to the documented
        # PyTorch Muon algorithm. This is an independent implementation
        # cross-check, not an assertion using our own NS implementation twice.
        torch.manual_seed(144)
        initial = torch.randn(2, 7)
        ours_param = nn.Parameter(initial.clone())
        torch_param = nn.Parameter(initial.clone())
        ours = PerHeadMuon(
            {ours_param: HeadLayout(1, 2)},
            lr=0.01,
            weight_decay=0.02,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        reference = torch.optim.Muon(
            [torch_param],
            lr=0.01,
            weight_decay=0.02,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        for _ in range(2):
            grad = torch.randn_like(initial)
            ours_param.grad = grad.clone()
            torch_param.grad = grad.clone()
            ours.step()
            reference.step()
            torch.testing.assert_close(
                ours_param.detach(), torch_param.detach(),
                atol=1e-6, rtol=1e-6,
            )
            torch.testing.assert_close(
                ours.state[ours_param]["momentum_buffer"],
                reference.state[torch_param]["momentum_buffer"],
                atol=0, rtol=0,
            )

    def test_independent_head_is_unaffected_by_other_head_gradient(self):
        torch.manual_seed(141)
        initial = torch.randn(4, 7)
        g1 = torch.randn(4, 7)
        g2 = g1.clone()
        g2[2:] = -g1[2:]  # change head direction, not just its scale
        outputs = []
        for grad in (g1, g2):
            p = nn.Parameter(initial.clone())
            opt = PerHeadMuon(
                {p: HeadLayout(2, 2)}, lr=0.01, weight_decay=0,
                momentum=0.9, nesterov=True, ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            )
            p.grad = grad
            opt.step()
            outputs.append(p.detach().clone())
        torch.testing.assert_close(outputs[0][:2], outputs[1][:2], atol=0, rtol=0)
        self.assertFalse(torch.equal(outputs[0][2:], outputs[1][2:]))

    def test_structural_head_layouts_follow_qkv_shapes(self):
        dense_config = IQModelConfig(
            vocab_size=37, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32,
            max_position_embeddings=64,
        )
        compressed_config = CompressedContextConfig(
            hidden_size=16, num_attention_heads=4, head_dim=8,
            q_lora_rank=8, partial_rotary_dim=4,
            max_position_embeddings=64, sliding_window=3,
            csa_compress_rate=2, hca_compress_rate=4,
            o_groups=2, o_lora_rank=4,
            index_n_heads=2, index_head_dim=4,
            index_topk=2,
        )
        class ActualAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = DenseContextAttention(dense_config)
                self.csa = CompressedSparseContextAttention(compressed_config)
                self.hca = HeavilyCompressedContextAttention(compressed_config)
        model = ActualAttention()
        coverage = classify_parameters(model)
        expected_head = {
            "dense.q_proj.weight",
            "dense.k_proj.weight",
            "dense.v_proj.weight",
            "csa.q_b_proj.weight",
            "csa.index_q_proj.weight",
            "hca.q_b_proj.weight",
        }
        self.assertEqual(set(coverage.per_head), expected_head)
        self.assertIn("csa.compressor_kv_proj.weight", coverage.muon)
        self.assertIn("csa.kv_proj.weight", coverage.muon)
        self.assertIn("dense.o_proj.weight", coverage.muon)
        self.assertEqual(
            len(set(coverage.trainable)),
            len([p for p in model.parameters() if p.requires_grad]),
        )
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
        self.assertEqual(len(opt.per_head_muon.param_groups), len(expected_head))
        for group in opt.per_head_muon.param_groups:
            p = group["params"][0]
            self.assertEqual(p.shape[0], group["heads"] * group["head_dim"])

    def test_checkpoint_restores_per_head_momentum_without_layout_mismatch(self):
        torch.manual_seed(142)
        config = IQModelConfig(
            vocab_size=31, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32,
            max_position_embeddings=32,
        )
        model = IQForCausalLM(config)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
        tokens = torch.tensor([[1, 2, 3, 4]])
        loss = model(tokens, labels=tokens).loss
        loss.backward()
        opt.step()
        saved_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved_optimizer = opt.state_dict()
        self.assertEqual(saved_optimizer["schema_version"], 2)
        self.assertIsNotNone(saved_optimizer["per_head_muon"])
        self.assertTrue(saved_optimizer["coverage"]["per_head"])
        restored_model = IQForCausalLM(config)
        restored_model.load_state_dict(saved_model, strict=True)
        restored = build_optimizer(restored_model, OptimizerConfig(lr=1e-3))
        restored.load_state_dict(saved_optimizer)
        restored_optimizer_state = restored.state_dict()
        self.assertEqual(
            saved_optimizer["coverage"],
            restored_optimizer_state["coverage"],
        )
        for old_group, new_group in zip(
            opt.per_head_muon.param_groups,
            restored.per_head_muon.param_groups, strict=True
        ):
            old = old_group["params"][0]
            new = new_group["params"][0]
            torch.testing.assert_close(
                opt.per_head_muon.state[old]["momentum_buffer"],
                restored.per_head_muon.state[new]["momentum_buffer"],
                atol=0, rtol=0,
            )
        old_legacy = dict(saved_optimizer, schema_version=1)
        old_legacy.pop("per_head_muon")
        old_legacy["coverage"] = {
            key: val for key, val in old_legacy["coverage"].items()
            if key != "per_head"
        }
        with self.assertRaisesRegex(ValueError, "legacy Muon checkpoint"):
            restored.load_state_dict(old_legacy)

    def test_per_head_control_is_explicit_not_silent(self):
        config = IQModelConfig(
            vocab_size=31, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32,
            max_position_embeddings=32,
        )
        model = IQForCausalLM(config)
        standard = classify_parameters(model, per_head_qkv=False)
        canonical = classify_parameters(model, per_head_qkv=True)
        self.assertEqual(standard.per_head, ())
        self.assertIn("blocks.0.attn.q_proj.weight", standard.muon)
        self.assertIn("blocks.0.attn.q_proj.weight", canonical.per_head)
        self.assertEqual(set(standard.trainable), set(canonical.trainable))

    def test_checkpoint_rejects_mutated_head_layout(self):
        p = nn.Parameter(torch.randn(4, 6))
        optimizer = PerHeadMuon(
            {p: HeadLayout(2, 2)},
            lr=0.01, momentum=0.9, nesterov=True,
            weight_decay=0, ns_steps=5, adjust_lr_fn="match_rms_adamw"
        )
        corrupted = optimizer.state_dict()
        corrupted["param_groups"][0]["heads"] = 1
        with self.assertRaisesRegex(ValueError, "heads layout changed"):
            optimizer.load_state_dict(corrupted)
        self.assertEqual(optimizer.param_groups[0]["heads"], 2)

    def test_nonfinite_gradient_blocks_all_optimizer_components(self):
        config = IQModelConfig(
            vocab_size=31, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32,
            max_position_embeddings=32,
        )
        model = IQForCausalLM(config)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
        original = {name: p.detach().clone() for name, p in model.named_parameters()}
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)
        model.blocks[0].attn.q_proj.weight.grad[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            opt.step()
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.detach(), original[name], atol=0, rtol=0)
        self.assertFalse(opt.per_head_muon.state)

    def test_head_layout_rejects_invalid_width(self):
        with self.assertRaisesRegex(ValueError, "output width"):
            HeadLayout(3, 2).validate(torch.randn(5, 4))
        for adjust in (None, "original", "match_rms_adamw", "spectral_unclamped"):
            self.assertGreater(
                head_adjusted_learning_rate(0.01, (2, 5), adjust),
                0,
            )


if __name__ == "__main__":
    unittest.main()
