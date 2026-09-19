from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from iq_model import DoRALinear, IQForCausalLM, IQModelConfig, install_dora
from iq_training import load_checkpoint, save_checkpoint
from iq_transfer import ApplyError, ParameterUpdate, apply_parameter_updates


class DenseRecipientTests(unittest.TestCase):
    def config(self) -> IQModelConfig:
        return IQModelConfig(
            vocab_size=37,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            rope_theta=10000.0,
            initializer_range=0.02,
        )

    def test_forward_backward_and_causality(self):
        torch.manual_seed(1)
        model = IQForCausalLM(self.config())
        ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        out = model(ids, labels=ids, return_hidden_states=True)
        self.assertEqual(tuple(out.logits.shape), (2, 5, 37))
        self.assertEqual(tuple(out.hidden_states.shape), (2, 5, 16))
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(grads)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

        model.eval()
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[1, 2, 3, 4, 9]])
        with torch.no_grad():
            logits_a = model(a).logits
            logits_b = model(b).logits
        self.assertTrue(torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6, rtol=1e-5))

    def test_dora_zero_delta_preserves_base_and_merge_parity(self):
        torch.manual_seed(2)
        base = torch.nn.Linear(8, 6, bias=False)
        dora = DoRALinear(base, rank=2)
        x = torch.randn(4, 8)
        dora.eval()
        expected = base(x)
        actual = dora(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

        with torch.no_grad():
            dora.lora_B.normal_(std=0.02)
        expected = dora(x)
        merged = dora.merge_into_base_()
        actual = merged(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_install_dora_replaces_selected_linear_only(self):
        model = IQForCausalLM(self.config())
        installed = install_dora(model, ["blocks.0.attn.q_proj", "blocks.0.mlp.up_proj"], rank=2)
        self.assertEqual(installed, ("blocks.0.attn.q_proj", "blocks.0.mlp.up_proj"))
        self.assertIsInstance(model.blocks[0].attn.q_proj, DoRALinear)
        self.assertIsInstance(model.blocks[0].mlp.up_proj, DoRALinear)
        self.assertIsInstance(model.blocks[0].attn.k_proj, torch.nn.Linear)

    def test_parameter_updates_validate_atomically(self):
        model = IQForCausalLM(self.config())
        target = "blocks.0.attn.q_proj.weight"
        original = model.blocks[0].attn.q_proj.weight.detach().clone()
        good = torch.ones_like(original)
        bad = torch.ones(3, 3)
        with self.assertRaises(ApplyError):
            apply_parameter_updates(
                model,
                [
                    ParameterUpdate(target, good, "transported phi q"),
                    ParameterUpdate("blocks.0.attn.k_proj.weight", bad, "broken"),
                ],
            )
        self.assertTrue(torch.equal(model.blocks[0].attn.q_proj.weight, original))
        changed = apply_parameter_updates(model, [ParameterUpdate(target, good, "transported phi q")])
        self.assertEqual(changed, (target,))
        self.assertTrue(torch.equal(model.blocks[0].attn.q_proj.weight, good))

    def test_checkpoint_resume_reproduces_next_loss(self):
        torch.manual_seed(3)
        cfg = self.config()
        model = IQForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

        optimizer.zero_grad(set_to_none=True)
        loss1 = model(ids, labels=ids).loss
        loss1.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                model_config_hash=cfg.fingerprint,
                step=1,
                consumed_tokens=ids.numel(),
                extra_state={"tag": "unit"},
            )
            optimizer.zero_grad(set_to_none=True)
            baseline_loss = model(ids, labels=ids).loss.detach().clone()

            restored = IQForCausalLM(cfg)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            metadata, extra = load_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                expected_model_config_hash=cfg.fingerprint,
            )
            resumed_loss = restored(ids, labels=ids).loss.detach()
            self.assertEqual(metadata.step, 1)
            self.assertEqual(extra["tag"], "unit")
            self.assertTrue(torch.equal(baseline_loss, resumed_loss))


    def test_checkpoint_supports_tied_embeddings(self):
        torch.manual_seed(4)
        cfg = IQModelConfig(
            vocab_size=37,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            tie_word_embeddings=True,
        )
        model = IQForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tied"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                model_config_hash=cfg.fingerprint,
                step=0,
                consumed_tokens=0,
            )
            restored = IQForCausalLM(cfg)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            load_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                expected_model_config_hash=cfg.fingerprint,
            )
        self.assertIs(restored.embed_tokens.weight, restored.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
