from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from iq_model import IQForCausalLM, IQModelConfig, install_dora
from iq_training import (
    OptimizerConfig,
    TrainingError,
    TrainStepConfig,
    build_optimizer,
    classify_parameters,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


class TrainingTests(unittest.TestCase):
    def config(self):
        return IQModelConfig(
            vocab_size=41,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )

    def test_optimizer_coverage_is_exact_and_semantic(self):
        model = IQForCausalLM(self.config())
        coverage = classify_parameters(model)
        self.assertIn("blocks.0.attn.q_proj.weight", coverage.muon)
        self.assertIn("blocks.0.mlp.down_proj.weight", coverage.muon)
        self.assertIn("embed_tokens.weight", coverage.adamw)
        self.assertIn("lm_head.weight", coverage.adamw)
        self.assertIn("blocks.0.input_norm.weight", coverage.adamw)
        expected = sorted(name for name, p in model.named_parameters() if p.requires_grad)
        self.assertEqual(sorted(coverage.trainable), expected)

    def test_dora_frozen_base_is_excluded_but_factors_are_covered(self):
        model = IQForCausalLM(self.config())
        install_dora(model, ["blocks.0.attn.q_proj"], rank=2, freeze_base=True)
        coverage = classify_parameters(model)
        self.assertIn("blocks.0.attn.q_proj.base.weight", coverage.frozen)
        self.assertIn("blocks.0.attn.q_proj.lora_A", coverage.muon)
        self.assertIn("blocks.0.attn.q_proj.lora_B", coverage.muon)
        self.assertIn("blocks.0.attn.q_proj.magnitude", coverage.adamw)

    def test_train_step_updates_and_supports_accumulation(self):
        torch.manual_seed(10)
        model = IQForCausalLM(self.config())
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0.0))
        before = model.blocks[0].attn.q_proj.weight.detach().clone()
        batches = [
            {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])},
            {"input_ids": torch.tensor([[5, 4, 3, 2, 1]])},
        ]
        metrics = train_step(
            model,
            optimizer,
            batches,
            TrainStepConfig(gradient_accumulation_steps=2, max_grad_norm=1.0),
        )
        self.assertTrue(metrics.loss > 0)
        self.assertTrue(metrics.grad_norm >= 0)
        self.assertEqual(metrics.microbatches, 2)
        self.assertEqual(metrics.tokens, 10)
        self.assertFalse(torch.equal(before, model.blocks[0].attn.q_proj.weight))

        with self.assertRaises(TrainingError):
            train_step(
                model,
                optimizer,
                batches[:1],
                TrainStepConfig(gradient_accumulation_steps=2),
            )

    def test_hybrid_optimizer_checkpoint_round_trip(self):
        torch.manual_seed(11)
        cfg = self.config()
        model = IQForCausalLM(cfg)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, weight_decay=0.0))
        batch = [{"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}]
        train_step(model, optimizer, batch)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hybrid"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                model_config_hash=cfg.fingerprint,
                step=1,
                consumed_tokens=5,
            )
            restored = IQForCausalLM(cfg)
            restored_optimizer = build_optimizer(restored, OptimizerConfig(lr=1e-3, weight_decay=0.0))
            metadata, _ = load_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                expected_model_config_hash=cfg.fingerprint,
            )
        self.assertEqual(metadata.step, 1)
        self.assertEqual(optimizer.state_dict()["coverage"], restored_optimizer.state_dict()["coverage"])


if __name__ == "__main__":
    unittest.main()
