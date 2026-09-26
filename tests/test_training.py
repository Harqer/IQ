from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from iq_model import IQForCausalLM, IQModelConfig, MTPConfig, install_dora
from iq_training import (
    IQPretrainingModel,
    OptimizerConfig,
    PretrainingConfigError,
    PretrainingObjectiveConfig,
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

        masked_metrics = train_step(
            model,
            optimizer,
            [
                {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
                    "document_ids": torch.tensor([[0, 0, 0, 1, 1]]),
                }
            ],
        )
        self.assertEqual(masked_metrics.tokens, 3)

        with self.assertRaises(TrainingError):
            train_step(
                model,
                optimizer,
                batches[:1],
                TrainStepConfig(gradient_accumulation_steps=2),
            )

    def test_pretraining_wrapper_adds_mtp_loss_and_preserves_optimizer_semantics(self):
        torch.manual_seed(12)
        main = IQForCausalLM(self.config())
        model = IQPretrainingModel(
            main,
            PretrainingObjectiveConfig(mtp_loss_weight=0.25),
            mtp_config=MTPConfig(num_prediction_layers=1),
        )
        coverage = classify_parameters(model)
        self.assertIn("main_model.embed_tokens.weight", coverage.adamw)
        self.assertIn("main_model.lm_head.weight", coverage.adamw)
        self.assertNotIn("main_model.embed_tokens.weight", coverage.muon)
        self.assertNotIn("main_model.lm_head.weight", coverage.muon)
        self.assertIn("mtp.layers.0.eh_proj.weight", coverage.muon)

        ids = torch.tensor([[1, 2, 3, 4, 5]])
        output = model(ids, labels=ids)
        self.assertIsNotNone(output.ntp_loss)
        self.assertIsNotNone(output.mtp_loss)
        expected = output.ntp_loss + 0.25 * output.mtp_loss
        self.assertTrue(torch.allclose(output.loss, expected))

        optimizer = build_optimizer(
            model,
            OptimizerConfig(lr=1e-3, weight_decay=0.0),
        )
        metrics = train_step(
            model,
            optimizer,
            [{"input_ids": ids}],
        )
        self.assertTrue(metrics.loss > 0)

    def test_pretraining_reasoning_boundary_and_ponder_are_explicit(self):
        class StubHybridConfig:
            reasoning = object()

            def to_dict(self):
                return {"schema_version": 1, "reasoning": {"enabled": True}}

        class StubReasoningModel(torch.nn.Module):
            def __init__(self, language_config):
                super().__init__()
                self.model_config = language_config
                self.config = StubHybridConfig()
                self.embed_tokens = torch.nn.Embedding(
                    language_config.vocab_size,
                    language_config.hidden_size,
                )
                self.lm_head = torch.nn.Linear(
                    language_config.hidden_size,
                    language_config.vocab_size,
                    bias=False,
                )
                self.ponder = torch.nn.Parameter(torch.tensor(2.0))
                self.seen_reasoning_context_lengths = None

            def forward(
                self,
                input_ids,
                *,
                labels=None,
                position_ids=None,
                attention_mask=None,
                document_ids=None,
                reasoning_context_lengths=None,
                return_hidden_states=False,
            ):
                self.seen_reasoning_context_lengths = reasoning_context_lengths
                hidden = self.embed_tokens(input_ids)
                logits = self.lm_head(hidden)
                ntp_loss = logits.float().square().mean()
                expected_steps = self.ponder.abs() + 1.0
                return SimpleNamespace(
                    logits=logits,
                    loss=ntp_loss,
                    hidden_states=hidden if return_hidden_states else None,
                    load_balance_loss=None,
                    router_z_loss=None,
                    expected_reasoning_steps=expected_steps,
                )

        main = StubReasoningModel(self.config())
        model = IQPretrainingModel(
            main,
            PretrainingObjectiveConfig(
                reasoning_ponder_loss_weight=0.25,
            ),
        )
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        context_lengths = torch.tensor([3], dtype=torch.long)

        output = model(
            ids,
            labels=ids,
            reasoning_context_lengths=context_lengths,
        )

        self.assertTrue(
            torch.equal(
                main.seen_reasoning_context_lengths,
                context_lengths,
            )
        )
        self.assertIsNotNone(output.reasoning_expected_steps)
        expected = (
            output.ntp_loss
            + 0.25 * output.reasoning_expected_steps
        )
        self.assertTrue(torch.allclose(output.loss, expected))

        no_ponder = IQPretrainingModel(
            main,
            PretrainingObjectiveConfig(),
        )
        output_no_ponder = no_ponder(
            ids,
            labels=ids,
            reasoning_context_lengths=context_lengths,
        )
        self.assertTrue(
            torch.allclose(
                output_no_ponder.loss,
                output_no_ponder.ntp_loss,
            )
        )

    def test_train_step_accepts_reasoning_context_lengths(self):
        from iq_training.train import _validate_batch

        batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ),
            "reasoning_context_lengths": torch.tensor(
                [2, 3],
                dtype=torch.long,
            ),
        }
        validated = _validate_batch(batch)
        self.assertTrue(
            torch.equal(
                validated["reasoning_context_lengths"],
                batch["reasoning_context_lengths"],
            )
        )

        with self.assertRaises(TrainingError):
            _validate_batch(
                {
                    "input_ids": batch["input_ids"],
                    "reasoning_context_lengths": torch.tensor(
                        [[2], [3]],
                        dtype=torch.long,
                    ),
                }
            )

    def test_pretraining_rejects_unavailable_moe_auxiliary_loss(self):
        model = IQPretrainingModel(
            IQForCausalLM(self.config()),
            PretrainingObjectiveConfig(
                moe_load_balance_loss_weight=0.01,
            ),
        )
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with self.assertRaises(PretrainingConfigError):
            model(ids, labels=ids)

    def test_pretraining_fingerprint_covers_mtp_and_objective_config(self):
        main_a = IQForCausalLM(self.config())
        main_b = IQForCausalLM(self.config())
        a = IQPretrainingModel(
            main_a,
            PretrainingObjectiveConfig(mtp_loss_weight=0.25),
            mtp_config=MTPConfig(num_prediction_layers=1),
        )
        b = IQPretrainingModel(
            main_b,
            PretrainingObjectiveConfig(mtp_loss_weight=0.5),
            mtp_config=MTPConfig(num_prediction_layers=1),
        )
        c = IQPretrainingModel(
            IQForCausalLM(self.config()),
            PretrainingObjectiveConfig(mtp_loss_weight=0.25),
            mtp_config=MTPConfig(num_prediction_layers=2),
        )
        d = IQPretrainingModel(
            IQForCausalLM(self.config()),
            PretrainingObjectiveConfig(
                mtp_loss_weight=0.25,
                moe_router_z_loss_weight=0.001,
            ),
            mtp_config=MTPConfig(num_prediction_layers=1),
        )
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(a.fingerprint, c.fingerprint)
        self.assertNotEqual(a.fingerprint, d.fingerprint)

    def test_mtp_pretraining_checkpoint_round_trip_preserves_shared_weights(self):
        torch.manual_seed(13)
        model = IQPretrainingModel(
            IQForCausalLM(self.config()),
            PretrainingObjectiveConfig(mtp_loss_weight=0.25),
            mtp_config=MTPConfig(num_prediction_layers=1),
        )
        optimizer = build_optimizer(
            model,
            OptimizerConfig(lr=1e-3, weight_decay=0.0),
        )
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        train_step(model, optimizer, [{"input_ids": ids}])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mtp"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                model_config_hash=model.fingerprint,
                step=1,
                consumed_tokens=ids.numel(),
            )
            restored = IQPretrainingModel(
                IQForCausalLM(self.config()),
                PretrainingObjectiveConfig(mtp_loss_weight=0.25),
                mtp_config=MTPConfig(num_prediction_layers=1),
            )
            restored_optimizer = build_optimizer(
                restored,
                OptimizerConfig(lr=1e-3, weight_decay=0.0),
            )
            metadata, _ = load_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                expected_model_config_hash=restored.fingerprint,
            )
        self.assertEqual(metadata.step, 1)
        self.assertIs(
            restored.mtp.embed_tokens,
            restored.main_model.embed_tokens,
        )
        self.assertIs(
            restored.mtp.shared_head,
            restored.main_model.lm_head,
        )
        self.assertEqual(
            optimizer.state_dict()["coverage"],
            restored_optimizer.state_dict()["coverage"],
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
