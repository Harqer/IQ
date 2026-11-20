"""CPU-only GLM lexical adapter training tests, tiny frozen surrogate."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from iq_training.glm53_lexical_train import (
    IQ235GLM53LexicalObjective,
    LexicalTrainingError,
    lexical_train_step,
    evaluate_lexical_holdout,
    validate_train_holdout_bytes,
)


class ToyIQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(6, 4)
        self.frozen_backbone = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 6, bias=False)

    def forward(self, input_ids, *, labels=None, attention_mask=None, **kwargs):
        hidden = self.frozen_backbone(self.embed_tokens(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def batch():
    ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids, dtype=torch.bool),
    }


class LexicalTrainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = ToyIQ()
        self.obj = IQ235GLM53LexicalObjective(
            self.model, range(5), tokenizer_mapping_sha256="TEST_ONLY",
            test_only_small_model=True, teacher_weight=0,
        )

    def test_only_embedding_and_head_trainable(self):
        self.assertEqual(
            {n for n, p in self.model.named_parameters() if p.requires_grad},
            {"embed_tokens.weight", "lm_head.weight"},
        )

    def test_causal_training_updates_only_lexical_weights(self):
        frozen = self.model.frozen_backbone.weight.detach().clone()
        original_embed = self.model.embed_tokens.weight.detach().clone()
        original_head = self.model.lm_head.weight.detach().clone()
        optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.04,
        )
        metrics = lexical_train_step(self.obj, optim, batch())
        self.assertGreater(metrics.supervised_tokens, 0)
        self.assertGreater(metrics.total_loss, 0)
        self.assertFalse(metrics.trained_checkpoint_verified)
        self.assertTrue(torch.equal(frozen, self.model.frozen_backbone.weight))
        self.assertFalse(torch.equal(original_embed, self.model.embed_tokens.weight))
        self.assertFalse(torch.equal(original_head, self.model.lm_head.weight))

    def test_teacher_distillation_requires_aligned_finite_teacher(self):
        obj = IQ235GLM53LexicalObjective(
            ToyIQ(), range(5), tokenizer_mapping_sha256="TEST_ONLY",
            test_only_small_model=True, teacher_weight=0.5,
        )
        with self.assertRaisesRegex(LexicalTrainingError, "teacher logits"):
            obj(**batch())
        b = batch()
        b["teacher_logits"] = torch.zeros(2, 4, 6)
        with self.assertRaisesRegex(LexicalTrainingError, "teacher input token IDs"):
            obj(**b)
        b["teacher_input_ids"] = b["input_ids"].clone()
        loss, metrics = obj(**b)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(metrics.teacher_kl)
        b["teacher_logits"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(LexicalTrainingError, "teacher logits"):
            obj(**b)

    def test_mismatched_teacher_token_prefix_rejected(self):
        obj = IQ235GLM53LexicalObjective(
            ToyIQ(), range(5), tokenizer_mapping_sha256="TEST_ONLY",
            test_only_small_model=True, teacher_weight=0.5,
        )
        b = batch()
        b["teacher_logits"] = torch.zeros(2, 4, 6)
        b["teacher_input_ids"] = b["input_ids"].clone()
        b["teacher_input_ids"][0, 0] = 4
        with self.assertRaisesRegex(LexicalTrainingError, "must exactly equal"):
            obj(**b)

    def test_undefined_glm_input_or_target_rejected(self):
        b = batch()
        b["input_ids"][0, 0] = 5
        with self.assertRaisesRegex(LexicalTrainingError, "undefined"):
            self.obj(**b)
        b = batch()
        b["labels"][0, 1] = 5
        with self.assertRaisesRegex(LexicalTrainingError, "undefined"):
            self.obj(**b)

    def test_mask_ignores_supervision_and_rejects_all_masked(self):
        b = batch()
        b["labels"][0, 1] = 5
        b["attention_mask"][0, 1] = False
        loss, metrics = self.obj(**b)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics.supervised_tokens, 5)
        b["attention_mask"].fill_(False)
        with self.assertRaisesRegex(LexicalTrainingError, "no supervised"):
            self.obj(**b)

    def test_reject_invalid_lexical_batch_before_backbone_forward(self):
        class CountingIQ(ToyIQ):
            def __init__(self):
                super().__init__()
                self.forward_calls = 0
            def forward(self, *args, **kwargs):
                self.forward_calls += 1
                return super().forward(*args, **kwargs)
        student = CountingIQ()
        objective = IQ235GLM53LexicalObjective(
            student, range(5), tokenizer_mapping_sha256="TEST_ONLY",
            test_only_small_model=True, teacher_weight=0.5,
        )
        bad = batch()
        bad["labels"][0, 1] = 5
        with self.assertRaisesRegex(LexicalTrainingError, "undefined"):
            objective(**bad)
        self.assertEqual(student.forward_calls, 0)
        bad = batch()
        bad["teacher_logits"] = torch.zeros(2, 4, 6)
        bad["teacher_input_ids"] = bad["input_ids"].clone()
        bad["teacher_input_ids"][0, 0] = 4
        with self.assertRaisesRegex(LexicalTrainingError, "must exactly equal"):
            objective(**bad)
        self.assertEqual(student.forward_calls, 0)

    def test_heldout_does_not_mutate_parameters(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.obj.train()
        result = evaluate_lexical_holdout(self.obj, batch())
        self.assertTrue(self.obj.training)
        self.assertFalse(result.generation_quality_verified)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))

    def test_training_holdout_exact_leakage_guard(self):
        report = validate_train_holdout_bytes([b"hello", b"goodbye"], [b"other"])
        self.assertEqual(report["heldout_samples"], 1)
        self.assertFalse(report["heldout_from_calibration_verified"])
        with self.assertRaisesRegex(LexicalTrainingError, "leakage"):
            validate_train_holdout_bytes([b"hello"], [b"hello"])
        with self.assertRaisesRegex(LexicalTrainingError, "duplicate"):
            validate_train_holdout_bytes([b"hello", b"hello"], [b"other"])
        with self.assertRaisesRegex(LexicalTrainingError, "UTF-8"):
            validate_train_holdout_bytes([b"\xff"], [b"okay"])

    def test_production_gate_refuses_unloaded_or_random_full_model(self):
        original = ToyIQ()
        with self.assertRaisesRegex(LexicalTrainingError, "full235"):
            IQ235GLM53LexicalObjective(
                original, range(5),
                tokenizer_mapping_sha256="TEST_ONLY",
            )

    def test_test_only_flag_cannot_bypass_production_model_geometry(self):
        class LargerFixture(ToyIQ):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(33, 4)
                self.lm_head = nn.Linear(4, 33, bias=False)
        with self.assertRaisesRegex(LexicalTrainingError, "tiny synthetic"):
            IQ235GLM53LexicalObjective(
                LargerFixture(), range(5), tokenizer_mapping_sha256="TEST_ONLY",
                test_only_small_model=True,
            )
        class MislabelledFullModel(ToyIQ):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Identity() for _ in range(235)])
        with self.assertRaisesRegex(LexicalTrainingError, "tiny synthetic"):
            IQ235GLM53LexicalObjective(
                MislabelledFullModel(), range(5),
                tokenizer_mapping_sha256="TEST_ONLY",
                test_only_small_model=True,
            )

    def test_production_does_not_accept_a_self_attested_integrity_dict(self):
        # The live model cannot be instantiated on CPU; inspect the actual
        # production gate with a source-level regression to prevent a future
        # boolean-flag bypass.
        from inspect import getsource
        gate = getsource(IQ235GLM53LexicalObjective.__init__)
        self.assertIn("caller-supplied source_integrity flags cannot authorize it", gate)
        self.assertNotIn('source_integrity.get("weights_injected")', gate)

    def test_tied_lexical_weights_rejected(self):
        model = ToyIQ()
        model.lm_head.weight = model.embed_tokens.weight
        with self.assertRaisesRegex(LexicalTrainingError, "untied"):
            IQ235GLM53LexicalObjective(
                model, range(5), tokenizer_mapping_sha256="TEST_ONLY",
                test_only_small_model=True,
            )


if __name__ == "__main__":
    unittest.main()
