"""CPU-only regression for GLM-first distillation loss and freeze/provenance gates."""
from __future__ import annotations

import unittest

import torch

from iq_training.glm_first_distillation import (
    GLMFirstDistillationConfig,
    GLMFirstDistillationError,
    GLMTeacherLastTokenLogits,
    _adapter_parameters,
    glm_first_distillation_step,
)
from test_glm_first_mamba_hybrid import (
    CONTRACT,
    ToyGLM,
    ToyMambaModel,
    wrap_preloaded_official_models,
)

REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
TOKENIZER_SHA = "1" * 64


class GLMFirstDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.teacher_model = ToyGLM().eval()
        self.donor = ToyMambaModel().eval()
        self.hybrid = wrap_preloaded_official_models(
            self.teacher_model, self.donor, contract=CONTRACT
        )
        self.hybrid.enable_branches(True)
        self.input_ids = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
        self.next_token_ids = torch.tensor([7, 9], dtype=torch.long)
        with torch.inference_mode():
            logits = self.teacher_model(self.input_ids).logits[:, -1, :].detach()
        self.targets = GLMTeacherLastTokenLogits(
            logits=logits,
            teacher_revision=REVISION,
            tokenizer_mapping_sha256=TOKENIZER_SHA,
        )
        self.config = GLMFirstDistillationConfig(
            teacher_revision=REVISION,
            tokenizer_mapping_sha256=TOKENIZER_SHA,
            temperature=2.0,
            kl_weight=1.0,
            next_token_ce_weight=1.0,
        )
        self.optimizer = torch.optim.AdamW(
            _adapter_parameters(self.hybrid), lr=1e-2
        )

    def test_optimizer_step_trains_gates_but_not_pretrained_donors(self):
        base_glm_embed = self.teacher_model.model.embed_tokens.weight.detach().clone()
        base_mamba_mixer = self.donor.backbone.layers[0].mixer.proj.weight.detach().clone()
        result = glm_first_distillation_step(
            self.hybrid,
            input_ids=self.input_ids,
            next_token_ids=self.next_token_ids,
            teacher=self.targets,
            optimizer=self.optimizer,
            config=self.config,
        )
        self.assertGreater(result.loss, 0)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)).item())
        self.assertGreater(result.gate_abs_max, 0)
        self.assertGreater(result.trainable_parameters, 0)
        torch.testing.assert_close(
            base_glm_embed, self.teacher_model.model.embed_tokens.weight.detach(),
            atol=0, rtol=0
        )
        torch.testing.assert_close(
            base_mamba_mixer, self.donor.backbone.layers[0].mixer.proj.weight.detach(),
            atol=0, rtol=0
        )
        self.assertTrue(
            all(not p.requires_grad for p in self.hybrid.glm_model.parameters())
        )

    def test_mismatched_tokenizer_provenance_fails(self):
        invalid = GLMTeacherLastTokenLogits(
            self.targets.logits, REVISION, "f" * 64
        )
        with self.assertRaisesRegex(GLMFirstDistillationError, "pinned"):
            glm_first_distillation_step(
                self.hybrid, input_ids=self.input_ids,
                next_token_ids=self.next_token_ids,
                teacher=invalid, optimizer=self.optimizer,
                config=self.config,
            )

    def test_wrong_teacher_shape_and_token_ids_fail(self):
        bad = GLMTeacherLastTokenLogits(
            self.targets.logits[:, :-1], REVISION, TOKENIZER_SHA
        )
        with self.assertRaisesRegex(GLMFirstDistillationError, "teacher logits"):
            glm_first_distillation_step(
                self.hybrid, input_ids=self.input_ids,
                next_token_ids=self.next_token_ids,
                teacher=bad, optimizer=self.optimizer,
                config=self.config,
            )
        with self.assertRaisesRegex(GLMFirstDistillationError, "outside GLM"):
            glm_first_distillation_step(
                self.hybrid, input_ids=self.input_ids,
                next_token_ids=torch.tensor([CONTRACT.glm_vocab, 1]),
                teacher=self.targets, optimizer=self.optimizer,
                config=self.config,
            )

    def test_optimizer_with_pretrained_weights_is_rejected(self):
        bad_optimizer = torch.optim.SGD(
            list(_adapter_parameters(self.hybrid))
            + list(self.teacher_model.parameters())[:1],
            lr=0.01
        )
        with self.assertRaisesRegex(GLMFirstDistillationError, "exactly Mamba"):
            glm_first_distillation_step(
                self.hybrid, input_ids=self.input_ids,
                next_token_ids=self.next_token_ids,
                teacher=self.targets, optimizer=bad_optimizer,
                config=self.config,
            )

    def test_inactive_branch_fails(self):
        self.hybrid.enable_branches(False)
        with self.assertRaisesRegex(GLMFirstDistillationError, "enable all"):
            glm_first_distillation_step(
                self.hybrid, input_ids=self.input_ids,
                next_token_ids=self.next_token_ids,
                teacher=self.targets, optimizer=self.optimizer,
                config=self.config,
            )

    def test_zero_ce_weight_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "zero-gated"):
            GLMFirstDistillationConfig(
                teacher_revision=REVISION,
                tokenizer_mapping_sha256=TOKENIZER_SHA,
                next_token_ce_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
