"""CPU tests for GLM lexical transplantation and teacher-logit training.

Tiny synthetic fixture only: these tests validate graph and gradient contracts,
not knowledge transfer quality on official 1.5B/GLM checkpoints.
"""
from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

from iq_model.glm_mamba3_student import (
    GLMMamba3Student, GLMMambaError, StudentGeometry, tokenizer_mapping_hash,
)
from iq_training.glm_mamba3_distillation import (
    DistillSettings, GLMLogitBatch, MambaGLMDistillationError,
    glm_to_mamba_distillation_step, student_trainable_parameters,
    token_batch_fingerprint, record_glm_teacher_last_token_logits,
)
from scripts.iq_glm53_mamba3 import (
    GLM_EMBED_KEY, GLM_HEAD_KEY, lexical_shard_paths,
)

GEOMETRY = StudentGeometry(
    glm_vocab=16, glm_hidden=12, mamba_hidden=8,
    mamba_layers=2, mimo_rank=4, state_size=3, max_context=64,
)
TEACHER_REVISION = "a" * 40


class FakeGLMTokenizer:
    eos_token_id = 15

    def get_vocab(self):
        return {f"<tok:{i}>": i for i in range(GEOMETRY.glm_vocab)}

    def __call__(self, text, *, return_tensors):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def decode(self, tokens, *, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class Mamba3(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.is_mimo = True
        self.mimo_rank = 4
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, h):
        return torch.tanh(self.linear(h))


class GatedMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, h):
        return self.linear(h)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = Mamba3()
        self.mlp = GatedMLP()


class TinySource(nn.Module):
    """Graph-shaped source; not an implementation of Mamba recurrence."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            d_model=8, n_layer=2, vocab_size=8, tie_embeddings=True,
            ssm_cfg={"layer": "Mamba3", "is_mimo": True,
                     "mimo_rank": 4, "d_state": 3},
        )
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([Block(), Block()])
        self.backbone.embedding = nn.Embedding(8, 8)
        self.backbone.norm_f = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 8, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids, num_last_tokens=0):
        h = self.backbone.embedding(input_ids)
        for block in self.backbone.layers:
            h = h + block.mixer(h) + block.mlp(h)
        h = self.backbone.norm_f(h)
        if num_last_tokens:
            h = h[:, -num_last_tokens:]
        return SimpleNamespace(logits=self.lm_head(h))

    def generate(self, input_ids, *, max_length, **kwargs):
        ids = input_ids
        while ids.shape[1] < max_length:
            next_id = self(ids, num_last_tokens=1).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=-1)
        return ids


class GLMMamba3StudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.tokenizer = FakeGLMTokenizer()
        self.mapping_sha256 = tokenizer_mapping_hash(self.tokenizer)
        self.source = TinySource()
        self.embedding = torch.randn(16, 12)
        self.head = torch.randn(16, 12)

    def student(self, *, expected=True, train_blocks=False):
        return GLMMamba3Student(
            self.source, self.embedding, self.head, self.tokenizer,
            geometry=GEOMETRY,
            expected_tokenizer_sha256=self.mapping_sha256 if expected else None,
            train_mamba_blocks=train_blocks,
        )

    def test_glm_lexical_weights_preserved_and_mamba_blocks_native(self):
        original_mimo = self.source.backbone.layers[0].mixer.linear.weight.detach().clone()
        student = self.student()
        self.assertEqual(student.core.config.vocab_size, 16)
        self.assertFalse(student.core.config.tie_embeddings)
        self.assertEqual(student.core.backbone.embedding.weight.shape, (16, 12))
        self.assertEqual(student.core.lm_head.weight.shape, (16, 12))
        torch.testing.assert_close(student.core.backbone.embedding.weight, self.embedding)
        torch.testing.assert_close(student.core.lm_head.weight, self.head)
        torch.testing.assert_close(
            student.core.backbone.layers[0].mixer.linear.weight, original_mimo)
        self.assertFalse(student.core.backbone.layers[0].mixer.linear.weight.requires_grad)
        self.assertTrue(student.core.backbone.embedding.adapter.weight.requires_grad)
        ids = torch.tensor([[1, 2, 3], [5, 7, 8]])
        self.assertEqual(tuple(student(ids).logits.shape), (2, 3, 16))
        self.assertTrue(torch.isfinite(student(ids).logits).all())

    def test_untrained_generate_uses_glm_token_ids(self):
        student = self.student()
        report = student.generate_text("sample", max_new_tokens=3)
        self.assertEqual(report["generated_tokens"], 3)
        self.assertEqual(report["glm_tokenizer_sha256"], self.mapping_sha256)
        self.assertFalse(report["capability_transfer_trained"])

    def test_wrong_lexical_geometry_and_tokenizer_hash_rejected(self):
        with self.assertRaisesRegex(GLMMambaError, "embedding shape"):
            GLMMamba3Student(
                self.source, self.embedding[:, :10], self.head, self.tokenizer,
                geometry=GEOMETRY)
        with self.assertRaisesRegex(GLMMambaError, "mapping hash"):
            GLMMamba3Student(
                self.source, self.embedding, self.head, self.tokenizer,
                geometry=GEOMETRY, expected_tokenizer_sha256="0" * 64)

    def test_mimo_rank_change_rejected(self):
        self.source.backbone.layers[1].mixer.mimo_rank = 1
        with self.assertRaisesRegex(GLMMambaError, "native mixer"):
            self.student()

    def test_ids_checked_against_glm_vocabulary_and_context(self):
        student = self.student()
        for ids, new_tokens in (
            (torch.tensor([[16]]), 1),
            (torch.tensor([[-1]]), 1),
            (torch.ones((1, 64), dtype=torch.long), 1),
            (torch.tensor([[1.2]]), 1),
            (torch.tensor([[1]]), 0),
        ):
            with self.subTest(ids=ids.tolist(), new_tokens=new_tokens):
                with self.assertRaises(GLMMambaError):
                    student.validate_input(ids, new_tokens)

    def test_distillation_updates_adapters_not_glm_weights(self):
        student = self.student()
        before_embedding = student.core.backbone.embedding.weight.detach().clone()
        before_head = student.core.lm_head.weight.detach().clone()
        original_mamba = student.core.backbone.layers[0].mixer.linear.weight.detach().clone()
        ids = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)
        tok = torch.tensor([6, 7], dtype=torch.long)
        teacher_logits = torch.randn(2, 16)
        target = GLMLogitBatch(
            teacher_logits, token_batch_fingerprint(ids), self.mapping_sha256,
            TEACHER_REVISION)
        settings = DistillSettings(TEACHER_REVISION, self.mapping_sha256)
        optimizer = torch.optim.Adam(student_trainable_parameters(student), lr=0.01)
        before_adapter = student.core.backbone.embedding.adapter.weight.detach().clone()
        metrics = glm_to_mamba_distillation_step(
            student, input_ids=ids, next_token_ids=tok, teacher=target,
            optimizer=optimizer, settings=settings)
        self.assertGreater(metrics["loss"], 0)
        self.assertFalse(torch.equal(before_adapter,
                                    student.core.backbone.embedding.adapter.weight))
        torch.testing.assert_close(before_embedding, student.core.backbone.embedding.weight)
        torch.testing.assert_close(before_head, student.core.lm_head.weight)
        torch.testing.assert_close(
            original_mamba, student.core.backbone.layers[0].mixer.linear.weight)

    def test_teacher_input_and_revision_mismatch_rejected(self):
        student = self.student()
        ids = torch.tensor([[1, 2]])
        opt = torch.optim.Adam(student_trainable_parameters(student), lr=0.01)
        settings = DistillSettings(TEACHER_REVISION, self.mapping_sha256)
        wrong = GLMLogitBatch(torch.zeros(1, 16), "0" * 64,
                              self.mapping_sha256, TEACHER_REVISION)
        with self.assertRaisesRegex(MambaGLMDistillationError, "exact input IDs"):
            glm_to_mamba_distillation_step(
                student, input_ids=ids, next_token_ids=torch.tensor([1]),
                teacher=wrong, optimizer=opt, settings=settings)
        good_hash = token_batch_fingerprint(ids)
        wrong_rev = GLMLogitBatch(torch.zeros(1, 16), good_hash,
                                  self.mapping_sha256, "b" * 40)
        with self.assertRaisesRegex(MambaGLMDistillationError, "revision mismatch"):
            glm_to_mamba_distillation_step(
                student, input_ids=ids, next_token_ids=torch.tensor([1]),
                teacher=wrong_rev, optimizer=opt, settings=settings)

    def test_unpinned_student_cannot_train(self):
        student = self.student(expected=False)
        ids = torch.tensor([[1, 2]])
        teacher = GLMLogitBatch(torch.zeros(1, 16), token_batch_fingerprint(ids),
                                 self.mapping_sha256, TEACHER_REVISION)
        optimizer = torch.optim.Adam(student_trainable_parameters(student), lr=0.01)
        with self.assertRaisesRegex(MambaGLMDistillationError, "independently pinned"):
            glm_to_mamba_distillation_step(
                student, input_ids=ids, next_token_ids=torch.tensor([1]),
                teacher=teacher, optimizer=optimizer,
                settings=DistillSettings(TEACHER_REVISION, self.mapping_sha256))


    def test_capture_real_teacher_forward_matches_input_ids(self):
        class ToyGLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="glm_moe_dsa", vocab_size=16)
                self.embed = nn.Embedding(16, 8)
                self.head = nn.Linear(8, 16)

            def forward(self, *, input_ids, use_cache, logits_to_keep):
                self.assert_call = (use_cache, logits_to_keep)
                h = self.embed(input_ids)
                return SimpleNamespace(logits=self.head(h[:, -1:]))

        teacher_model = ToyGLM()
        ids = torch.tensor([[1, 2, 3], [3, 4, 5]])
        recorded = record_glm_teacher_last_token_logits(
            teacher_model, ids, teacher_revision=TEACHER_REVISION,
            tokenizer_mapping_sha256=self.mapping_sha256, expected_vocab=16)
        self.assertEqual(tuple(recorded.logits.shape), (2, 16))
        self.assertFalse(recorded.logits.requires_grad)
        self.assertEqual(recorded.input_ids_sha256, token_batch_fingerprint(ids))
        self.assertEqual(teacher_model.assert_call, (False, 1))
        with self.assertRaisesRegex(MambaGLMDistillationError, "not GLM"):
            record_glm_teacher_last_token_logits(
                TinySource(), ids, teacher_revision=TEACHER_REVISION,
                tokenizer_mapping_sha256=self.mapping_sha256, expected_vocab=16)

    def test_shard_preflight_does_not_read_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({
                "model_type": "glm_moe_dsa", "vocab_size": 154880,
                "hidden_size": 6144, "eos_token_id": [154820, 154827, 154829],
            }))
            (root / "lexical.safetensors").touch()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {
                    GLM_EMBED_KEY: "lexical.safetensors",
                    GLM_HEAD_KEY: "lexical.safetensors",
                }}))
            resolved = lexical_shard_paths(root)
            self.assertEqual(resolved[GLM_EMBED_KEY], root / "lexical.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {GLM_EMBED_KEY: "../outside.safetensors",
                                          GLM_HEAD_KEY: "lexical.safetensors"}}))
            with self.assertRaisesRegex(GLMMambaError, "invalid indexed"):
                lexical_shard_paths(root)

    def test_cli_preflight_without_glm_weights(self):
        command = [sys.executable, "-m", "scripts.iq_glm53_mamba3", "--preflight"]
        output = subprocess.run(command, capture_output=True, text=True, check=True)
        report = json.loads(output.stdout)
        self.assertEqual(report["glm_vocab_size"], 154880)
        self.assertEqual(report["mamba_hidden"], 2048)


if __name__ == "__main__":
    unittest.main()
