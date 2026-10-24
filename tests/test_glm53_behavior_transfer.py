"""CPU checks for cross-tokenizer GLM behavioral transfer data and loss."""

from __future__ import annotations

import unittest

import torch

from iq_training.behavior_transfer import (
    BehaviorTransferError,
    TeacherExample,
    build_iq_example,
    causal_response_loss,
    collate_behavior_examples,
    validate_teacher_examples,
)


REV = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def __len__(self):
        return 128256

    def __call__(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("teacher tokenizer semantics must never be used")
        return {"input_ids": [ord(x) + 10 for x in text]}


def teacher_record(prompt, response, rid):
    return {
        "prompt": prompt,
        "response": response,
        "teacher_model": "zai-org/GLM-5.3-BF16",
        "teacher_revision": REV,
        "request_id": rid,
    }


class TestBehaviorTransfer(unittest.TestCase):
    def test_disjoint_teacher_examples_and_provenance(self):
        rows = [
            teacher_record("2 + 2?", "4", "req-1"),
            teacher_record("capital of France?", "Paris", "req-2"),
        ]
        records = validate_teacher_examples(rows)
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].content_digest(), records[1].content_digest())

    def test_reject_duplicate_prompts_and_wrong_model(self):
        rows = [
            teacher_record("hello", "world", "req-1"),
            teacher_record("hello", "again", "req-2"),
        ]
        with self.assertRaisesRegex(BehaviorTransferError, "duplicate prompt"):
            validate_teacher_examples(rows)
        rows[1]["prompt"] = "goodbye"
        rows[1]["teacher_model"] = "some-other-model"
        with self.assertRaisesRegex(BehaviorTransferError, "GLM-5.3"):
            validate_teacher_examples(rows)

    def test_response_supervision_only_and_eos(self):
        item = TeacherExample.from_record(teacher_record("2+2", "4", "req-1"))
        encoded = build_iq_example(TinyTokenizer(), item, max_tokens=128)
        first_supervised = int((encoded["labels"] != -100).nonzero()[0])
        self.assertGreater(first_supervised, 1)
        self.assertTrue(bool((encoded["labels"][:first_supervised] == -100).all()))
        self.assertEqual(int(encoded["labels"][-1]), TinyTokenizer.eos_token_id)
        self.assertTrue(torch.equal(
            encoded["input_ids"][first_supervised:], encoded["labels"][first_supervised:]
        ))

    def test_reject_truncation_instead_of_changing_teacher_response(self):
        item = TeacherExample.from_record(teacher_record("text", "answer", "req-1"))
        with self.assertRaisesRegex(BehaviorTransferError, "reject instead"):
            build_iq_example(TinyTokenizer(), item, max_tokens=8)

    def test_next_token_loss_and_gradient(self):
        tokenizer = TinyTokenizer()
        examples = [
            build_iq_example(
                tokenizer,
                TeacherExample.from_record(teacher_record("a", "x", "req-1")),
                max_tokens=128,
            ),
            build_iq_example(
                tokenizer,
                TeacherExample.from_record(teacher_record("shorter", "y", "req-2")),
                max_tokens=128,
            ),
        ]
        batch = collate_behavior_examples(examples, pad_id=2)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertTrue(bool((batch["labels"][~batch["attention_mask"]] == -100).all()))
        vocab = 256
        logits = torch.randn(
            *batch["input_ids"].shape, vocab, requires_grad=True
        )
        loss = causal_response_loss(logits, batch["labels"])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_bad_labels_not_silent(self):
        logits = torch.zeros((1, 3, 10))
        labels = torch.tensor([[-100, -100, -100]])
        with self.assertRaisesRegex(BehaviorTransferError, "no supervised"):
            causal_response_loss(logits, labels)


if __name__ == "__main__":
    unittest.main()
