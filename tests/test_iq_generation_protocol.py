"""Real CPU tensor tests for IQ HF full-prefix generation, not mocked HTTP.

No pretrained weights, GPU, paid jobs or Hugging Face downloads are involved.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from scripts.serve_iq_hf import generate


class TinyTokenizer:
    eos_token_id = [5, 7]
    vocab = {"<|eot_id|>": 5, "prompt": 1}

    def __len__(self):
        return 16

    def get_vocab(self):
        return self.vocab

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class ToyInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.calls = 0

    def forward(self, input_ids, return_hidden_states=False):
        self.calls += 1
        logits = torch.full(
            (*input_ids.shape, 16),
            -5.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, 5] = 9.0  # stop token for both chat and completion
        return SimpleNamespace(logits=logits)


class FullPrefixGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.model = ToyInferenceModel().eval()
        self.messages = [{"role": "user", "content": "Compute 2+2"}]

    def test_chat_uses_tokenizer_template_and_stops_on_eos_list(self):
        with patch.dict(os.environ, {"IQ_DIAGNOSTICS": "0"}):
            answer, finish, prompt_tokens, generated = generate(
                self.tokenizer, self.model, self.messages, 4, "chat"
            )
        self.assertEqual((finish, prompt_tokens, generated), ("stop", 3, 1))
        self.assertEqual(answer, "5")
        self.assertEqual(self.model.calls, 1)

    def test_padded_output_head_rows_are_never_generated(self):
        class PaddedModel(ToyInferenceModel):
            def forward(self, input_ids, return_hidden_states=False):
                base = super().forward(input_ids).logits
                logits = torch.full(
                    (*input_ids.shape, 20), -5.0,
                    device=input_ids.device,
                )
                logits[..., :16] = base
                logits[:, -1, 19] = 1e6  # forbidden, no tokenizer ID
                return SimpleNamespace(logits=logits)

        padded = PaddedModel().eval()
        answer, finish, _prompt, count = generate(
            self.tokenizer, padded, self.messages, 4
        )
        self.assertEqual((answer, finish, count), ("5", "stop", 1))

    def test_completion_uses_raw_text_tokenizer(self):
        result = generate(
            self.tokenizer, self.model, self.messages, 1, "completion"
        )
        self.assertEqual(result[1], "stop")

    def test_invalid_generation_budget_fails_before_model_invocation(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            generate(self.tokenizer, self.model, self.messages, 0)
        self.assertEqual(self.model.calls, 0)

    def test_context_extension_above_donor_pretraining_window_is_blocked(self):
        def large_template(*_args, **_kwargs):
            return torch.ones((1, 2047), dtype=torch.long)

        self.tokenizer.apply_chat_template = large_template
        with self.assertRaisesRegex(ValueError, "validated 2048-token context"):
            generate(self.tokenizer, self.model, self.messages, 4)
        self.assertEqual(self.model.calls, 0)

    def test_bad_template_does_not_fall_back_to_untrained_role_prefixes(self):
        def broken(*args, **kwargs):
            raise ValueError("no usable template")

        self.tokenizer.apply_chat_template = broken
        with self.assertRaisesRegex(ValueError, "lacks a usable chat template"):
            generate(self.tokenizer, self.model, self.messages, 1)
        self.assertEqual(self.model.calls, 0)


if __name__ == "__main__":
    unittest.main()
