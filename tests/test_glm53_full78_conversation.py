"""CPU-only conversation logic checks. Never instantiate a full IQ model."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.iq_hf_full78_chat_client import validate_endpoint
from scripts.iq_hf_full78_conversation import (
    generate_reply, load_turns, run_conversation, tokenize_conversation,
)


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.histories: list[list[dict[str, str]]] = []

    def __len__(self):
        return 16

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize
        self.histories.append([dict(m) for m in messages])
        return list(range(1, 1 + 3 * len(messages)))

    def encode(self, text, *, add_special_tokens):
        self.histories.append([{"role": "transcript", "content": text}])
        return [1, 5, 6]

    def convert_tokens_to_ids(self, token):
        return 3 if token == "<|eot_id|>" else 0

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join({5: "OK", 3: ""}.get(x, "?") for x in ids)


class FakeModel:
    def __init__(self, token=5):
        self.calls = []
        self.token = token

    def __call__(self, ids):
        self.calls.append(ids.clone())
        logits = torch.zeros((1, ids.shape[1], 16))
        logits[:, -1, self.token] = 10
        return type("Output", (), {"logits": logits})()


class ConversationTests(unittest.TestCase):
    def test_single_turn_real_model_logits_required(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        result = generate_reply(
            tokenizer, model, [{"role": "user", "content": "Hello"}],
            max_new_tokens=2, max_context_tokens=32,
            prompt_format="chat", device=torch.device("cpu"), turn_index=1,
        )
        self.assertEqual(result["response"], "OKOK")
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(tuple(model.calls[1].shape), (1, 4))

    def test_eot_stops_generation(self):
        result = generate_reply(
            FakeTokenizer(), FakeModel(token=3),
            [{"role": "user", "content": "Hi"}],
            max_new_tokens=8, max_context_tokens=32,
            prompt_format="chat", device=torch.device("cpu"), turn_index=1,
        )
        self.assertEqual(result["completion_tokens"], 1)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertFalse(result["response_nonempty"])

    def test_two_turns_keep_user_and_assistant_history(self):
        tok = FakeTokenizer()
        results = run_conversation(
            tok, FakeModel(), ["Remember amber", "What was the word?"],
            max_new_tokens=1, max_context_tokens=32,
            prompt_format="chat", device=torch.device("cpu"),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([x["response"] for x in results], ["OK", "OK"])
        history = tok.histories[-1]
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user"])
        self.assertEqual(history[0]["content"], "Remember amber")
        self.assertEqual(history[1]["content"], "OK")

    def test_transcript_mode_is_explicit(self):
        tok = FakeTokenizer()
        ids = tokenize_conversation(
            tok, [{"role": "user", "content": "Hello"}],
            prompt_format="transcript", max_context_tokens=32,
        )
        self.assertEqual(ids, [1, 5, 6])
        self.assertIn("User: Hello", tok.histories[-1][0]["content"])

    def test_no_silent_prompt_truncation(self):
        with self.assertRaisesRegex(ValueError, "refusing to silently"):
            tokenize_conversation(
                FakeTokenizer(), [{"role": "user", "content": "Hello"}],
                prompt_format="chat", max_context_tokens=2,
            )

    def test_auto_prompt_format_on_base_model(self):
        tok = FakeTokenizer()
        ids = tokenize_conversation(
            tok, [{"role": "user", "content": "Hello"}],
            prompt_format="auto", max_context_tokens=32,
        )
        self.assertEqual(ids, [1, 5, 6])
        self.assertEqual(tok.histories[-1][0]["role"], "transcript")

    def test_chat_endpoint_must_be_hugging_face_https(self):
        self.assertEqual(
            validate_endpoint("https://example.hf.space"),
            "https://example.hf.space",
        )
        for url in ("http://example.hf.space", "https://evil.example.com",
                    "https://huggingface.co.attacker.com"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_endpoint(url)

    def test_conversation_json_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "turns.json"
            p.write_text(json.dumps(["Hello", "How are you?"]))
            self.assertEqual(len(load_turns(p)), 2)
            p.write_text(json.dumps(["hello"] * 5))
            with self.assertRaisesRegex(ValueError, "1..4"):
                load_turns(p)


if __name__ == "__main__":
    unittest.main()
