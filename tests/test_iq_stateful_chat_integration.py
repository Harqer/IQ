"""Full production chat generation integration tests without paid HF GPU jobs."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from scripts import serve_iq_hf


class TinyTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 32

    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        assert add_generation_prompt and return_tensors == "pt"
        return torch.tensor([[1, 3, 4]], dtype=torch.long)

    def convert_tokens_to_ids(self, text):
        return 2 if text == "<|eot_id|>" else 31

    def decode(self, ids, skip_special_tokens=False):
        return "".join({5: "hello", 6: " world", 2: ""}.get(i, "?") for i in ids)


class FakeEngine:
    def __init__(self, fail=False):
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(32, 4))
        self.max_context_tokens = 64
        self.state = SimpleNamespace(length=3)
        self.calls = []
        self.failed = fail
        self.reset_count = 0

    def _logits(self, index):
        scores = torch.zeros((1, 1, 32))
        scores[:, :, index] = 12.
        return scores

    def sync_prompt(self, ids):
        self.calls.append(("prefill", ids.shape[1]))
        return self._logits(5)

    def decode_token(self, ids):
        if self.failed:
            raise RuntimeError("damaged recurrent state")
        token = int(ids[0, 0])
        self.calls.append(("decode", token))
        self.state.length += 1
        return self._logits(6 if token == 5 else 2)

    def reset(self):
        self.reset_count += 1
        self.state.length = 0


class ProductionChatInferenceTests(unittest.TestCase):
    def test_real_server_generation_wires_stateful_backend_and_stream(self):
        engine = FakeEngine()
        streamed = []
        output = serve_iq_hf.generate(
            TinyTokenizer(), model=None,
            messages=[{"role": "user", "content": "Hi"}],
            max_new_tokens=128,
            prompt_format="chat", on_token=streamed.append, engine=engine,
        )
        self.assertEqual(output, ("hello world", "stop", 3, 3))
        self.assertEqual(streamed, ["hello", " world"])
        self.assertEqual(engine.calls, [
            ("prefill", 3), ("decode", 5), ("decode", 6),
        ])

    def test_partial_state_is_invalidated_after_decode_error(self):
        engine = FakeEngine(fail=True)
        with self.assertRaisesRegex(RuntimeError, "damaged recurrent"):
            serve_iq_hf.generate(
                TinyTokenizer(), model=None,
                messages=[{"role": "user", "content": "Hi"}],
                max_new_tokens=128, engine=engine,
            )
        self.assertEqual(engine.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
