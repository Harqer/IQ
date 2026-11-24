"""HTTP integration smoke checks: an actual decoded response, not only logits.

Uses a fake deterministic generator so it can run without downloading 1.5B
weights; the separate bulk command tests the REAL pretrained model and adapter.
"""
from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from iq_local.server import build_handler


class Runner:
    vocab_size = 128256
    max_context_tokens = 64
    provider = None

    def __init__(self):
        self.calls = []

    def generate(self, ids, tokens, *, stop_tokens=None):
        self.calls.append((ids, tokens, stop_tokens))
        return [12366, 13]


class Tokenizer:
    eos_token_id = 128001

    def __len__(self):
        return 128256

    def convert_tokens_to_ids(self, token):
        return {"<|eot_id|>": 128009}.get(token)

    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        assert add_generation_prompt
        return torch.tensor([[128000, 1, 2, 3]], dtype=torch.long)

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        assert ids == [12366, 13]
        return "Paris."


def _post(port, payload):
    data = json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                  data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as response:
        return response.status, json.load(response)


def test_http_returns_complete_generated_text():
    runner = Runner()
    handler = build_handler(runner, Tokenizer(), "iq-mamba3-phase5e")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, output = _post(server.server_port, {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 2, "stream": False,
        })
        assert status == 200
        assert output["choices"][0]["message"]["content"] == "Paris."
        assert output["usage"]["completion_tokens"] == 2
        assert runner.calls == [([128000, 1, 2, 3], 2, {128001, 128009})]
        # Server must fail an over-budget prompt BEFORE invoking generate().
        try:
            _post(server.server_port, {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 64,
            })
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("Expected HTTP 400 for too-long generation")
        assert len(runner.calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
