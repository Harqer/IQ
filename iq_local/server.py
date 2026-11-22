"""OpenAI-compatible HTTP surface over the local streaming stack (Task 11).

`build_handler` mirrors the request/response conventions of the HF Jobs chat
server (`scripts/serve_iq_hf.py:186-273`): same path set, message validation,
request-size cap, error mapping, and compatible chat-completion response
shape — with a local model name, `/health` counters from the streaming
provider, and stop-token handling delegated to `HybridRunner.generate`.

Concurrency: ThreadingHTTPServer accepts concurrent connections, but one
StreamingWeightProvider (and its runner's streaming state) must be used by
one request at a time (`iq_local/provider.py` docstring), so inference is
serialized with a lock. Health endpoints do not take the lock. The handler
never reads or echoes HF_TOKEN; bucket auth stays inside BucketClient.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler

import torch

MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 64


def build_handler(runner, tokenizer, model_name: str):
    lock = threading.Lock()

    def encode_prompt(messages: list[dict], prompt_format: str):
        if prompt_format == "completion":
            input_ids = tokenizer(
                messages[0]["content"], return_tensors="pt"
            )["input_ids"]
        else:
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                input_ids = (
                    encoded if isinstance(encoded, torch.Tensor)
                    else encoded["input_ids"]
                )
            except (ValueError, AttributeError, KeyError, TypeError):
                transcript = (
                    "\n".join(
                        f"{message['role']}: {message['content']}"
                        for message in messages
                    )
                    + "\nassistant:"
                )
                input_ids = tokenizer(
                    transcript, return_tensors="pt"
                )["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("tokenizer must return a tensor prompt")
        ids = input_ids.reshape(-1).tolist()
        vocab = runner.model.model_config.vocab_size
        if not ids or len(ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt must contain 1..{MAX_PROMPT_TOKENS} tokens")
        if any(token < 0 or token >= vocab for token in ids):
            raise ValueError("prompt token outside the model vocabulary")
        return ids

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                counters = getattr(getattr(runner, "provider", None), "counters", {})
                self.respond(
                    200,
                    {"status": "ok", "model": model_name, "counters": dict(counters)},
                )
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/chat", "/v1/chat/completions", "/chat/completions"):
                self.respond(404, {"error": "not found"})
                return
            compatible = self.path != "/chat"
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("request must be 1..65536 bytes")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise TypeError("expected JSON object")
                messages = data.get("messages")
                if messages is None:
                    messages = [{"role": "user", "content": data.get("prompt", "")}]
                if (
                    not isinstance(messages, list)
                    or not 1 <= len(messages) <= 24
                    or any(
                        not isinstance(message, dict)
                        or message.get("role") not in ("system", "user", "assistant")
                        or not isinstance(message.get("content"), str)
                        or not message["content"]
                        for message in messages
                    )
                ):
                    raise ValueError("invalid messages: expected 1..24 chat messages")
                if compatible and data.get("stream", False):
                    raise ValueError("streaming is not supported by IQ yet")
                tokens = (
                    data.get("max_tokens", 16)
                    if compatible
                    else data.get("max_new_tokens", 16)
                )
                if type(tokens) is not int or not 1 <= tokens <= MAX_NEW_TOKENS:
                    raise ValueError(f"max_tokens must be 1..{MAX_NEW_TOKENS}")
                prompt_format = data.get("prompt_format", "chat")
                if compatible and prompt_format != "chat":
                    raise ValueError("prompt_format applies only to legacy /chat")
                prompt_ids = encode_prompt(messages, prompt_format)
                stop_ids = {tokenizer.eos_token_id}
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(eot_id, int) and 0 <= eot_id < len(tokenizer):
                    stop_ids.add(eot_id)
                stop_ids.discard(None)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            try:
                with lock:
                    generated = runner.generate(
                        prompt_ids, tokens, stop_tokens=stop_ids or None
                    )
            except Exception:  # noqa: BLE001 - HTTP inference boundary
                traceback.print_exc(file=sys.stderr)
                self.respond(500, {"error": "inference failed; see server logs"})
                return
            completion_tokens = len(generated)
            finish_reason = (
                "stop"
                if generated and generated[-1] in stop_ids
                else "length"
            )
            response = tokenizer.decode(
                generated, skip_special_tokens=True
            )
            if compatible:
                self.respond(
                    200,
                    {
                        "id": "chatcmpl-iq-local",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "system_fingerprint": "iq-local-streaming",
                        "usage": {
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": completion_tokens,
                            "total_tokens": len(prompt_ids) + completion_tokens,
                        },
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": response},
                                "finish_reason": finish_reason,
                            }
                        ],
                    },
                )
            else:
                self.respond(200, {"response": response})

    return Handler
