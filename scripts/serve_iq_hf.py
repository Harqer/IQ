"""IQ full-prefix MIMO chat API served behind HF Jobs' authenticated proxy.

Expose port 8000 with HF Jobs --expose, NOT --expose-public.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer

from iq_transfer import load_complete_iq_checkpoint

MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 64


def build_model():
    root = Path(os.getenv("IQ_CHECKPOINT_DIR", "/checkpoints"))
    tokenizer_root = Path(os.getenv("IQ_TOKENIZER_DIR", "/tokenizer"))
    print(f"IQ: tokenizer={tokenizer_root} checkpoint={root}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True
    )
    config = json.loads((root / "iq_config.json").read_text(encoding="utf-8"))
    if len(tokenizer) != config["model"]["vocab_size"]:
        raise RuntimeError(
            f"IQ checkpoint vocab={config['model']['vocab_size']}, "
            f"tokenizer length={len(tokenizer)}"
        )
    model = load_complete_iq_checkpoint(
        root, device=torch.device("cuda:0"), dtype=torch.bfloat16
    ).eval()
    print("IQ: checkpoint loaded; chat endpoint ready", flush=True)
    return tokenizer, model


def generate(tokenizer, model, messages: list[dict], max_new_tokens: int) -> str:
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = (
            encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
        )
    except (ValueError, AttributeError, KeyError, TypeError):
        transcript = (
            "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )
            + "\nassistant:"
        )
        input_ids = tokenizer(transcript, return_tensors="pt")["input_ids"]
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not 1 <= input_ids.shape[1] <= MAX_PROMPT_TOKENS
    ):
        raise ValueError(f"prompt must contain 1..{MAX_PROMPT_TOKENS} tokens")

    input_ids = input_ids.to("cuda:0")
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            result = model(input_ids)
            logits = result.logits[:, -1, :].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("non-finite IQ logits")
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            generated.append(token_id)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if (
                tokenizer.eos_token_id is not None
                and token_id == tokenizer.eos_token_id
            ):
                break
    return tokenizer.decode(generated, skip_special_tokens=True)


def create_handler(tokenizer, model):
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
                self.respond(200, {"status": "ready"})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/chat":
                self.respond(404, {"error": "not found"})
                return
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
                tokens = data.get("max_new_tokens", 16)
                if type(tokens) is not int or not 1 <= tokens <= MAX_NEW_TOKENS:
                    raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
                response = generate(tokenizer, model, messages, tokens)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            except Exception:  # noqa: BLE001 - HTTP inference boundary
                traceback.print_exc(file=sys.stderr)
                self.respond(500, {"error": "inference failed; see job logs"})
                return
            self.respond(200, {"response": response})

    return Handler


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("IQ inference requires CUDA")
    tokenizer, model = build_model()
    port = int(os.getenv("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), create_handler(tokenizer, model))
    print(f"IQ: listening on port {port} for authenticated /chat", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
