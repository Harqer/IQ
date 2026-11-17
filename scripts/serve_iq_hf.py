"""IQ full-prefix MIMO chat API served behind HF Jobs' authenticated proxy.

Expose port 8000 with HF Jobs --expose, NOT --expose-public.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer

from iq_transfer import load_complete_iq_checkpoint
from iq_observability.full_trace import IQTrace

MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 64


def build_model():
    root = Path(os.getenv("IQ_CHECKPOINT_DIR", "/checkpoints"))
    tokenizer_root = Path(os.getenv("IQ_TOKENIZER_DIR", "/tokenizer"))
    print(f"IQ: tokenizer={tokenizer_root} checkpoint={root}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True
    )
    from iq_transfer.checkpoint_preflight import (
        preflight_iq_checkpoint,
        verify_iq_tokenizer,
    )
    # Confirm source identity and index paths *before* allocating the
    # heavy hybrid model. Llama and GPT-OSS donor checkpoints use different
    # token IDs: do not hardcode a universal Llama lexical boundary.
    evidence = preflight_iq_checkpoint(root, tokenizer_dir=tokenizer_root)
    lexical_source = verify_iq_tokenizer(
        tokenizer, evidence.config, evidence.manifest
    )
    print(
        f"IQ: lexical foundation={lexical_source} "
        f"shards={evidence.shard_count} weights={evidence.weight_count}",
        flush=True,
    )
    if os.getenv("IQ_FULL_TRACE", "0") == "1":
        trace = IQTrace(os.getenv("IQ_TRACE_DIR", "/tmp/iq-traces"))
        try:
            trace.event("checkpoint.metadata", shards=evidence.shard_count,
                        weight_tensors=evidence.weight_count,
                        calibration_mode=evidence.manifest.get("calibration_mode") if evidence.manifest else "unknown")
            with trace.stage("checkpoint.weight_load"):
                model = load_complete_iq_checkpoint(
                    root, device=torch.device("cuda:0"), dtype=torch.bfloat16, telemetry=trace
                ).eval()
        finally:
            trace.close()
    else:
        model = load_complete_iq_checkpoint(
            root, device=torch.device("cuda:0"), dtype=torch.bfloat16
        ).eval()
    print("IQ: checkpoint loaded; chat endpoint ready", flush=True)
    return tokenizer, model


def _generate_impl(
    tokenizer, model, messages: list[dict], max_new_tokens: int,
    prompt_format: str = "chat", recorder: IQTrace | None = None,
) -> tuple[str, str, int, int]:
    if prompt_format == "completion":
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("raw completion requires one user message")
        input_ids = tokenizer(
            messages[0]["content"], return_tensors="pt"
        )["input_ids"]
    elif prompt_format == "chat":
        try:
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = (
                encoded if isinstance(encoded, torch.Tensor)
                else encoded["input_ids"]
            )
        except (ValueError, AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                "checkpoint tokenizer lacks a usable chat template; "
                "use prompt_format='completion' for raw text rather than "
                "silently changing the pretrained conversational format"
            ) from exc
    else:
        raise ValueError("prompt_format must be chat or completion")
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not 1 <= input_ids.shape[1] <= MAX_PROMPT_TOKENS
    ):
        raise ValueError(f"prompt must contain 1..{MAX_PROMPT_TOKENS} tokens")

    prompt_tokens = int(input_ids.shape[1])
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
    # The official pretrained Mamba donor advertises a 2,048-token
    # context. IQ's larger configured positional range is not proof
    # of coherent extrapolation without recurrent state evaluation.
    if prompt_tokens + max_new_tokens > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"requested prompt+generation exceeds validated {MAX_PROMPT_TOKENS}-token context"
        )
    input_ids = input_ids.to(model.embed_tokens.weight.device)
    generated: list[int] = []
    finish_reason = "length"
    diagnostics = os.getenv("IQ_DIAGNOSTICS", "0") == "1"
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int):
        stop_ids = {eos_ids}
    elif isinstance(eos_ids, (list, tuple, set)):
        stop_ids = {value for value in eos_ids if isinstance(value, int)}
    else:
        stop_ids = set()
    vocab = tokenizer.get_vocab()
    eot_id = vocab.get("<|eot_id|>")
    if isinstance(eot_id, int) and 0 <= eot_id < len(tokenizer):
        stop_ids.add(eot_id)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if recorder is not None:
                recorder.round_start()
            if diagnostics and not generated:
                traced: list[tuple[int, torch.Tensor]] = []
                handles = []

                def make_probe(index: int):
                    def probe(_module, _inputs, output):
                        hidden = (
                            output if isinstance(output, torch.Tensor)
                            else getattr(output, "hidden_states", None)
                        )
                        if isinstance(hidden, torch.Tensor):
                            traced.append((index, hidden.detach()[:, -1, :]))
                    return probe

                for index, layer in enumerate(model.layers):
                    handles.append(layer.register_forward_hook(make_probe(index)))
                try:
                    result = model(input_ids, return_hidden_states=True)
                finally:
                    for handle in handles:
                        handle.remove()
            else:
                result = model(input_ids)
            if recorder is not None:
                recorder.assert_complete()
                recorder.event(
                    "inference.forward", token_index=len(generated),
                    prompt_prefix_tokens=int(input_ids.shape[1]),
                    moe_outputs=len(result.expert_counts),
                    reasoning_steps=result.reasoning_steps_executed,
                )
            logits = result.logits[:, -1, :].float()
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise RuntimeError("unexpected IQ causal LM logit shape")
            if logits.shape[-1] < len(tokenizer):
                raise RuntimeError("IQ model head is smaller than tokenizer vocabulary")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("non-finite IQ logits")
            # Some foundation heads (GPT-OSS) have padded embedding/output
            # rows beyond the tokenizer's last defined token ID. Mask them
            # before sampling and diagnostics; never decode untrained IDs.
            if logits.shape[-1] > len(tokenizer):
                logits[:, len(tokenizer):] = torch.finfo(logits.dtype).min
            if diagnostics and not generated:
                probabilities = torch.softmax(logits[0], dim=-1)
                values, indices = probabilities.topk(5)
                report = {
                    "event": "IQ_FIRST_TOKEN",
                    "prompt_tokens": prompt_tokens,
                    "logit_range": [
                        float(logits[0].min().item()),
                        float(logits[0].max().item()),
                    ],
                    "logit_std": float(logits[0].std().item()),
                    "entropy": float(
                        -(probabilities * probabilities.clamp_min(1e-30).log())
                        .sum().item()
                    ),
                    "top5": [
                        {
                            "id": int(token_id),
                            "probability": float(probability),
                            "text": tokenizer.decode([int(token_id)]),
                        }
                        for token_id, probability in zip(
                            indices.tolist(), values.tolist()
                        )
                    ],
                    "layers": [
                        {
                            "index": index,
                            "rms": float(
                                hidden.float().square().mean().sqrt().item()
                            ),
                        }
                        for index, hidden in traced
                    ],
                }
                if isinstance(result.hidden_states, torch.Tensor):
                    report["last_hidden_rms"] = float(
                        result.hidden_states[0, -1]
                        .float().square().mean().sqrt().item()
                    )
                print(json.dumps(report, ensure_ascii=False), flush=True)
            if recorder is not None:
                probabilities = torch.softmax(logits[0], dim=-1)
                top_values, top_ids = probabilities.topk(5)
                recorder.event(
                    "inference.logits", token_index=len(generated),
                    minimum=float(logits.min().item()), maximum=float(logits.max().item()),
                    entropy=float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum().item()),
                    top_token_ids=top_ids.tolist(), top_probabilities=top_values.tolist(),
                )
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            generated.append(token_id)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if token_id in stop_ids:
                finish_reason = "stop"
                break
    return (tokenizer.decode(generated, skip_special_tokens=True), finish_reason, prompt_tokens, len(generated))


def generate(
    tokenizer, model, messages: list[dict], max_new_tokens: int,
    prompt_format: str = "chat", request_id: str | None = None,
) -> tuple[str, str, int, int]:
    if os.getenv("IQ_FULL_TRACE", "0") != "1":
        return _generate_impl(tokenizer, model, messages, max_new_tokens, prompt_format)
    trace = IQTrace(os.getenv("IQ_TRACE_DIR", "/tmp/iq-traces"))
    try:
        trace.attach(model)
        with trace.stage("inference.complete_request", request_id=request_id or trace.run_id):
            return _generate_impl(tokenizer, model, messages, max_new_tokens, prompt_format, trace)
    finally:
        trace.close()


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
            if self.path not in ("/chat", "/v1/chat/completions", "/chat/completions"):
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
                compatible = self.path != "/chat"
                if compatible and data.get("stream", False):
                    raise ValueError("streaming is not supported by IQ yet")
                tokens = data.get("max_tokens", 16) if compatible else data.get("max_new_tokens", 16)
                if type(tokens) is not int or not 1 <= tokens <= MAX_NEW_TOKENS:
                    raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
                prompt_format = data.get("prompt_format", "chat")
                if compatible and prompt_format != "chat":
                    raise ValueError("prompt_format applies only to legacy /chat")
                request_id = self.headers.get("X-Baseten-Request-Id") or None
                if request_id:
                    response, finish_reason, prompt_tokens, completion_tokens = generate(
                        tokenizer, model, messages, tokens, prompt_format, request_id=request_id
                    )
                else:
                    response, finish_reason, prompt_tokens, completion_tokens = generate(
                        tokenizer, model, messages, tokens, prompt_format
                    )
                if request_id:
                    print(json.dumps({"event": "iq.response", "request_id": request_id,
                                      "prompt_tokens": prompt_tokens,
                                      "completion_tokens": completion_tokens}), flush=True)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            except Exception:  # noqa: BLE001 - HTTP inference boundary
                traceback.print_exc(file=sys.stderr)
                self.respond(500, {"error": "inference failed; see job logs"})
                return
            if compatible:
                self.respond(
                    200,
                    {
                        "id": "chatcmpl-iq",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "IQ",
                        "system_fingerprint": "iq-full-prefix-mimo",
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
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
