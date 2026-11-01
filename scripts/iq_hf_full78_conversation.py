#!/usr/bin/env python3
"""H200 full78 IQ scripted multi-turn conversation test (read-only checkpoint).

Loads the same full 235-layer IQ model once and pages only selected real routed
experts into CUDA per forward. Uses full-prefix recomputation for every token:
no SISO Mamba fallback, fake responses, fixed answer strings, or truncation.
A saved JSON list of user prompts is run as ONE conversation with prior turns.

This script does NOT start/cancel/bill a Hugging Face Job. Run it only inside
a user-authorized HF Job with read-only /checkpoints and /tokenizer mounts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

from iq_model.hybrid import IQHybridForCausalLM
from scripts.iq_hf_full78_sparse_inference import (
    ExpertStreamer, REQUIRED_DEVICE_MEMORY_GIB, load_base_weights,
    partition_weight_map, read_checkpoint, validate_tokenizer,
)

MAX_TURNS = 4
MAX_CONTEXT_TOKENS = 128
MAX_NEW_TOKENS = 16
MAX_ACTIVE_EXPERTS = 256


def load_turns(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_TURNS:
        raise ValueError("Conversation must be a JSON list with 1..4 user messages")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in data):
        raise ValueError("Every user message must be nonempty text <=1000 characters")
    return data


def tokenize_conversation(tokenizer: Any, messages: list[dict[str, str]],
                          *, prompt_format: str, max_context_tokens: int) -> list[int]:
    """No silent truncation; raw completion is an explicit separate baseline."""
    if prompt_format == "chat":
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise ValueError("Tokenizer has no chat template; use --prompt-format transcript")
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.flatten().tolist()
        if isinstance(encoded, dict):
            encoded = encoded.get("input_ids")
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.flatten().tolist()
        if (isinstance(encoded, list) and len(encoded) == 1
            and isinstance(encoded[0], list)):
            encoded = encoded[0]
    elif prompt_format == "transcript":
        text = "\n".join(
            ("User: " if m["role"] == "user" else "Assistant: ") + m["content"]
            for m in messages
        ) + "\nAssistant:"
        encoded = tokenizer.encode(text, add_special_tokens=True)
    else:
        raise ValueError("prompt_format must be chat or transcript")
    if (not isinstance(encoded, list) or
        not all(type(x) is int and 0 <= x < len(tokenizer) for x in encoded) or
        not 1 <= len(encoded) <= max_context_tokens):
        raise ValueError(
            f"Conversation token count exceeds {max_context_tokens}; "
            "refusing to silently forget or truncate prior turns"
        )
    return encoded


@torch.inference_mode()
def generate_reply(
    tokenizer: Any, model: Any, history: list[dict[str, str]],
    *, max_new_tokens: int, max_context_tokens: int, prompt_format: str,
    device: torch.device, turn_index: int,
) -> dict[str, Any]:
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError("max_new_tokens must be 1..16")
    encoded = tokenize_conversation(
        tokenizer, history, prompt_format=prompt_format,
        max_context_tokens=max_context_tokens,
    )
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    generated: list[int] = []
    stop_ids = set()
    if isinstance(getattr(tokenizer, "eos_token_id", None), int):
        stop_ids.add(int(tokenizer.eos_token_id))
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and 0 <= eot < len(tokenizer):
        stop_ids.add(eot)
    finish_reason = "length"
    for step in range(max_new_tokens):
        # Every output token runs ALL 235 real IQ physical layers on the
        # entire current prefix, preserving native Mamba-3 MIMO state semantics.
        if input_ids.shape[1] > max_context_tokens:
            raise RuntimeError("Autoregressive context budget exceeded; no truncation")
        result = model(input_ids)
        logits = result.logits
        if (logits.shape != (1, input_ids.shape[1], len(tokenizer)) or
            not bool(torch.isfinite(logits[:, -1]).all())):
            raise RuntimeError("Full78 generated invalid logits")
        next_id = int(logits[0, -1, :].argmax().item())
        generated.append(next_id)
        print("IQ_FULL78_DIALOGUE_TOKEN=" + json.dumps({
            "turn": turn_index, "step": step + 1, "id": next_id,
            "token_text": tokenizer.decode([next_id]),
        }, ensure_ascii=False), flush=True)
        if next_id in stop_ids:
            finish_reason = "stop"
            break
        if step + 1 < max_new_tokens:
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)),
                dim=1,
            )
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "turn": turn_index, "prompt_tokens": len(encoded),
        "completion_tokens": len(generated), "finish_reason": finish_reason,
        "response": response, "ids": generated,
        "response_nonempty": bool(response.strip()),
    }


def run_conversation(tokenizer: Any, model: Any, turns: list[str], *,
                     prompt_format: str, max_context_tokens: int,
                     max_new_tokens: int, device: torch.device) -> list[dict]:
    history: list[dict[str, str]] = []
    results: list[dict] = []
    for index, question in enumerate(turns, 1):
        history.append({"role": "user", "content": question})
        output = generate_reply(
            tokenizer, model, history, max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            prompt_format=prompt_format, device=device, turn_index=index,
        )
        output["user"] = question
        results.append(output)
        history.append({"role": "assistant", "content": output["response"]})
        print("IQ_FULL78_DIALOGUE_TURN=" + json.dumps(output, ensure_ascii=False),
              flush=True)
    return results


def load_runtime(checkpoint: Path, tokenizer_root: Path,
                 *, device: torch.device) -> tuple[Any, IQHybridForCausalLM, ExpertStreamer]:
    """Construct full78 once; return a live sparse-expert model and streamer."""
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("full78 dialogue requires real CUDA")
    if torch.cuda.get_device_properties(device).total_memory < REQUIRED_DEVICE_MEMORY_GIB * 1024**3:
        raise RuntimeError("full78 dialogue requires >=120 GiB H200-class GPU")
    _, config, weight_map = read_checkpoint(checkpoint)
    base, experts = partition_weight_map(weight_map)
    tokenizer = validate_tokenizer(tokenizer_root)
    model = IQHybridForCausalLM(
        config, device=device, dtype=torch.bfloat16, sparse_routed_moe=True,
    ).eval()
    load_base_weights(model, checkpoint, base)
    streamer = ExpertStreamer(
        checkpoint, model, experts, max_active=MAX_ACTIVE_EXPERTS,
    )
    return tokenizer, model, streamer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("/checkpoints/IQ/runs/glm53-full78-local-20261025a"))
    p.add_argument("--tokenizer", type=Path, default=Path("/tokenizer"))
    p.add_argument("--conversation", type=Path,
                   default=Path("scripts/iq_full78_sample_conversation.json"))
    p.add_argument("--prompt-format", choices=("chat", "transcript"),
                   default="chat")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-context-tokens", type=int, default=96)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 1 <= args.max_context_tokens <= MAX_CONTEXT_TOKENS:
        p.error("max-context-tokens must be 1..128")
    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS:
        p.error("max-new-tokens must be 1..16")
    turns = load_turns(args.conversation)
    device = torch.device("cuda:0")
    tokenizer, model, streamer = load_runtime(
        args.checkpoint, args.tokenizer, device=device,
    )
    print("IQ_FULL78_DIALOGUE_READY=" + json.dumps({
        "physical_layers": 235, "test_turns": len(turns),
        "max_new_tokens_per_turn": args.max_new_tokens,
        "max_context_tokens": args.max_context_tokens,
        "mode": args.prompt_format, "device": str(device),
    }), flush=True)
    try:
        results = run_conversation(
            tokenizer, model, turns, prompt_format=args.prompt_format,
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens, device=device,
        )
    finally:
        streamer.close()
    summary = {
        "turns": len(results),
        "all_nonempty": all(r["response_nonempty"] for r in results),
        "total_completions": sum(r["completion_tokens"] for r in results),
        "quality_proven": False,
        "note": "Outputs are real generated text, not a proven coherent conversation",
    }
    print("IQ_FULL78_DIALOGUE_FINISHED=" + json.dumps(summary), flush=True)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError("Refusing to overwrite output transcript")
        args.output.write_text(
            json.dumps({"results": results, "summary": summary}, indent=2,
                       ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
