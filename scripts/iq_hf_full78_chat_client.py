#!/usr/bin/env python3
"""Use Python to talk to an already-running PRIVATE full78 IQ Hugging Face Job.

Retains conversation history, supports scripted two-turn test or interactive
terminal chat, and prints REAL responses. Does not start, expose, or cancel a
paid GPU Job. Requires an authenticated private HF Job endpoint URL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import InferenceClient, get_token

from scripts.iq_hf_full78_conversation import load_turns


def validate_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Endpoint must be an authenticated HTTPS HF Job URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Endpoint must have no query or fragment")
    return url.rstrip("/")


def chat(client: InferenceClient, history: list[dict[str, str]], max_tokens: int) -> str:
    answer = client.chat.completions.create(
        messages=history, max_tokens=max_tokens, stream=False,
    ).choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("IQ returned no natural-language response")
    return answer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True,
                   help="Private HF Jobs authenticated endpoint base URL")
    p.add_argument("--conversation", type=Path, default=Path(
        "scripts/iq_full78_sample_conversation.json"))
    p.add_argument("--interactive", action="store_true",
                   help="Use a human input loop instead of the scripted turns")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 16:
        p.error("max-tokens must be 1..16 for bounded full-prefix inference")
    endpoint = validate_endpoint(args.endpoint)
    token = get_token()
    if not token:
        p.error("HF auth token missing; run hf auth login")
    client = InferenceClient(base_url=endpoint, token=token, timeout=600)
    history: list[dict[str, str]] = []
    turns = None if args.interactive else iter(load_turns(args.conversation))

    print("IQ full78 chat ready. /quit exits locally; HF Job continues until canceled.")
    while True:
        try:
            if args.interactive:
                question = input("You: ").strip()
            else:
                question = next(turns)
                print("You:", question, flush=True)
        except (EOFError, StopIteration, KeyboardInterrupt):
            break
        if question == "/quit":
            break
        if not question:
            continue
        next_history = history + [{"role": "user", "content": question}]
        response = chat(client, next_history, args.max_tokens)
        history = next_history + [{"role": "assistant", "content": response}]
        print("IQ:", response, flush=True)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            p.error("Refusing to overwrite an existing transcript")
        args.output.write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
