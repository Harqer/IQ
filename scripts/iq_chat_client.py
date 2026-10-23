#!/usr/bin/env python3
"""IQ Hugging Face Jobs chat client; standard library only, no SDK mismatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def read_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    path = Path(os.environ.get("HF_TOKEN_PATH", root / "token"))
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    raise RuntimeError("Missing HF token: run 'hf auth login' or set HF_TOKEN")


def request_json(url: str, token: str, payload: dict | None, timeout: int) -> dict:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, headers=headers, data=data, method="POST" if data else "GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Endpoint unavailable: {exc.reason}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--owner", default="kofecheckmate")
    parser.add_argument("--base-url", help="Override HF Jobs proxy URL")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with the number only.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("--max-new-tokens must be 1..64")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    base = (args.base_url or f"https://{args.job_id}--8000.hf.jobs").rstrip("/")
    try:
        token = read_token()
        health = request_json(f"{base}/health", token, None, min(30, args.timeout))
        if health.get("status") != "ready":
            raise RuntimeError(f"IQ not ready: {health}")
        result = request_json(
            f"{base}/chat",
            token,
            {"messages": [{"role": "user", "content": args.prompt}],
             "max_new_tokens": args.max_new_tokens},
            args.timeout,
        )
        if not isinstance(result.get("response"), str):
            raise RuntimeError(f"Invalid IQ response schema: {result}")
        print("IQ:", result["response"])
        return 0
    except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
        print(f"IQ CLIENT ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
