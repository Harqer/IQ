#!/usr/bin/env python3
"""Explicit-cost capture of genuine hosted GLM-5.3 chat responses.

Writes claim-labeled GLM teacher JSONL for IQ response distillation.
No network request is made unless --execute is supplied. Provider-reported
model identity and request IDs are retained; API model revision is NOT verified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from iq_training.behavior_transfer import SUPPORTED_GLM_TEACHERS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True, help="JSONL, each line {'prompt': '...'}")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=sorted(SUPPORTED_GLM_TEACHERS), default="glm-5.3")
    p.add_argument("--teacher-revision", required=True, help="claimed 40-char HF snapshot revision")
    p.add_argument("--endpoint", default="https://api.z.ai/api/paas/v4/chat/completions")
    p.add_argument("--api-key-env", default="ZAI_API_KEY")
    p.add_argument("--max-prompts", type=int, default=16)
    p.add_argument("--max-output-tokens", type=int, default=128)
    p.add_argument("--execute", action="store_true", help="Actually call a paid provider")
    args = p.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        p.error("output exists; refusing to overwrite or silently retry paid calls")
    if not (1 <= args.max_prompts <= 100 and 1 <= args.max_output_tokens <= 1024):
        p.error("max-prompts must be 1..100 and max-output-tokens 1..1024")
    if len(args.teacher_revision) != 40 or any(
        x not in "0123456789abcdef" for x in args.teacher_revision.lower()
    ):
        p.error("teacher-revision must be a pinned git SHA (provider identity unverified)")
    endpoint = urlparse(args.endpoint)
    if endpoint.scheme != "https" and not (
        endpoint.scheme == "http" and endpoint.hostname in ("127.0.0.1", "localhost")
    ):
        p.error("only HTTPS or a local HTTP inference endpoint is permitted")
    try:
        with args.prompts.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    except (OSError, ValueError) as exc:
        p.error(f"invalid prompts file: {exc}")
    if (
        not 2 <= len(records) <= args.max_prompts
        or any(not isinstance(row, dict)
               or set(row) != {"prompt"}
               or not isinstance(row["prompt"], str)
               or not row["prompt"].strip() for row in records)
    ):
        p.error("supply 2..max-prompts nonempty {'prompt': string} JSONL rows")
    if len({row["prompt"] for row in records}) != len(records):
        p.error("duplicate prompt would contaminate the hold-out split")
    print(f"Teacher source: {args.model}; prompts: {len(records)}")
    print("Provider's exact checkpoint revision will NOT be authenticated by this API.")
    if not args.execute:
        print("DRY RUN. No requests sent or charged. Add --execute to make API calls.")
        return 0
    secret = os.environ.get(args.api_key_env)
    if not secret:
        p.error(f"set {args.api_key_env} in your shell; never paste it into a command")
    output = args.out.with_suffix(".partial")
    try:
        with output.open("x", encoding="utf-8") as stream:
            for row in records:
                payload = json.dumps({
                    "model": args.model,
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "max_tokens": args.max_output_tokens,
                    "temperature": 0,
                }).encode("utf-8")
                req = Request(
                    args.endpoint,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlopen(req, timeout=180) as response:
                    raw = json.load(response)
                actual = raw.get("model")
                if actual != args.model:
                    raise ValueError(f"API returned unexpected model {actual!r}")
                request_id = raw.get("id")
                if not isinstance(request_id, str) or not request_id.strip():
                    raise ValueError("provider omitted real request ID")
                text = raw["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("provider returned empty/nontext response")
                record = {
                    "prompt": row["prompt"],
                    "response": text,
                    "teacher_model": actual,
                    "teacher_revision": args.teacher_revision,
                    "request_id": request_id,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
        output.replace(args.out)
        print(f"Wrote {len(records)} real provider responses to {args.out}.")
        print("Model/revision attestation is not cryptographically verified.")
        return 0
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Teacher capture failed: {exc}", file=sys.stderr)
        print(f"Any recorded successful responses remain at {output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
