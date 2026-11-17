#!/usr/bin/env python3
"""Read-only Baseten IQ account/availability inspection. Never deploys or runs inference.

Defaults to a zero-network dry-run. --inspect makes authenticated HTTP GETs
only, to official Baseten Management API endpoints. Never prints the key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://api.baseten.co"
MODEL_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def paths(model_id: str | None, include_instance_types: bool = False) -> list[str]:
    result = ["/v1/models"]
    if model_id is not None:
        if not MODEL_ID.fullmatch(model_id):
            raise ValueError("Model ID contains invalid characters")
        result.append(f"/v1/models/{model_id}/deployments")
    if include_instance_types:
        result.append("/v1/instance_types")
    return result


def inspect_path(path: str, token: str, *, opener=urlopen) -> dict:
    if not token or "\r" in token or "\n" in token:
        raise ValueError("An API key is required")
    if not (
        path in {"/v1/models", "/v1/instance_types"}
        or (path.startswith("/v1/models/")
            and path.endswith("/deployments")
            and MODEL_ID.fullmatch(path.removeprefix("/v1/models/").removesuffix("/deployments")))
    ):
        raise ValueError("Only known read-only management paths are permitted")
    req = Request(
        BASE + path,
        headers={"Authorization": f"Api-Key {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(req, timeout=10) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("Baseten response exceeds allowed size")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected Baseten response format")
            return payload
    except HTTPError as exc:
        # Do NOT output request headers, response bodies or arbitrary secrets.
        raise RuntimeError(f"Baseten read-only API HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Baseten read-only API unavailable: " + type(exc).__name__) from None


def summarize(path: str, result: dict) -> dict:
    """Display bounded nonsecret summary rather than dumping arbitrary API content."""
    def items(name: str) -> list[dict]:
        collection = result.get(name)
        if not isinstance(collection, list):
            return []
        summaries = []
        for item in collection[:100]:
            if not isinstance(item, dict):
                continue
            allowed = {}
            for key in ("id", "name", "status", "model_name", "deployment_id",
                        "environment", "accelerator", "type"):
                value = item.get(key)
                if type(value) is str and len(value) <= 128:
                    allowed[key] = value
            summaries.append(allowed)
        return summaries

    if path == "/v1/models":
        return {"models": items("models")}
    if path == "/v1/instance_types":
        return {"instance_types": items("instance_types")}
    return {"deployments": items("deployments")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true",
                        help="Explicitly authorize read-only Baseten API GET calls")
    parser.add_argument("--model-id", help="Inspect deployments for this exact model")
    parser.add_argument("--instance-types", action="store_true",
                        help="Inspect available hardware types (metadata only)")
    args = parser.parse_args(argv)
    try:
        requested = paths(args.model_id, args.instance_types)
        if not args.inspect:
            print(json.dumps({
                "mode": "dry-run", "compute_launched": False,
                "requests_sent": 0, "intended_GET_paths": requested,
                "note": "Use --inspect to send read-only management requests.",
            }, sort_keys=True))
            return 0
        token = os.getenv("BASETEN_API_KEY")
        if not token:
            raise ValueError("BASETEN_API_KEY missing in this environment")
        observed = {path: summarize(path, inspect_path(path, token)) for path in requested}
        print(json.dumps({
            "mode": "read-only", "compute_launched": False,
            "inference_called": False, "requests_sent": len(requested),
            "results": observed,
        }, sort_keys=True))
        return 0
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "reason": type(exc).__name__,
                          "compute_launched": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
