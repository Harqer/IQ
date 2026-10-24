#!/usr/bin/env python3
"""Probe selected GLM-5.3 safetensors headers using strict HTTP byte ranges.

Reads only 8-byte length + JSON header per shard; never fetches tensor payload.
Fail closed if Hugging Face/CDN does not honor Range (status must be 206).
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

REPO = "zai-org/GLM-5.3-BF16"
REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAX_HEADER = 32 * 1024 * 1024
RANGE_PATTERN = re.compile(r"^bytes (\\d+)-(\\d+)/(\\d+)$")


class RangeProbeError(ValueError):
    pass


def parse_range_response(
    *,
    status_code: int,
    content_range: str | None,
    payload: bytes,
    start: int,
    end: int,
) -> tuple[bytes, int]:
    if status_code != 206:
        raise RangeProbeError(
            f"remote storage did not honor Range request (HTTP {status_code}); "
            "refusing full-file download"
        )
    match = RANGE_PATTERN.fullmatch(content_range or "")
    if not match:
        raise RangeProbeError("missing/invalid Content-Range; refusing full-file download")
    actual_start, actual_end, total = (int(x) for x in match.groups())
    if actual_start != start or actual_end != end or total <= end:
        raise RangeProbeError("server returned wrong byte range")
    if len(payload) != end - start + 1:
        raise RangeProbeError("received truncated or excess payload")
    return payload, total


def parse_safetensors_header(header_size: int, header: bytes, file_size: int) -> dict:
    if not 2 <= header_size <= MAX_HEADER or len(header) != header_size:
        raise RangeProbeError("invalid safetensors JSON header length")
    try:
        decoded = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RangeProbeError("invalid safetensors header JSON") from exc
    if not isinstance(decoded, dict):
        raise RangeProbeError("invalid safetensors header object")
    data_size = file_size - 8 - header_size
    if data_size < 0:
        raise RangeProbeError("declared header exceeds file size")
    for name, item in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise RangeProbeError("invalid tensor entry")
        offs = item.get("data_offsets")
        if not isinstance(offs, list) or len(offs) != 2 or any(
            type(x) is not int for x in offs
        ):
            raise RangeProbeError("invalid tensor offsets")
        begin, finish = offs
        if begin < 0 or finish < begin or finish > data_size:
            raise RangeProbeError("tensor byte offsets extend past file boundary")
        if not isinstance(item.get("dtype"), str) or not isinstance(item.get("shape"), list):
            raise RangeProbeError("tensor dtype/shape missing")
    return decoded


def fetch_exact_range(session, url: str, start: int, end: int, *, token: str | None = None):
    if end < start or end - start + 1 > MAX_HEADER:
        raise RangeProbeError("range exceeds bounded metadata request")
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with session.get(
        url, headers=headers, stream=True, allow_redirects=True, timeout=(20, 120)
    ) as response:
        if response.status_code != 206:
            raise RangeProbeError(
                f"remote storage ignored ranged request (HTTP {response.status_code}); "
                "no tensor payload downloaded"
            )
        data = response.raw.read(end - start + 2)
        return parse_range_response(
            status_code=response.status_code,
            content_range=response.headers.get("Content-Range"),
            payload=data,
            start=start,
            end=end,
        )


def probe_shard(session, url: str, token: str | None = None):
    head, file_bytes = fetch_exact_range(session, url, 0, 7, token=token)
    header_size = struct.unpack("<Q", head)[0]
    if header_size < 2 or header_size > MAX_HEADER:
        raise RangeProbeError("unreasonable safetensors header length")
    rest, total = fetch_exact_range(
        session, url, 8, 7 + header_size, token=token
    )
    if total != file_bytes:
        raise RangeProbeError("file size changed between range reads")
    return parse_safetensors_header(header_size, rest, total), total, header_size + 8


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--max-shards", type=int, default=1, help="Probe at most this many shard headers")
    p.add_argument("--output", type=Path, help="new JSON result path")
    a = p.parse_args()
    try:
        import requests
        from huggingface_hub import get_token, hf_hub_url
        plan = json.loads(a.plan.read_text(encoding="utf-8"))
        if plan.get("source_revision") != REVISION:
            raise RangeProbeError("unpinned source revision")
        shards = plan.get("official_shards")
        if not isinstance(shards, list) or not shards:
            raise RangeProbeError("plan does not list official shards")
        if not 1 <= a.max_shards <= len(shards):
            raise RangeProbeError("invalid maximum shard count")
        if len(set(shards)) != len(shards) or any(
            not re.fullmatch(r"model-\\d{5}-of-00282\\.safetensors", name)
            for name in shards
        ):
            raise RangeProbeError("unexpected source shard filenames")
        result = []
        session = requests.Session()
        for name in shards[:a.max_shards]:
            url = hf_hub_url(REPO, name, revision=REVISION)
            header, total, header_bytes = probe_shard(session, url, get_token())
            result.append({
                "shard": name,
                "tensor_records": len(header) - int("__metadata__" in header),
                "safetensors_header_bytes": header_bytes,
                "total_file_bytes": total,
                "range_206_verified": True,
            })
        report = {
            "status": "safetensors_range_header_verified",
            "shards_probed": len(result),
            "shards_in_plan": len(shards),
            "bytes_of_file_payload_requested": 0,
            "results": result,
            "note": "Headers only; actual tensor-range streaming and conversion not implemented.",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if a.output:
            with a.output.open("x", encoding="utf-8") as out:
                json.dump(report, out, indent=2, sort_keys=True)
                out.write("\\n")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Ranged safetensors probe FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"HTTP range probe error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
