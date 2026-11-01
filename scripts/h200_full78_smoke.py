"""One paid H200 Job: load IQ once and test inference over localhost.

Runs inside the SAME H200 container as the actual IQ private chat server,
without relying on the HF proxy that returned Bad Request near job expiration.
This is an end-to-end native CUDA smoke test, not CPU/SISO emulation.
No extra Job is launched. On exit stop the local server process group.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"


def read_smoke_stream(stream) -> tuple[str, str, int]:
    """Parse actual OpenAI-style SSE, require a finish marker and [DONE]."""
    pieces: list[str] = []
    reason = ""
    completed = False
    chunks = 0
    for raw in stream:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            completed = True
            break
        event = json.loads(payload)
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            value = delta.get("content")
            if isinstance(value, str) and value:
                pieces.append(value)
                print(f"IQ_SMOKE_DELTA={value!r}", flush=True)
            if choice.get("finish_reason") is not None:
                reason = str(choice["finish_reason"])
        chunks += 1
    if not completed or not reason:
        raise RuntimeError(
            f"Incomplete IQ generation stream: done={completed}, "
            f"finish_reason={reason!r}, chunks={chunks}, partial={''.join(pieces)!r}"
        )
    return "".join(pieces), reason, chunks


def wait_health(server: subprocess.Popen, seconds: int) -> None:
    expiry = time.monotonic() + seconds
    while time.monotonic() < expiry:
        code = server.poll()
        if code is not None:
            raise RuntimeError(f"IQ chat server stopped during startup: exit={code}")
        try:
            with urlopen(BASE + "/health", timeout=3) as response:
                if response.status == 200 and json.load(response).get("status") == "ready":
                    print("IQ_SMOKE_HEALTH_PASS=1", flush=True)
                    return
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(5)
    raise TimeoutError(f"IQ smoke server not ready after {seconds}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is 2+2? Answer with a single number.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--startup-timeout", type=int, default=700)
    parser.add_argument("--request-timeout", type=int, default=950)
    options = parser.parse_args()
    if not 1 <= options.max_tokens <= 256:
        parser.error("max-tokens must be 1..256")
    if options.startup_timeout + options.request_timeout > 1680:
        parser.error("startup+request timeout must leave 120s of the 30min Job")
    print(
        f"IQ_SMOKE_START=prompt={options.prompt!r} max_tokens={options.max_tokens} "
        f"startup_limit={options.startup_timeout}s request_limit={options.request_timeout}s",
        flush=True,
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "scripts.serve_iq_full78_hf"],
        stdout=None, stderr=None, start_new_session=True,
    )
    try:
        wait_health(server, options.startup_timeout)
        data = json.dumps({
            "model": "IQ",
            "messages": [{"role": "user", "content": options.prompt}],
            "max_tokens": options.max_tokens,
            "stream": True,
        }).encode("utf-8")
        req = Request(
            BASE + "/v1/chat/completions", data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        print("IQ_SMOKE_POST_START=localhost:no-proxy", flush=True)
        with urlopen(req, timeout=options.request_timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"IQ server returned HTTP {response.status}")
            answer, reason, chunks = read_smoke_stream(response)
        print(
            f"IQ_SMOKE_COMPLETE=1 finish_reason={reason} chunks={chunks} "
            f"answer={answer!r}",
            flush=True,
        )
        if not answer.strip():
            print("IQ_SMOKE_WARNING=Model generated no visible text", flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)
        print("IQ_SMOKE_SERVER_STOPPED=1", flush=True)


if __name__ == "__main__":
    main()
