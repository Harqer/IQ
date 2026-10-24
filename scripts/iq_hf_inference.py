#!/usr/bin/env python3
"""Run IQ on an HF H200, chat through InferenceClient, and cancel the paid Job.

Default: offline preflight, no GPU allocation. Use --run-h200 to authorize cost.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from huggingface_hub import (
        InferenceClient, Volume, cancel_job, get_token,
        inspect_job, run_job, whoami,
    )
except ImportError as exc:
    raise SystemExit(
        "Install in the active venv: python -m pip install -U huggingface_hub"
    ) from exc

REPO = Path(__file__).resolve().parent.parent
IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"
TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def preflight() -> None:
    if "expose" not in inspect.signature(run_job).parameters:
        raise RuntimeError(
            "This huggingface_hub version lacks Jobs port exposure. "
            "Upgrade in this venv: python -m pip install -U huggingface_hub"
        )
    if not get_token():
        raise RuntimeError("No HF token; run hf auth login in this environment.")
    for name in ("scripts/serve_iq_hf.py", "scripts/install_model_runtime.sh",
                 "scripts/check_iq_checkpoint_files.py"):
        if not (REPO / name).is_file():
            raise RuntimeError(f"Missing repository file: {name}")
    print("Local preflight OK (no GPU started).", flush=True)


def get_deployed_commit() -> str:
    branch = git("branch", "--show-current")
    if branch not in ("main", ""):
        raise RuntimeError(
            f"Expected canonical main or detached origin/main, found '{branch}'"
        )
    subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin", "main"], check=True)
    head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
    if head != origin:
        raise RuntimeError("Current commit differs from origin/main; reconcile before GPU startup.")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Checkout contains uncommitted tracked changes; refusing GPU startup.")
    return head


def launch_command(commit: str) -> list[str]:
    setup = (
        "set -euo pipefail; "
        "apt-get update -qq && "
        "apt-get install -y -qq git python3-venv python3-dev >/dev/null; "
        "git clone -q https://github.com/Harqer/IQ.git /workspace/IQ; "
        "cd /workspace/IQ; "
        f"git checkout -q {commit}; "
        "export IQ_TOKENIZER_DIR=/tokenizer IQ_DIAGNOSTICS=1; "
        "python3 scripts/check_iq_checkpoint_files.py /checkpoints; "
        "bash scripts/hf_container_venv.sh python scripts/serve_iq_hf.py"
    )
    return ["bash", "-lc", setup]


def health_ready(base: str, token: str) -> bool:
    req = Request(
        f"{base}/health",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(req, timeout=10) as response:
            return json.load(response).get("status") == "ready"
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return False


def wait_ready(job_id: str, namespace: str, token: str, seconds: int) -> str:
    base = f"https://{job_id}--8000.hf.jobs"
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        stage = inspect_job(job_id=job_id, namespace=namespace).status.stage
        if stage in TERMINAL:
            raise RuntimeError(f"IQ job ended before serving: {stage}")
        if health_ready(base, token):
            return base
        time.sleep(10)
    raise TimeoutError("IQ readiness deadline reached")


def chat_once(client: InferenceClient, history: list[dict], limit: int) -> str:
    completion = client.chat.completions.create(
        messages=history, max_tokens=limit, stream=False
    )
    answer = completion.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("IQ returned an empty or malformed completion")
    return answer

SMOKE_PROMPT = "What is 2 + 2? Reply with only the digit."


def check_generation_smoke(answer: str) -> tuple[bool, str]:
    """Specific arithmetic canary; passing is NOT proof of general model quality."""
    value = answer.strip()
    if not value:
        return False, "empty response"
    if len(value) > 80:
        return False, "excessively long arithmetic response"
    if re.search(r"([a-zA-Z]{3,10})\1{2,}", value):
        return False, "repeated subword output"
    if re.fullmatch(r"(?:4(?:\.0+)?|four)[.! ]*", value, re.IGNORECASE):
        return True, "arithmetic smoke passed"
    if re.fullmatch(r"(?:the answer is|answer:|it is)\s+(?:4(?:\.0+)?|four)[.! ]*",
                    value, re.IGNORECASE):
        return True, "arithmetic smoke passed"
    return False, "did not answer 2+2 correctly and concisely"


def verify_generation(client: InferenceClient) -> str:
    answer = chat_once(client, [{"role": "user", "content": SMOKE_PROMPT}], 8)
    passed, reason = check_generation_smoke(answer)
    if not passed:
        raise RuntimeError(
            f"MODEL_QUALITY_FAILURE ({reason}): {answer[:120]!r}. "
            "HTTP success / strict tensor coverage do not imply coherent language."
        )
    return answer


def raw_completion_probe(base: str, token: str) -> str:
    """Diagnose base-LM completion separately from instruction-style chat."""
    payload = json.dumps({
        "prompt": "2 + 2 =",
        "prompt_format": "completion",
        "max_new_tokens": 4,
    }).encode("utf-8")
    request = Request(
        f"{base}/chat",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    answer = result.get("response")
    if not isinstance(answer, str):
        raise RuntimeError(f"invalid raw-completion response: {result!r}")
    return answer


def run(args: argparse.Namespace) -> None:
    token = get_token()
    if token is None:
        raise RuntimeError("HF token missing")
    # Check credentials before allocating any GPU.
    whoami(token=token)
    commit = get_deployed_commit()
    job_id = None
    namespace = args.namespace
    try:
        job = run_job(
            image=IMAGE,
            command=launch_command(commit),
            flavor="h200",
            timeout="30m",
            attempts=1,
            namespace=namespace,
            volumes=[
                Volume(type="bucket", source=f"{namespace}/mocha",
                       mount_path="/checkpoints", read_only=True),
                Volume(type="model", source="unsloth/Meta-Llama-3.1-8B",
                       mount_path="/tokenizer"),
            ],
            expose=[8000],
        )
        job_id = job.id
        print(f"HF Job: {job.url} (auto-cancel on exit)", flush=True)
        base = wait_ready(job_id, namespace, token, args.ready_timeout)
        client = InferenceClient(base_url=base, token=token, timeout=args.request_timeout)
        # Reject a loaded-but-incoherent transplant before offering interaction.
        try:
            smoke_answer = verify_generation(client)
        except RuntimeError:
            if getattr(args, "probe_completion", False):
                try:
                    raw = raw_completion_probe(base, token)
                    print(
                        f"IQ RAW COMPLETION DIAGNOSTIC (not a chat pass): {raw!r}",
                        flush=True,
                    )
                except Exception as probe_error:
                    print(
                        f"IQ raw completion probe failed: {probe_error}",
                        file=sys.stderr,
                    )
            raise
        print(f"IQ arithmetic smoke passed: {smoke_answer!r}", flush=True)
        history: list[dict] = []
        if not args.interactive:
            history.append({"role": "user", "content": args.prompt})
            print("IQ:", smoke_answer if args.prompt == SMOKE_PROMPT else chat_once(client, history, args.max_tokens), flush=True)
            return
        print("IQ ready. Type /quit or Ctrl+C to stop and cancel the Job.", flush=True)
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                break
            if question == "/quit":
                break
            if not question:
                continue
            history.append({"role": "user", "content": question})
            try:
                answer = chat_once(client, history, args.max_tokens)
            except Exception:
                history.pop()
                raise
            history.append({"role": "assistant", "content": answer})
            history[:] = history[-16:]  # Bound history to the server chat limit.
            print("IQ:", answer, flush=True)
    finally:
        if job_id is not None:
            print(f"Canceling paid H200 Job {job_id}...", flush=True)
            for attempt in range(3):
                try:
                    cancel_job(job_id=job_id, namespace=namespace)
                    print("Cancellation requested.", flush=True)
                    break
                except Exception as exc:
                    print(f"Cancel attempt {attempt + 1}/3 failed: {exc}", file=sys.stderr)
                    time.sleep(2)
            else:
                print(
                    f"URGENT: check and cancel manually: hf jobs cancel {namespace}/{job_id}",
                    file=sys.stderr,
                )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-h200", action="store_true", help="Authorize paid H200 inference")
    p.add_argument("--interactive", action="store_true", help="Chat until /quit or Ctrl+C")
    p.add_argument(
        "--probe-completion", action="store_true",
        help="If chat smoke fails, try one raw causal-LM completion in the SAME job",
    )
    p.add_argument(
        "--accept-unvalidated-checkpoint", action="store_true",
        help="Explicitly acknowledge the existing weight-only transplant did not pass language quality",
    )
    p.add_argument("--namespace", default="kofecheckmate")
    p.add_argument("--prompt", default=SMOKE_PROMPT)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--request-timeout", type=int, default=300)
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 64:
        p.error("--max-tokens must be 1..64")
    if args.ready_timeout < 1 or args.request_timeout < 1:
        p.error("timeouts must be positive")
    if args.run_h200 and not args.accept_unvalidated_checkpoint:
        p.error(
            "The mounted mocha checkpoint has weight_only calibration and failed "
            "language generation. Require --accept-unvalidated-checkpoint before any paid H200 run."
        )
    if args.interactive and not args.run_h200:
        p.error("--interactive requires --run-h200")
    try:
        preflight()
        if not args.run_h200:
            print("Paid GPU remains OFF. Add --run-h200 only when ready.")
            return 0
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
        run(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; cleanup executed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"IQ ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
