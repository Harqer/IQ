#!/usr/bin/env python3
"""Interactive full78 IQ chat in YOUR local terminal, backed by one HF H200 Job.

Default is free preflight. --run-h200 explicitly authorizes one 30-minute
billable GPU Job, loads the full78 sparse checkpoint read-only, and exposes a
private/authenticated chat endpoint. Type prompts in this CLI; /quit or Ctrl+C
requests cancellation and verifies shutdown. The server uses full-prefix
Mamba-3 MIMO and real on-demand expert safetensors. No hidden second GPU Job.

The 235-layer model has not yet completed a full first-token inference; this
CLI may fail readiness within the Job timeout. A successful HTTP response is
NOT proof of coherent language or knowledge transfer.
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

from huggingface_hub import (
    InferenceClient,
    Volume,
    cancel_job,
    get_token,
    inspect_job,
    run_job,
    whoami,
)

REPO = Path(__file__).resolve().parent.parent
BRANCH = "audit/glm53-cuda-stack-preflight-20261024"
EXPECTED_ACCOUNT = "kofecheckmate"
IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"
CHECKPOINT = "kofecheckmate/iq-glm53-all256"
TOKENIZER = "unsloth/Meta-Llama-3.1-8B"
PORT = 8000
TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
def normalized_stage(raw: object) -> str:
    # HF Hub exposes stage as an enum in some versions and a string in others.
    value = getattr(raw, "value", raw)
    return str(value).rsplit(".", 1)[-1].upper()


MAX_TOKENS = 16
MAX_HISTORY_MESSAGES = 16


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *arguments],
        text=True, stderr=subprocess.PIPE,
    ).strip()


def checked_commit() -> str:
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(
            f"Expected branch {BRANCH!r}, got {branch!r}. "
            "Check out the full78 development branch."
        )
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked local edits present; commit them before launching.")
    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Invalid local Git SHA")
    subprocess.run(
        ["git", "-C", str(REPO), "fetch", "--quiet", "origin", BRANCH],
        check=True,
    )
    if git("rev-parse", f"origin/{BRANCH}") != commit:
        raise RuntimeError("Local branch is not at the current remote HEAD: git pull --ff-only.")
    return commit


def validate_auth() -> str:
    token = get_token()
    if not token:
        raise RuntimeError("HF login required: hf auth login")
    user = whoami(token=token)
    account = user.get("name") if isinstance(user, dict) else None
    if account != EXPECTED_ACCOUNT:
        raise RuntimeError(
            f"CLI authenticated as {account!r}, expected {EXPECTED_ACCOUNT!r}. "
            "This prevents accidentally charging the wrong account."
        )
    return token


def command_for(commit: str) -> list[str]:
    # Git commit is validated 40-hex; everything else is constant.
    setup = (
        "set -euo pipefail; "
        "apt-get update -qq && "
        "apt-get install -y -qq git python3-venv python3-dev >/dev/null; "
        "git clone -q https://github.com/Harqer/IQ.git /workspace/IQ; "
        "cd /workspace/IQ; "
        f"git checkout -q {commit}; "
        "export IQ_TOKENIZER_DIR=/tokenizer; "
        "bash scripts/hf_container_venv.sh python -m scripts.serve_iq_full78_hf"
    )
    return ["bash", "-lc", setup]


def endpoint(job_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{24}", job_id):
        raise RuntimeError("Unexpected Hugging Face Job ID")
    return f"https://{job_id}--{PORT}.hf.jobs"


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


def wait_for_server(job_id: str, token: str, deadline_seconds: int) -> str:
    base = endpoint(job_id)
    end = time.monotonic() + deadline_seconds
    last_stage = None
    while time.monotonic() < end:
        stage = normalized_stage(inspect_job(job_id=job_id, namespace=EXPECTED_ACCOUNT).status.stage)
        if stage in TERMINAL:
            raise RuntimeError(f"HF Job ended before chat became ready ({stage}). See job logs.")
        if stage != last_stage:
            print(f"HF Job stage: {stage}", flush=True)
            last_stage = stage
        if stage == "RUNNING" and health_ready(base, token):
            print("IQ_FULL78_CLI_READY=1", flush=True)
            return base
        time.sleep(10)
    raise TimeoutError(
        "IQ full78 chat did not become ready before deadline; "
        "the H200 Job will be canceled."
    )


def reply(client: InferenceClient, history: list[dict[str, str]], max_tokens: int) -> str:
    result = client.chat.completions.create(
        messages=history, max_tokens=max_tokens, stream=False,
    )
    response = result.choices[0].message.content
    if not isinstance(response, str):
        raise RuntimeError("Malformed IQ response; see HF Job logs")
    return response


def chat_loop(base: str, token: str, *, max_tokens: int, timeout: int,
              transcript: Path | None = None) -> None:
    client = InferenceClient(base_url=base, token=token, timeout=timeout)
    history: list[dict[str, str]] = []
    print(
        "IQ full78 CLI: type a question; /clear clears history; "
        "/quit exits AND cancels the H200 job.",
        flush=True,
    )
    try:
        while True:
            try:
                message = input("You: ").strip()
            except EOFError:
                break
            if not message:
                continue
            if message == "/quit":
                break
            if message == "/clear":
                history.clear()
                print("Conversation cleared.", flush=True)
                continue
            if message == "/help":
                print("Commands: /clear, /quit, /help", flush=True)
                continue
            if len(history) + 2 > MAX_HISTORY_MESSAGES:
                print("Conversation history limit reached. Type /clear to start over.", flush=True)
                continue
            proposal = history + [{"role": "user", "content": message}]
            answer = reply(client, proposal, max_tokens)
            print("IQ:", answer, flush=True)
            # Keep only successful exchanges; never silently drop prior turns.
            history = proposal + [{"role": "assistant", "content": answer}]
    finally:
        if transcript is not None:
            if transcript.exists() or transcript.is_symlink():
                raise RuntimeError(f"Refusing to overwrite existing transcript {transcript}")
            transcript.write_text(
                json.dumps(history, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def cancel_and_verify(job_id: str) -> bool:
    print(f"Canceling paid H200 Job {job_id}...", flush=True)
    for attempt in range(3):
        try:
            cancel_job(job_id=job_id, namespace=EXPECTED_ACCOUNT)
            break
        except Exception as error:
            # The Job might have already timed out/terminated.
            try:
                stage = normalized_stage(inspect_job(
                    job_id=job_id, namespace=EXPECTED_ACCOUNT,
                ).status.stage)
                if stage in TERMINAL:
                    print(f"Job already terminal: {stage}", flush=True)
                    return True
            except Exception:
                pass
            print(f"Cancel attempt {attempt+1} failed: {error}", file=sys.stderr)
            time.sleep(2)
    for _ in range(5):
        try:
            stage = normalized_stage(inspect_job(
                job_id=job_id, namespace=EXPECTED_ACCOUNT,
            ).status.stage)
            if stage in TERMINAL:
                print(f"IQ_HF_JOB_TERMINATED={stage}", flush=True)
                return True
        except Exception as error:
            print(f"Job status verification failed: {error}", file=sys.stderr)
        time.sleep(2)
    print(
        f"UNVERIFIED CANCELLATION: check https://huggingface.co/jobs/{EXPECTED_ACCOUNT}/{job_id}",
        file=sys.stderr,
    )
    return False


def job_options(commit: str) -> dict:
    """Construct the HF Job request without invoking the paid API.

    No attempts= argument: supported SDKs default to one attempt, and
    currently installed versions may reject the newer optional parameter.
    """
    return {
        "image": IMAGE,
        "command": command_for(commit),
        "flavor": "h200",
        "timeout": "30m",
        "namespace": EXPECTED_ACCOUNT,
        "volumes": [
            Volume(type="bucket", source=CHECKPOINT,
                   mount_path="/checkpoints", read_only=True),
            Volume(type="model", source=TOKENIZER,
                   mount_path="/tokenizer", read_only=True),
        ],
        "expose": [PORT],
    }


def validate_jobs_api(options: dict) -> None:
    """Check all submission kwargs against the installed HF SDK before billing."""
    try:
        inspect.signature(run_job).bind(**options)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "Installed huggingface_hub Jobs API is incompatible with the "
            f"requested H200 job parameters: {exc}. "
            "Upgrade huggingface_hub before submitting a paid job."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-h200", action="store_true",
                        help="Explicitly authorize one billable H200 (30-minute timeout)")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=1200,
                        help="Maximum server readiness wait in seconds (<=1400)")
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--transcript", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= MAX_TOKENS:
        parser.error("max-tokens must be 1..16")
    if not 1 <= args.ready_timeout <= 1400 or args.request_timeout < 1:
        parser.error("Invalid readiness/request timeout")
    commit = checked_commit()
    token = validate_auth()
    options = job_options(commit)
    validate_jobs_api(options)
    print(f"IQ full78 CLI preflight OK, commit={commit}", flush=True)
    if not args.run_h200:
        print(
            "No GPU started. To launch a billable H200 for 30 minutes, "
            "rerun with --run-h200.",
            flush=True,
        )
        return 0
    job_id: str | None = None
    try:
        job = run_job(**options)
        job_id = str(job.id)
        print(
            f"HF_JOB_ID={job_id}\n"
            f"HF_JOB_URL=https://huggingface.co/jobs/{EXPECTED_ACCOUNT}/{job_id}\n"
            "The remote GPU auto-times out after 30 minutes.",
            flush=True,
        )
        base = wait_for_server(job_id, token, args.ready_timeout)
        chat_loop(
            base, token, max_tokens=args.max_tokens,
            timeout=args.request_timeout, transcript=args.transcript,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupt requested; canceling the H200 Job.", flush=True)
        return 130
    finally:
        if job_id is not None:
            if not cancel_and_verify(job_id):
                print("WARNING: Cancellation unverified. Manually cancel the HF Job.",
                      file=sys.stderr)


if __name__ == "__main__":
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"IQ_FULL78_CLI_ERROR={exc}", file=sys.stderr)
        sys.exit(1)
