#!/usr/bin/env bash
# Launch one full IQ H200 inference, wait for a real text response, always cancel.
set -Eeuo pipefail

: "${HF_OWNER:=kofecheckmate}"
: "${IQ_GPU:=h200}"
: "${IQ_TIMEOUT:=30m}"
: "${IQ_PROMPT:=What is 2 + 2? Answer with the number only.}"
: "${IQ_MAX_NEW_TOKENS:=8}"
: "${IQ_IMAGE:=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel}"
: "${IQ_COMMIT:=a82572cf789870112e2a6fd1b6de66d62d12c682}"
: "${IQ_POLL_SECONDS:=10}"
: "${IQ_READY_ATTEMPTS:=90}"

if [[ "${1:-}" != "--run-h200" ]]; then
  echo "Preflight only. Paid GPU will NOT start without --run-h200."
  command -v hf >/dev/null || { echo >&2 "Install HF CLI first: pipx install huggingface_hub[cli]"; exit 2; }
  python3 -m py_compile scripts/iq_chat_client.py
  python3 scripts/iq_chat_client.py --help >/dev/null
  echo "Local client preflight OK. Run bash scripts/run_iq_h200_inference.sh --run-h200 to approve paid inference."
  exit 0
fi

command -v hf >/dev/null || { echo >&2 "hf CLI missing"; exit 2; }
python3 -m py_compile scripts/iq_chat_client.py
hf auth whoami >/dev/null
job=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "$job" ]]; then
    echo "Canceling H200 job $job..."
    hf jobs cancel "$HF_OWNER/$job" || echo >&2 "WARNING: cancellation failed; manually run: hf jobs cancel $HF_OWNER/$job"
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# hf CLI handles the bucket mounts; no weights copied into the Git checkout.
echo "Starting paid full-checkpoint H200 inference..."
hf jobs run --help | grep -q -- '--volume' || { echo >&2 "hf CLI lacks --volume support"; exit 2; }
out=$(hf jobs run \
  --flavor "$IQ_GPU" \
  --timeout "$IQ_TIMEOUT" \
  --volume "hf://buckets/$HF_OWNER/mocha:/checkpoints" \
  --volume "hf://models/unsloth/Meta-Llama-3.1-8B:/tokenizer" \
  --expose 8000 \
  --detach \
  "$IQ_IMAGE" \
  bash -lc "set -euo pipefail; apt-get update -qq && apt-get install -y -qq git python3-venv python3-dev >/dev/null; git clone -q https://github.com/Harqer/IQ.git /workspace/IQ; cd /workspace/IQ; git checkout -q $IQ_COMMIT; export IQ_TOKENIZER_DIR=/tokenizer; python3 scripts/check_iq_checkpoint_files.py /checkpoints; bash scripts/hf_container_venv.sh python scripts/serve_iq_hf.py")
echo "$out"
job=$(printf '%s\n' "$out" | sed -nE 's/.*([0-9a-f]{24}).*/\1/p' | head -1)
if [[ -z "$job" ]]; then
  echo >&2 "Cannot parse job ID. Inspect HF Jobs immediately; automatic cleanup cannot identify the job."
  exit 1
fi
echo "Job: $job"
ready=0
for ((i=1; i<=IQ_READY_ATTEMPTS; i++)); do
  if hf jobs logs "$HF_OWNER/$job" 2>&1 | grep -q 'IQ: listening on port 8000'; then
    ready=1
    break
  fi
  if hf jobs inspect "$HF_OWNER/$job" 2>&1 | grep -Eiq 'FAILED|CANCELED|ERROR'; then
    echo >&2 "IQ job ended before readiness."
    hf jobs logs "$HF_OWNER/$job" || true
    exit 1
  fi
  sleep "$IQ_POLL_SECONDS"
done
if [[ "$ready" != 1 ]]; then
  echo >&2 "Timed out waiting for IQ readiness."
  hf jobs logs "$HF_OWNER/$job" || true
  exit 1
fi
python3 scripts/iq_chat_client.py --job-id "$job" --owner "$HF_OWNER" \
  --prompt "$IQ_PROMPT" --max-new-tokens "$IQ_MAX_NEW_TOKENS"
