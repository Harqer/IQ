#!/usr/bin/env bash
set -euo pipefail

# Remote-only weight-transfer contract.
# Donor checkpoints MUST be mounted read-only by the compute platform.
# The IQ destination MUST be a remote writable mount.
# This script never downloads or copies donor/final model weights to worker-local disk.

GPT_OSS_ROOT="${GPT_OSS_MOUNT:-/mnt/donors/gpt-oss-20b}"
GPT_OSS_ORIGINAL="${GPT_OSS_ORIGINAL:-$GPT_OSS_ROOT/original}"
MAMBA3_ROOT="${MAMBA3_MOUNT:-/mnt/donors/mamba3-mimo-1.5b}"
IQ_REMOTE_ROOT="${IQ_REMOTE_MOUNT:-/mnt/iq-output}"
OUTPUT_DIR="${IQ_COMPLETE_OUTPUT:-$IQ_REMOTE_ROOT}"

GPT_OSS_REVISION="${GPT_OSS_REVISION:-0d1f28dce7d3a8794b20345f496dfabb28d51e70}"
MAMBA3_REVISION="${MAMBA3_REVISION:-bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required remote-mounted donor file missing: $1" >&2
    exit 2
  fi
}

require_file "$GPT_OSS_ORIGINAL/config.json"
require_file "$GPT_OSS_ORIGINAL/model.safetensors"
require_file "$MAMBA3_ROOT/config.json"
require_file "$MAMBA3_ROOT/pytorch_model.bin"

case "$OUTPUT_DIR" in
  "$IQ_REMOTE_ROOT"|"$IQ_REMOTE_ROOT"/*) ;;
  *)
    echo "IQ_COMPLETE_OUTPUT must remain under IQ_REMOTE_MOUNT ($IQ_REMOTE_ROOT)." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"

python -m iq_transfer.cli complete-iq \
  --gpt-oss-original "$GPT_OSS_ORIGINAL" \
  --mamba3-checkpoint "$MAMBA3_ROOT" \
  --gpt-oss-revision "$GPT_OSS_REVISION" \
  --mamba3-revision "$MAMBA3_REVISION" \
  --output "$OUTPUT_DIR"

echo "Complete IQ checkpoint written directly to remote mount: $OUTPUT_DIR"
