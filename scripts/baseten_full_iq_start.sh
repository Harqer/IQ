#!/usr/bin/env bash
set -euo pipefail

# Baseten custom-server entrypoint: full 82-shard checkpoint only.
# The server runs ONLY with an already-mounted real checkpoint and tokenizer.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

export IQ_CHECKPOINT_DIR="${IQ_CHECKPOINT_DIR:-/checkpoints}"
export IQ_TOKENIZER_DIR="${IQ_TOKENIZER_DIR:-/tokenizer}"
export IQ_TRACE_DIR="${IQ_TRACE_DIR:-/tmp/iq-traces}"
export IQ_FULL_TRACE=1
export IQ_REQUIRE_CHECKPOINT_SHA256=1
export IQ_EXPECTED_INDEX_SHA256="0154a99903ea86daea357f855ee24ae10e430e2adda022f2a6843476eaaa6128"
export IQ_EXPECTED_SHA256SUMS_SHA256="705b9127a8575acc5aba9e48dafa58ddfbb6d7616917711bffcd4f52478e25e2"
export IQ_TRACE_SYNC="${IQ_TRACE_SYNC:-1}"
export PORT="${PORT:-8000}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Fail without allocating inference memory if the physical checkpoint is absent.
python scripts/check_iq_checkpoint_files.py "$IQ_CHECKPOINT_DIR"
exec bash scripts/hf_container_venv.sh python scripts/serve_iq_hf.py
