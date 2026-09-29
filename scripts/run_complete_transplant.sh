#!/usr/bin/env bash
set -euo pipefail

GPT_OSS_REPO="${GPT_OSS_REPO:-openai/gpt-oss-20b}"
GPT_OSS_REVISION="${GPT_OSS_REVISION:-0d1f28dce7d3a8794b20345f496dfabb28d51e70}"
MAMBA3_REPO="${MAMBA3_REPO:-state-spaces/mamba3-mimo-1.5b}"
MAMBA3_REVISION="${MAMBA3_REVISION:-bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec}"
ROOT="${IQ_TRANSFER_ROOT:-$PWD/.iq-complete-transfer}"
GPT_ROOT="$ROOT/gpt-oss-20b"
GPT_ORIGINAL="$GPT_ROOT/original"
MAMBA_DIR="$ROOT/mamba3"
OUTPUT_DIR="${IQ_COMPLETE_OUTPUT:-$ROOT/IQ-complete}"

: "${HF_TOKEN:?Set HF_TOKEN from an OpenShift Secret}"

mkdir -p "$GPT_ROOT" "$MAMBA_DIR" "$OUTPUT_DIR"
python -m pip install -r requirements-model.txt

hf download "$GPT_OSS_REPO"   original/config.json   original/model.safetensors   tokenizer.json   tokenizer_config.json   special_tokens_map.json   generation_config.json   LICENSE   USAGE_POLICY   --revision "$GPT_OSS_REVISION"   --local-dir "$GPT_ROOT"   --token "$HF_TOKEN"

# chat_template.jinja exists on current donor revisions but older pinned snapshots
# may keep the template inside tokenizer_config.json, so fetch it opportunistically.
hf download "$GPT_OSS_REPO" chat_template.jinja   --revision "$GPT_OSS_REVISION"   --local-dir "$GPT_ROOT"   --token "$HF_TOKEN" >/dev/null 2>&1 || true

hf download "$MAMBA3_REPO"   config.json pytorch_model.bin   --revision "$MAMBA3_REVISION"   --local-dir "$MAMBA_DIR"   --token "$HF_TOKEN"

python -m iq_transfer.cli complete-iq   --gpt-oss-original "$GPT_ORIGINAL"   --mamba3-checkpoint "$MAMBA_DIR"   --gpt-oss-revision "$GPT_OSS_REVISION"   --mamba3-revision "$MAMBA3_REVISION"   --output "$OUTPUT_DIR"

if [[ -n "${HF_TARGET_REPO:-}" ]]; then
  hf upload "$HF_TARGET_REPO" "$OUTPUT_DIR" . --token "$HF_TOKEN"
  echo "Uploaded complete IQ checkpoint to https://huggingface.co/$HF_TARGET_REPO"
else
  echo "HF_TARGET_REPO is unset; complete checkpoint remains at $OUTPUT_DIR"
fi
