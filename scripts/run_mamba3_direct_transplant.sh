#!/usr/bin/env bash
set -euo pipefail

DONOR_REPO="${MAMBA3_DONOR_REPO:-state-spaces/mamba3-mimo-1.5b}"
DONOR_REVISION="${MAMBA3_DONOR_REVISION:-bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec}"
ROOT="${IQ_TRANSFER_ROOT:-$PWD/.iq-transfer}"
DONOR_DIR="$ROOT/donor"
OUTPUT_DIR="${IQ_TRANSFER_OUTPUT:-$ROOT/iq-mamba3-overlay}"

: "${HF_TOKEN:?Set HF_TOKEN from an OpenShift Secret}"
: "${IQ_RECIPIENT_CONFIG:?Set IQ_RECIPIENT_CONFIG to the frozen IQHybridConfig JSON}"

mkdir -p "$DONOR_DIR" "$OUTPUT_DIR"

python -m pip install -r requirements-model.txt

hf download "$DONOR_REPO" \
  config.json pytorch_model.bin \
  --revision "$DONOR_REVISION" \
  --local-dir "$DONOR_DIR" \
  --token "$HF_TOKEN"

python -m iq_transfer.cli mamba3-direct \
  --checkpoint "$DONOR_DIR" \
  --checkpoint-revision "$DONOR_REVISION" \
  --recipient-config "$IQ_RECIPIENT_CONFIG" \
  --output "$OUTPUT_DIR"

if [[ -n "${HF_TARGET_REPO:-}" ]]; then
  hf upload "$HF_TARGET_REPO" "$OUTPUT_DIR" . --token "$HF_TOKEN"
  echo "Uploaded transplant artifact to https://huggingface.co/$HF_TARGET_REPO"
else
  echo "HF_TARGET_REPO is unset; artifact remains at $OUTPUT_DIR"
fi
