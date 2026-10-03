#!/usr/bin/env bash
set -euo pipefail

# Canonical cloud-only GLM-5.3 -> IQ transfer.
# Run this inside Hugging Face Jobs (or another cloud worker), never on a
# developer laptop. Source checkpoints live only in ephemeral job storage.
GLM_REPO="${GLM53_REPO:-zai-org/GLM-5.3-BF16}"
GLM_REVISION="${GLM53_REVISION:?Set GLM53_REVISION to the pinned BF16 commit}"
MAMBA_REPO="${MAMBA3_REPO:-state-spaces/mamba3-mimo-1.5b}"
MAMBA_REVISION="${MAMBA3_REVISION:-bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec}"
HF_TARGET_BUCKET="${HF_TARGET_BUCKET:-hf://buckets/Visionboxx/macha}"
WORK_ROOT="${IQ_TRANSFER_ROOT:-/tmp/iq-glm53-transfer}"
GLM_DIR="$WORK_ROOT/glm53"
MAMBA_DIR="$WORK_ROOT/mamba3"
ACTIVATIONS_DIR="$WORK_ROOT/activations"
CALIBRATION_DIR="$WORK_ROOT/calibration"
OUTPUT_DIR="$WORK_ROOT/IQ"

: "${HF_TOKEN:?HF_TOKEN must be supplied as a secret}"
: "${GLM53_ACTIVATIONS_URI:?Set GLM53_ACTIVATIONS_URI to a persisted GLM-5.3 activation bundle}"

mkdir -p "$GLM_DIR" "$MAMBA_DIR" "$ACTIVATIONS_DIR" "$CALIBRATION_DIR" "$OUTPUT_DIR"

cleanup() {
  rm -rf "$GLM_DIR" "$MAMBA_DIR" "$ACTIVATIONS_DIR" "$CALIBRATION_DIR" "$OUTPUT_DIR"
}
trap cleanup EXIT

python -m pip install -r requirements-transfer.txt

# Download only inside ephemeral job storage.
hf download "$GLM_REPO"   --revision "$GLM_REVISION"   --local-dir "$GLM_DIR"   --token "$HF_TOKEN"

hf download "$MAMBA_REPO"   config.json pytorch_model.bin tokenizer.json tokenizer_config.json   --revision "$MAMBA_REVISION"   --local-dir "$MAMBA_DIR"   --token "$HF_TOKEN"

# Activation bundles are expected to live in Hub storage, not Git.
hf sync "$GLM53_ACTIVATIONS_URI" "$ACTIVATIONS_DIR"

python -m iq_transfer.cli glm53-bootstrap-calibration   --checkpoint "$GLM_DIR"   --source-activations "$ACTIVATIONS_DIR/glm53-activations"   --output "$CALIBRATION_DIR"   --checkpoint-revision "$GLM_REVISION"   --donor-license "${GLM53_DONOR_LICENSE:-GLM-5.3}"   --warm-device "${WARM_DEVICE:-cuda}"

python -m iq_transfer.cli glm53-compile   --checkpoint "$GLM_DIR"   --calibration "$CALIBRATION_DIR"   --mamba3-checkpoint "$MAMBA_DIR"   --output "$OUTPUT_DIR"   --checkpoint-revision "$GLM_REVISION"   --mamba3-revision "$MAMBA_REVISION"   --donor-license "${GLM53_DONOR_LICENSE:-GLM-5.3}"

# Bucket sync is the final durable write. No model weights are committed to Git.
hf sync "$OUTPUT_DIR" "$HF_TARGET_BUCKET"

echo "IQ checkpoint synced to $HF_TARGET_BUCKET"
