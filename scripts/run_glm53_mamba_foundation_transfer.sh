#!/usr/bin/env bash
set -euo pipefail

# Canonical cloud-only Mamba-3 foundation + GLM-5.3 capability transfer.
# Source checkpoints live only in ephemeral worker storage. Durable artifacts
# are synchronized incrementally to the Hugging Face bucket; no weights are
# committed to Git and --delete is never used for bucket syncs.
GLM_REPO="\${GLM53_REPO:-zai-org/GLM-5.3-BF16}"
GLM_REVISION="\${GLM53_REVISION:?Set GLM53_REVISION to the pinned BF16 commit}"
MAMBA_REPO="\${MAMBA3_REPO:-state-spaces/mamba3-mimo-1.5b}"
MAMBA_REVISION="\${MAMBA3_REVISION:-bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec}"

HF_TARGET_BUCKET="\${HF_TARGET_BUCKET:-hf://buckets/Visionboxx/macha}"
HF_CALIBRATION_URI="\${HF_CALIBRATION_URI:-$HF_TARGET_BUCKET/calibration}"
HF_MAMBA_URI="\${HF_MAMBA_URI:-$HF_TARGET_BUCKET/mamba-foundation}"
HF_IQ_URI="\${HF_IQ_URI:-$HF_TARGET_BUCKET/IQ}"

WORK_ROOT="\${IQ_TRANSFER_ROOT:-/tmp/iq-glm53-transfer}"
GLM_DIR="$WORK_ROOT/glm53"
MAMBA_DIR="$WORK_ROOT/mamba3"
ACTIVATIONS_DIR="$WORK_ROOT/activations"
CALIBRATION_DIR="$WORK_ROOT/calibration"
MAMBA_OUTPUT_DIR="$WORK_ROOT/mamba-foundation"
OUTPUT_DIR="$WORK_ROOT/IQ"

: "\${HF_TOKEN:?HF_TOKEN must be supplied by the cloud worker/auth environment}"
: "\${GLM53_ACTIVATIONS_URI:?Set GLM53_ACTIVATIONS_URI to the persisted GLM-5.3 activation bundle}"

mkdir -p \
  "$GLM_DIR" \
  "$MAMBA_DIR" \
  "$ACTIVATIONS_DIR" \
  "$CALIBRATION_DIR" \
  "$MAMBA_OUTPUT_DIR" \
  "$OUTPUT_DIR"

cleanup() {
  rm -rf \
    "$GLM_DIR" \
    "$MAMBA_DIR" \
    "$ACTIVATIONS_DIR" \
    "$CALIBRATION_DIR" \
    "$MAMBA_OUTPUT_DIR" \
    "$OUTPUT_DIR"
}
trap cleanup EXIT

sync_to_bucket() {
  local source="$1"
  local destination="$2"
  echo "Syncing $source -> $destination"
  hf sync "$source" "$destination"
}

python -m pip install -r requirements-transfer.txt

# All model bytes are downloaded only into ephemeral worker storage.
hf download "$GLM_REPO" \
  --revision "$GLM_REVISION" \
  --local-dir "$GLM_DIR" \
  --token "$HF_TOKEN"

hf download "$MAMBA_REPO" \
  config.json pytorch_model.bin tokenizer.json tokenizer_config.json \
  --revision "$MAMBA_REVISION" \
  --local-dir "$MAMBA_DIR" \
  --token "$HF_TOKEN"

# Bucket -> ephemeral worker. hf sync is bidirectional.
hf sync "$GLM53_ACTIVATIONS_URI" "$ACTIVATIONS_DIR"

# Stage 1: compile the actual Mamba-3 foundation transplant and make it durable
# before beginning the much larger GLM capability transform.
python -m iq_transfer.cli mamba3-direct \
  --checkpoint "$MAMBA_DIR" \
  --output "$MAMBA_OUTPUT_DIR" \
  --checkpoint-revision "$MAMBA_REVISION"

sync_to_bucket "$MAMBA_OUTPUT_DIR" "$HF_MAMBA_URI"

# Stage 2: construct WARM / MLA / DSA / MoE calibration state.
python -m iq_transfer.cli glm53-bootstrap-calibration \
  --checkpoint "$GLM_DIR" \
  --source-activations "$ACTIVATIONS_DIR/glm53-activations" \
  --output "$CALIBRATION_DIR" \
  --checkpoint-revision "$GLM_REVISION" \
  --donor-license "\${GLM53_DONOR_LICENSE:-GLM-5.3}" \
  --warm-device "\${WARM_DEVICE:-cuda}"

sync_to_bucket "$CALIBRATION_DIR" "$HF_CALIBRATION_URI"

# Stage 3: compile the single IQ checkpoint. The compiler uses official
# Mamba-3 globals/recurrent weights plus GLM-transformed capability modules.
python -m iq_transfer.cli glm53-compile \
  --checkpoint "$GLM_DIR" \
  --calibration "$CALIBRATION_DIR" \
  --mamba3-checkpoint "$MAMBA_DIR" \
  --output "$OUTPUT_DIR" \
  --checkpoint-revision "$GLM_REVISION" \
  --mamba3-revision "$MAMBA_REVISION" \
  --donor-license "\${GLM53_DONOR_LICENSE:-GLM-5.3}"

sync_to_bucket "$OUTPUT_DIR" "$HF_IQ_URI"

echo "Transfer artifacts synchronized:"
echo "  Mamba foundation: $HF_MAMBA_URI"
echo "  Calibration:      $HF_CALIBRATION_URI"
echo "  IQ checkpoint:    $HF_IQ_URI"
