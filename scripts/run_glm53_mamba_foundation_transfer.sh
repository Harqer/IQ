#!/usr/bin/env bash
set -euo pipefail

# Canonical cloud-only Mamba-3 foundation + GLM-5.3 capability transfer.
# Source checkpoints live only in ephemeral worker storage. Durable artifacts
# are synchronized incrementally to the Hugging Face bucket; no weights are
# committed to Git and --delete is never used for bucket syncs.
GLM_REPO="${GLM53_REPO:-zai-org/GLM-5.3-BF16}"
GLM_REVISION="${GLM53_REVISION:?Set GLM53_REVISION to the pinned BF16 commit}"
MAMBA_REPO="${MAMBA3_REPO:-state-spaces/mamba3-mimo-1.5b}"
MAMBA_REVISION="${MAMBA3_REVISION:-bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec}"

HF_TARGET_BUCKET="${HF_TARGET_BUCKET:-hf://buckets/Visionboxx/macha}"
HF_CALIBRATION_URI="${HF_CALIBRATION_URI:-$HF_TARGET_BUCKET/calibration}"
HF_MAMBA_URI="${HF_MAMBA_URI:-$HF_TARGET_BUCKET/mamba-foundation}"
HF_IQ_URI="${HF_IQ_URI:-$HF_TARGET_BUCKET/IQ}"

WORK_ROOT="${IQ_TRANSFER_ROOT:-/tmp/iq-glm53-transfer}"
GLM_DIR="$WORK_ROOT/glm53"
MAMBA_DIR="$WORK_ROOT/mamba3"
ACTIVATIONS_DIR="$WORK_ROOT/activations"
TOKEN_BATCH_DIR="$WORK_ROOT/token-batches"
CALIBRATION_DIR="$WORK_ROOT/calibration"
MAMBA_OUTPUT_DIR="$WORK_ROOT/mamba-foundation"
OUTPUT_DIR="$WORK_ROOT/IQ"

: "${HF_TOKEN:?HF_TOKEN must be supplied by the cloud worker/auth environment}"

mkdir -p \
  "$GLM_DIR" \
  "$MAMBA_DIR" \
  "$ACTIVATIONS_DIR" \
  "$TOKEN_BATCH_DIR" \
  "$CALIBRATION_DIR" \
  "$MAMBA_OUTPUT_DIR" \
  "$OUTPUT_DIR"

cleanup() {
  rm -rf \
    "$GLM_DIR" \
    "$MAMBA_DIR" \
    "$ACTIVATIONS_DIR" \
    "$TOKEN_BATCH_DIR" \
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

python -m pip install -r requirements-model.txt

# Fetch only immutable metadata first. The full ~753B BF16 checkpoint is never
# downloaded as one local snapshot.
hf download "$GLM_REPO" \
  config.json model.safetensors.index.json \
  --revision "$GLM_REVISION" \
  --local-dir "$GLM_DIR" \
  --token "$HF_TOKEN"

mapfile -t BOOTSTRAP_SHARDS < <(
  python -m iq_transfer.cli glm53-plan-bootstrap-shards \
    --config "$GLM_DIR/config.json" \
    --index "$GLM_DIR/model.safetensors.index.json"
)
echo "GLM WARM/bootstrap shard count (on-demand): ${#BOOTSTRAP_SHARDS[@]}"

hf download "$MAMBA_REPO" \
  config.json pytorch_model.bin tokenizer.json tokenizer_config.json \
  --revision "$MAMBA_REVISION" \
  --local-dir "$MAMBA_DIR" \
  --token "$HF_TOKEN"

# Stage 0: use an existing activation bundle, or build it by streaming one
# GLM layer/shard at a time from a fixed token-batch artifact.
if [[ -n "${GLM53_ACTIVATIONS_URI:-}" ]]; then
  hf sync "$GLM53_ACTIVATIONS_URI" "$ACTIVATIONS_DIR"
elif [[ -n "${GLM53_TOKEN_BATCHES_URI:-}" ]]; then
  hf sync "$GLM53_TOKEN_BATCHES_URI" "$TOKEN_BATCH_DIR"
  python -m iq_transfer.cli glm53-stream-capture \
    --checkpoint "$GLM_DIR" \
    --repo-id "$GLM_REPO" \
    --checkpoint-revision "$GLM_REVISION" \
    --token-batches "$TOKEN_BATCH_DIR/glm53-token-batches" \
    --output "$ACTIVATIONS_DIR/glm53-activations" \
    --device "${GLM53_CAPTURE_DEVICE:-cpu}" \
    --attention-implementation "${GLM53_CAPTURE_ATTN:-eager}"
  sync_to_bucket "$ACTIVATIONS_DIR" "$HF_CALIBRATION_URI/activations"
else
  echo "No GLM activation artifact supplied; using data-free weight-only bootstrap"
fi

# Stage 1: compile the actual Mamba-3 foundation transplant and make it durable
# before beginning the much larger GLM capability transform.
python -m iq_transfer.cli mamba3-direct \
  --checkpoint "$MAMBA_DIR" \
  --output "$MAMBA_OUTPUT_DIR" \
  --checkpoint-revision "$MAMBA_REVISION"

sync_to_bucket "$MAMBA_OUTPUT_DIR" "$HF_MAMBA_URI"

# Stage 2: construct WARM / MLA / DSA / MoE calibration state.
BOOTSTRAP_ARGS=(
  --checkpoint "$GLM_DIR"
  --output "$CALIBRATION_DIR"
  --checkpoint-revision "$GLM_REVISION"
  --repo-id "$GLM_REPO"
  --donor-license "${GLM53_DONOR_LICENSE:-GLM-5.3}"
  --streaming-source
  --warm-device "${WARM_DEVICE:-cuda}"
)
if [[ -f "$ACTIVATIONS_DIR/glm53-activations.safetensors" ]]; then
  BOOTSTRAP_ARGS+=(--source-activations "$ACTIVATIONS_DIR/glm53-activations")
fi
python -m iq_transfer.cli glm53-bootstrap-calibration "${BOOTSTRAP_ARGS[@]}"

sync_to_bucket "$CALIBRATION_DIR" "$HF_CALIBRATION_URI"

mapfile -t COMPILE_SHARDS < <(
  python -m iq_transfer.cli glm53-plan-compile-shards \
    --config "$GLM_DIR/config.json" \
    --index "$GLM_DIR/model.safetensors.index.json" \
    --calibration "$CALIBRATION_DIR"
)
echo "GLM calibrated compile shard count (on-demand): ${#COMPILE_SHARDS[@]}"

# Stage 3: compile the single IQ checkpoint. The compiler uses official
# Mamba-3 globals/recurrent weights plus GLM-transformed capability modules.
python -m iq_transfer.cli glm53-compile \
  --checkpoint "$GLM_DIR" \
  --calibration "$CALIBRATION_DIR" \
  --mamba3-checkpoint "$MAMBA_DIR" \
  --output "$OUTPUT_DIR" \
  --checkpoint-revision "$GLM_REVISION" \
  --repo-id "$GLM_REPO" \
  --mamba3-revision "$MAMBA_REVISION" \
  --donor-license "${GLM53_DONOR_LICENSE:-GLM-5.3}" \
  --streaming-source

sync_to_bucket "$OUTPUT_DIR" "$HF_IQ_URI"

echo "Transfer artifacts synchronized:"
echo "  Mamba foundation: $HF_MAMBA_URI"
echo "  Calibration:      $HF_CALIBRATION_URI"
echo "  IQ checkpoint:    $HF_IQ_URI"
