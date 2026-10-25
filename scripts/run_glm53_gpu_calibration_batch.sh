#!/usr/bin/env bash
# Real CUDA WARM repair on a prebuilt HF GPU worker, resumable source stage
# groups. Each successful stage is SHA256-committed to the PUBLIC output bucket
# immediately; the full78 source manifest and original expert shards remain
# untouched. This script never books hardware or spends prepaid credits.
set -euo pipefail

if [[ "${IQ_HF_CONTAINER_PROVISIONED:-}" != 1 ]]; then
  echo >&2 "GPU image must already have CUDA Torch and IQ dependencies installed"
  exit 3
fi
if [[ "${IQ_PROJECTION_DEVICE:-}" != cuda ]]; then
  echo >&2 "GPU required; no CPU transfer fallback"
  exit 3
fi
if [[ "${IQ_CONFIRM_NO_OTHER_WRITER:-}" != 1 ]]; then
  echo >&2 "Single-writer guarantee not acknowledged"
  exit 3
fi
if [[ "${IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT:-}" != 1 ]]; then
  echo >&2 "Public HF bucket best-effort storage policy must be acknowledged"
  exit 3
fi

RUN_ID="${RUN_ID:-glm53-full78-20261024-c}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{4,79}$ ]]; then
  echo >&2 "invalid checkpoint run id"
  exit 3
fi
MAX_STAGES="${IQ_MAX_CALIBRATION_STAGES:-4}"
if [[ ! "$MAX_STAGES" =~ ^[0-9]+$ ]] || (( MAX_STAGES < 1 || MAX_STAGES > 8 )); then
  echo >&2 "IQ_MAX_CALIBRATION_STAGES must be 1..8"
  exit 3
fi

DONOR_REV=abf42e6c0e53bc29a875511301e7b7fcd220315b
ORIGINAL="/hf-bucket/legacy/visionboxx-macha/calibration"
PUBLIC="/out/calibration78/$RUN_ID"
CHECKPOINT="/out/IQ/runs/$RUN_ID"
WORK="/tmp/iq-warm-gpu-$RUN_ID"

if [[ ! -f "$ORIGINAL/.calibration_integrity.json" ||
      ! -f "$PUBLIC/glm53_calibration.json" ||
      ! -d "$CHECKPOINT" ||
      -e "$WORK" ]]; then
  echo >&2 "missing verified 24-stage source / saved 28-stage destination or dirty scratch"
  exit 3
fi

cd /workspace/IQ
python scripts/check_glm53_storage_policy.py
python scripts/check_glm53_gpu_stack.py --require-cuda \
  --min-vram-gib "${IQ_MIN_GPU_VRAM_GIB:-20}" --smoke --pip-check

mkdir -p "$WORK/donor"
python scripts/reconcile_glm53_partial.py \
  --partial-calibration "$WORK/overlay" \
  --existing-calibration "$PUBLIC" \
  --original-calibration "$ORIGINAL" \
  --partial-checkpoint "$CHECKPOINT" \
  --expected-old-fingerprint 9cba3cfd5aded22306a1c3f52f9096b70c96fb18a87929daeafe7e4c47542611 \
  --link-overlay

# Verify and relink already completed stage batches from earlier GPU sessions.
python -m iq_transfer.glm53_partial_recovery_receipts \
  --overlay "$WORK/overlay" \
  --public-calibration "$PUBLIC"

hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$DONOR_REV" --local-dir "$WORK/donor"

echo "IQ_REAL_CUDA_WARM_BATCH stages=$MAX_STAGES donor_rev=$DONOR_REV"
python -m iq_transfer.cli glm53-bootstrap-calibration \
  --checkpoint "$WORK/donor" \
  --output "$WORK/overlay" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --checkpoint-revision "$DONOR_REV" \
  --donor-license GLM-5.3 \
  --routed-experts 256 --full-78 \
  --repair-overlay --repair-max-stages "$MAX_STAGES" \
  --repair-publish-directory "$PUBLIC/recovery-stage-uploads" \
  --warm-device cuda --streaming-source

echo "IQ_GPU_SESSION_COMPLETE: validate receipts independently after the worker exits"
