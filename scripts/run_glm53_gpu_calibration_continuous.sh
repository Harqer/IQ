#!/usr/bin/env bash
# Recover all remaining full78 CUDA WARM maps in a single GPU allocation.
# Each underlying batch commits per-stage receipts to the durable public bucket.
# This is calibration ONLY: no expert compilation before independent 866-file verification.
set -euo pipefail

RUN_ID="${RUN_ID:-glm53-full78-20261024-c}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{4,79}$ ]]; then
  echo >&2 "Invalid RUN_ID"
  exit 3
fi
if [[ "${IQ_PROJECTION_DEVICE:-}" != cuda || "${IQ_CONFIRM_NO_OTHER_WRITER:-}" != 1 ]]; then
  echo >&2 "CUDA and exclusive writer acknowledgment required"
  exit 3
fi
# The underlying batch supports 1..8 stages and requires clean scratch.
export IQ_MAX_CALIBRATION_STAGES="${IQ_MAX_CALIBRATION_STAGES:-8}"
if [[ ! "$IQ_MAX_CALIBRATION_STAGES" =~ ^[1-8]$ ]]; then
  echo >&2 "IQ_MAX_CALIBRATION_STAGES must be 1..8"
  exit 3
fi
MAX_ROUNDS="${IQ_CALIBRATION_MAX_ROUNDS:-5}"
if [[ ! "$MAX_ROUNDS" =~ ^[1-9]$ ]]; then
  echo >&2 "IQ_CALIBRATION_MAX_ROUNDS must be 1..9"
  exit 3
fi

SCRATCH="/tmp/iq-warm-gpu-$RUN_ID"
ROOT="/out/calibration78/$RUN_ID/recovery-stage-uploads"
cd /workspace/IQ

committed_count() {
  python - "$ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
stage_ids = []
for folder in root.glob("stage-*"):
    if not folder.is_dir() or folder.is_symlink():
        raise SystemExit(f"Untrusted stage folder: {folder}")
    try:
        index = int(folder.name.removeprefix("stage-"))
    except ValueError:
        raise SystemExit(f"Unexpected stage folder: {folder}")
    if not 0 <= index < 50 or not (folder / "receipt.json").is_file():
        raise SystemExit(f"Incomplete stage folder: {folder}")
    stage_ids.append(index)
if len(stage_ids) != len(set(stage_ids)) or len(stage_ids) > 35:
    raise SystemExit("Invalid recovered stage inventory")
print(len(stage_ids))
PY
}

if [[ -e "$SCRATCH" ]]; then
  echo >&2 "Refusing dirty scratch at startup: $SCRATCH"
  exit 3
fi

for (( round = 1; round <= MAX_ROUNDS; round++ )); do
  before="$(committed_count)"
  if (( before == 35 )); then
    echo "IQ_ALL_35_NEW_WARM_STAGES_PERSISTED"
    echo "IQ_NEXT_REQUIRED_STEP=independently_verify_complete_866_file_calibration"
    exit 0
  fi
  echo "IQ_CONTINUOUS_WARM_BATCH_ROUND=$round staged=$before/35"
  bash scripts/run_glm53_gpu_calibration_batch.sh
  after="$(committed_count)"
  if (( after <= before )); then
    echo >&2 "No durable progress; refusing another billed CUDA batch"
    exit 3
  fi
  echo "IQ_DURABLE_WARM_STAGE_PROGRESS=$after/35"
  # The previous process exited successfully and published receipts.
  # Clear ONLY this job's hard-coded temporary scratch to permit the next batch.
  if [[ -d "$SCRATCH" && ! -L "$SCRATCH" ]]; then
    rm -rf -- "$SCRATCH"
  elif [[ -e "$SCRATCH" ]]; then
    echo >&2 "Untrusted scratch path; refusing cleanup"
    exit 3
  fi
done
echo "IQ_CALIBRATION_WINDOW_FINISHED staged=$(committed_count)/35"
echo "IQ_NEXT_REQUIRED_STEP=resume_missing_stages_if_any_then_verify_866_files"
