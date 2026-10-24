#!/usr/bin/env bash
# Full GLM-5.3 -> expanded IQ: all 78 original layers, 75 x 256 routed expert slots.
# Deploy ONLY to an HF Jobs cloud worker with source:ro and output:rw buckets.
set -euo pipefail
echo "BLOCKED_UNSAFE_LEGACY_FULL78: monolithic job could publish an incomplete calibration and waste paid compute." >&2
echo "Use scripts/repair_glm53_full78_calibration.sh, then independent SHA256 verification, then scripts/start_glm53_full78_gpu_from_verified_calibration.sh or the provenanced resume." >&2
exit 3

python scripts/full78_storage_preflight.py
if [[ "${IQ_CLOUD_WORKER:-}" != 1 || ! -d /hf-bucket/legacy/visionboxx-macha/calibration || ! -d /out ]]; then
  echo "FULL78 REQUIREMENT: cloud worker with immutable calibration mount and writable private bucket" >&2
  exit 2
fi
RUN_ID="${RUN_ID:?Unique run id required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{4,79}$ ]]; then
  echo "Invalid run id" >&2; exit 2
fi
DONOR_REV=abf42e6c0e53bc29a875511301e7b7fcd220315b
MAMBA_REV=bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec
SOURCE_CALIB=/hf-bucket/legacy/visionboxx-macha/calibration
DEST_CALIB="/out/calibration78/$RUN_ID"
DEST_IQ="/out/IQ/runs/$RUN_ID"
WORK_ROOT="/tmp/iq-full78-$RUN_ID"
LOCAL_CALIB="$WORK_ROOT/calibration"
if [[ -e "$DEST_CALIB" || -e "$DEST_IQ" || -e "$WORK_ROOT" ]]; then
  echo "Refusing existing run; immutable artifacts are never overwritten." >&2
  exit 2
fi
mkdir -p "$WORK_ROOT/glm53"
hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$DONOR_REV" --local-dir "$WORK_ROOT/glm53"
# Current weight-only WARM solution validated on read; no changes to original.
python - "$SOURCE_CALIB" <<'PY'
import sys
from iq_transfer.glm53_calibration import GLM53CalibrationSolution
from iq_transfer.complete_transplant import canonical_glm53_config
solution=GLM53CalibrationSolution.load(sys.argv[1])
assert solution.source_layers==78
assert len(solution.stages)==24
assert solution.calibration_mode=='weight_only'
assert solution.target_config_fingerprint==canonical_glm53_config().fingerprint
print('EXISTING_24_WARM_MAPS_VERIFIED',flush=True)
PY

echo "FULL78: Stage 1 create *new* 78-stage WARM calibration, reusing existing 24."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
python -m iq_transfer.cli glm53-bootstrap-calibration \
  --checkpoint "$WORK_ROOT/glm53" \
  --output "$LOCAL_CALIB" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --checkpoint-revision "$DONOR_REV" \
  --donor-license GLM-5.3 \
  --routed-experts 256 \
  --full-78 --reuse-weight-only "$SOURCE_CALIB" \
  --warm-device cuda --streaming-source

python - "$LOCAL_CALIB" <<'PY'
import sys
from iq_transfer.glm53_calibration import GLM53CalibrationSolution
c=GLM53CalibrationSolution.load(sys.argv[1])
assert c.source_layers==78 and len(c.stages)==78
assert [x.source_layer for x in c.stages]==list(range(78))
assert len([x for x in c.stages if x.expert_usage is not None])==75
print('VERIFIED_ALL_78_CALIBRATIONS_WITH_75_ROUTED_LAYERS',flush=True)
PY


echo "FULL78: publish complete local calibration only after local validation"
mkdir -p "$DEST_CALIB"
python - "$LOCAL_CALIB" "$DEST_CALIB" <<'PY'
import pathlib,shutil,sys
source=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2])
files=sorted(p for p in source.iterdir() if p.is_file())
assert len(files)>800, f"Expected >800 calibration artifacts, got {len(files)}"
for index,p in enumerate(files,1):
    shutil.copyfile(p, target/p.name)
    if index%50==0:
        print(f"CALIBRATION_UPLOAD_PROGRESS={index}/{len(files)}",flush=True)
print(f"CALIBRATION_UPLOAD_COPIED={len(files)}",flush=True)
PY
echo "CALIBRATION_PUBLISHED_LOCAL_CHECK_COMPLETE"
echo "CRITICAL: run separate read-only remote calibration verification BEFORE weight compilation."
