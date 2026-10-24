#!/usr/bin/env bash
# Full GLM-5.3 -> expanded IQ: all 78 original layers, 75 x 256 routed expert slots.
# Deploy ONLY to an HF Jobs cloud worker with source:ro and output:rw buckets.
set -euo pipefail
# The BF16 full78 output is over 500 GiB including routed experts and
# calibration. Never start a paid run against the free 100-GB private quota.
CAPACITY_GIB="${IQ_CONFIRMED_PRIVATE_STORAGE_GIB:-0}"
if [[ ! "$CAPACITY_GIB" =~ ^[0-9]+$ ]] || (( CAPACITY_GIB < 600 )); then
  echo "BLOCKED: full78 BF16 needs verified >=600 GiB free private durable storage; current HF Free tier offers only 100 GB." >&2
  echo "Upgrade storage or configure an authorized durable output with enough capacity, then explicitly set IQ_CONFIRMED_PRIVATE_STORAGE_GIB." >&2
  exit 3
fi
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
if [[ -e "$DEST_CALIB" || -e "$DEST_IQ" || -e "$WORK_ROOT" ]]; then
  echo "Refusing existing run; immutable artifacts are never overwritten." >&2
  exit 2
fi
mkdir -p "$WORK_ROOT/glm53" "$WORK_ROOT/mamba3"
hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$DONOR_REV" --local-dir "$WORK_ROOT/glm53"
hf download state-spaces/mamba3-mimo-1.5b config.json pytorch_model.bin \
  --revision "$MAMBA_REV" --local-dir "$WORK_ROOT/mamba3"
# The official Mamba-3 language model uses the Meta-Llama 3.1 tokenizer,
# which is separately gated. It is NOT required to transform donor weights.
# Never block all 78 layers of weight compilation on a tokenizer permission.
# Package the tokenizer only after separately authorized access is available.
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
  --output "$DEST_CALIB" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --checkpoint-revision "$DONOR_REV" \
  --donor-license GLM-5.3 \
  --routed-experts 256 \
  --full-78 --reuse-weight-only "$SOURCE_CALIB" \
  --warm-device cpu --streaming-source

python - "$DEST_CALIB" <<'PY'
import sys
from iq_transfer.glm53_calibration import GLM53CalibrationSolution
c=GLM53CalibrationSolution.load(sys.argv[1])
assert c.source_layers==78 and len(c.stages)==78
assert [x.source_layer for x in c.stages]==list(range(78))
assert len([x for x in c.stages if x.expert_usage is not None])==75
print('VERIFIED_ALL_78_CALIBRATIONS_WITH_75_ROUTED_LAYERS',flush=True)
PY

echo "FULL78: Stage 2 plan all official GLM shards (HTTP ranges only)."
python -m iq_transfer.cli glm53-plan-compile-shards \
  --config "$WORK_ROOT/glm53/config.json" \
  --index "$WORK_ROOT/glm53/model.safetensors.index.json" \
  --calibration "$DEST_CALIB" \
  --routed-experts 256 --full-78 \
  > "$WORK_ROOT/required_shards.txt"
echo "RANGE_PLAN_SHARDS=$(wc -l < "$WORK_ROOT/required_shards.txt")"
echo "FULL78: Stage 3 stream 78 attention + 78 FFN stages directly into a new durable run."
python -m iq_transfer.cli glm53-compile \
  --checkpoint "$WORK_ROOT/glm53" \
  --calibration "$DEST_CALIB" \
  --mamba3-checkpoint "$WORK_ROOT/mamba3" \
  --output "$DEST_IQ" \
  --routed-experts 256 --full-78 \
  --checkpoint-revision "$DONOR_REV" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --mamba3-revision "$MAMBA_REV" \
  --donor-license GLM-5.3 --streaming-source

python - "$DEST_IQ" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
data=json.loads((root/'transplant_manifest.json').read_text())
source_layers={int(x) for x in data['source_layer_map'].values()}
experts=data['selected_source_expert_ids']
assert source_layers==set(range(78))
assert len(experts)==78
routed=[x for x in experts.values() if isinstance(x,list)]
assert len(routed)==75
assert all(len(x)==256 and set(x)==set(range(256)) for x in routed)
assert (root/'model.safetensors.index.json').exists()
index=json.loads((root/'model.safetensors.index.json').read_text())
assert all((root/path).is_file() for path in index['weight_map'].values())
# The recipient checkpoint is not tokenization-ready until the official
# tokenizer is separately attached; do not claim inference readiness here.
tokenizer_packaged = all((root/fname).is_file() for fname in (
    'tokenizer.json','tokenizer_config.json'
))

(root/'COMPILE_COMPLETE.json').write_text(json.dumps({
  'status':'FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED',
  'source_layers':78,'mapped_layers':78,
  'routed_source_layers':75,'routed_experts_per_layer':256,
  'mapped_routed_expert_slots':19200,
  'not_exact_bitwise_weight_reuse':True,
  'tokenizer_packaged':tokenizer_packaged,
  'inference_ready':False,
  'end_to_end_knowledge_retention_proven':False
},sort_keys=True)+'\n')
print('FULL78_ALL_19200_ROUTED_SLOTS_COMPILED_AND_INDEXED',flush=True)
PY
