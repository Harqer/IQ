#!/usr/bin/env bash
# RECOVERY MODE for stopped source job only; full GLM-5.3 -> expanded IQ: all 78 original layers, 75 x 256 routed expert slots.
# Deploy ONLY to an HF Jobs cloud worker with source:ro and output:rw buckets.
set -euo pipefail
python scripts/full78_storage_preflight.py
if [[ "${IQ_CONFIRM_PREVIOUS_JOB_STOPPED:-}" != "1" ]]; then
  echo 'Refusing to resume while previous run may still write.' >&2
  exit 2
fi
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
if [[ ! -f "$DEST_CALIB/glm53_calibration.json" || ! -d "$DEST_IQ" || -e "$WORK_ROOT" ]]; then
  echo "Resume requires saved calibration, incomplete IQ shards, and clean scratch." >&2
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
echo "FULL78: Resuming already calibrated 78-stage IQ output; no calibration recomputation."
python - "$DEST_CALIB" <<'PY'
import json, sys
from pathlib import Path
raw=json.loads((Path(sys.argv[1])/'glm53_calibration.json').read_text())
stages=raw['stages']
assert raw['source_layers']==78 and len(stages)==78
assert [int(x['source_layer']) for x in stages]==list(range(78))
assert len([x for x in stages if x.get('expert_usage')])==75
assert all((Path(sys.argv[1])/x['residual_map']).with_suffix('.safetensors').is_file() for x in stages)
print("SAVED_FULL78_CALIBRATION_VERIFIED",flush=True)
PY

echo "FULL78: Probe first unsaved stage-3 donor expert before paid compilation."
timeout 300s python - "$WORK_ROOT/glm53" "$DEST_CALIB" "$DONOR_REV" <<'PY'
import sys, torch
from iq_transfer.glm53_lazy_calibration import LazyGLM53Calibration
from iq_transfer.glm53_moe import select_experts_by_usage
from iq_transfer.http_range_source import HubSafetensorsSource
solution = LazyGLM53Calibration(sys.argv[2])
stage = solution.stages[3]
selection = select_experts_by_usage(stage.expert_usage, target_experts=256)
donor_id = selection[41]
key = f"model.layers.{stage.source_layer}.mlp.experts.{donor_id}.gate_proj.weight"
source = HubSafetensorsSource(
    sys.argv[1], repo_id="zai-org/GLM-5.3-BF16", revision=sys.argv[3],
    max_request_bytes=16 * 1024 * 1024, max_tensor_bytes=512 * 1024 * 1024,
)
data = source.get(key)
assert data.dtype == torch.bfloat16 and data.ndim == 2
assert torch.isfinite(data.float()).all()
source.close()
print(f"UNSAVED_EXPERT_SOURCE_READ_OK stage=3 slot=41 donor={donor_id}", flush=True)
PY

echo "FULL78: Stage 3 stream 78 attention + 78 FFN stages directly into a new durable run."
python -m iq_transfer.cli glm53-compile \
  --checkpoint "$WORK_ROOT/glm53" \
  --calibration "$DEST_CALIB" \
  --mamba3-checkpoint "$WORK_ROOT/mamba3" \
  --output "$DEST_IQ" \
  --routed-experts 256 --full-78 --resume \
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
