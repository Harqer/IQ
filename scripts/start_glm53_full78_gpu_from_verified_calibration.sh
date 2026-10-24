#!/usr/bin/env bash
# FRESH GPU COMPILATION ONLY from a previously published, hash-verified 78-stage calibration.
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
: "${IQ_STORAGE_ATTESTATION:?Fail-closed: provide recent provider-verified free-space evidence}"
REQUIRED_BYTES="$(python -c 'from scripts.full78_storage_preflight import required_private_bytes;print(required_private_bytes())')"
python scripts/glm53_full78_gate.py storage \
  --attestation "$IQ_STORAGE_ATTESTATION" \
  --required-bytes "$REQUIRED_BYTES"
if [[ "${IQ_PROJECTION_DEVICE:-}" != cuda ]]; then
  echo "BLOCKED: production GPU resume requires IQ_PROJECTION_DEVICE=cuda; CPU is diagnostics-only" >&2
  exit 3
fi
if [[ "${IQ_CONFIRM_NO_OTHER_WRITER:-}" != "1" ]]; then
  echo 'BLOCKED: externally confirm no writer for this target output directory.' >&2
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
CALIBRATION_RUN_ID="${CALIBRATION_RUN_ID:?Point to verified calibration run}"
DEST_CALIB="/out/calibration78/$CALIBRATION_RUN_ID"
DEST_IQ="/out/IQ/runs/$RUN_ID"
WORK_ROOT="/tmp/iq-full78-$RUN_ID"
if [[ ! -f "$DEST_CALIB/glm53_calibration.json" || -e "$DEST_IQ" || -e "$WORK_ROOT" ]]; then
  echo "Fresh compile requires existing verified calibration and NEW empty output." >&2
  exit 2
fi
# Read-only file-by-file verification; the calibration manifest alone is insufficient.
python scripts/glm53_full78_gate.py calibration --root "$DEST_CALIB"
python scripts/glm53_calibration_sha256.py verify --root "$DEST_CALIB"
mkdir -p "$WORK_ROOT/glm53" "$WORK_ROOT/mamba3"
hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$DONOR_REV" --local-dir "$WORK_ROOT/glm53"
hf download state-spaces/mamba3-mimo-1.5b config.json pytorch_model.bin \
  --revision "$MAMBA_REV" --local-dir "$WORK_ROOT/mamba3"
# The official Mamba-3 language model uses the Meta-Llama 3.1 tokenizer,
# which is separately gated. It is NOT required to transform donor weights.
# Never block all 78 layers of weight compilation on a tokenizer permission.
# Package the tokenizer only after separately authorized access is available.
echo "FULL78: Starting NEW run from independent verified 78-stage calibration."
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

echo "FULL78: Fail-fast probe source stage3 slot0 before full GPU compilation."
timeout 300s python - "$WORK_ROOT/glm53" "$DEST_CALIB" "$DONOR_REV" <<'PY'
import sys, torch
from iq_transfer.glm53_lazy_calibration import LazyGLM53Calibration
from iq_transfer.glm53_moe import select_experts_by_usage
from iq_transfer.http_range_source import HubSafetensorsSource
solution = LazyGLM53Calibration(sys.argv[2])
stage = solution.stages[3]
selection = select_experts_by_usage(stage.expert_usage, target_experts=256)
donor_id = selection[0]
key = f"model.layers.{stage.source_layer}.mlp.experts.{donor_id}.gate_proj.weight"
source = HubSafetensorsSource(
    sys.argv[1], repo_id="zai-org/GLM-5.3-BF16", revision=sys.argv[3],
    max_request_bytes=16 * 1024 * 1024, max_tensor_bytes=512 * 1024 * 1024,
)
data = source.get(key)
assert data.dtype == torch.bfloat16 and data.ndim == 2
assert torch.isfinite(data.float()).all()
source.close()
print(f"UNSAVED_EXPERT_SOURCE_READ_OK stage=3 slot=0 donor={donor_id}", flush=True)
PY

# Limit each worker to a recoverable compute window instead of the platform timeout.
MAX_SECONDS="${IQ_MAX_RUN_SECONDS:-14400}"
if [[ ! "$MAX_SECONDS" =~ ^[0-9]+$ ]] || (( MAX_SECONDS < 900 || MAX_SECONDS > 82800 )); then
  echo "BLOCKED: IQ_MAX_RUN_SECONDS must be 900..82800 (15m..23h)" >&2
  exit 3
fi
export IQ_TRANSFER_DEADLINE_EPOCH="$(($(date +%s) + MAX_SECONDS))"
echo "GPU_BUDGET_WINDOW_SECONDS=$MAX_SECONDS"
echo "FULL78: Stage 3 start new immutable output; stream all 78 attention/FFN stages."
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
python scripts/glm53_full78_gate.py checkpoint-complete --root "$DEST_IQ"
echo FULL78_LOCAL_COMPLETION_VERIFIED_REQUIRES_SEPARATE_REMOTE_AUDIT

