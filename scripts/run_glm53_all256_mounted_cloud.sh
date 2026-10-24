#!/usr/bin/env bash
# Full capacity, all 256 routed GLM experts at *each of IQ's 23 mapped sparse stages*.
# This is not exact transfer of unmapped GLM layers: source reference is preserved.
# HF Jobs: mount hf://buckets/kofecheckmate/mocha:/hf-bucket:rw on cpu-xl/cpu-performance.
set -euo pipefail
if [[ "${IQ_CLOUD_WORKER:-}" != "1" || ! -d /hf-bucket/legacy/visionboxx-macha ]]; then
  echo "REJECT: cloud worker with existing HF bucket mount required; no local execution." >&2
  exit 2
fi
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{4,79}$ ]]; then
  echo "REJECT: RUN_ID must be 5..80 safe characters" >&2
  exit 2
fi
HF_ROOT=/hf-bucket/legacy/visionboxx-macha
CALIBRATION_DIR="$HF_ROOT/calibration"
REMOTE_OUTPUT="$HF_ROOT/IQ/runs/$RUN_ID"
WORK_ROOT="${IQ_TRANSFER_ROOT:-/tmp/iq-all256-$RUN_ID}"
if [[ -e "$WORK_ROOT" || -e "$REMOTE_OUTPUT" ]]; then
  echo "REJECT: scratch or remote run already exists. Never overwrite previous IQ artifacts." >&2
  exit 2
fi
if [[ ! -f "$CALIBRATION_DIR/glm53_calibration.json" ]]; then
  echo "REJECT: canonical calibration missing from mounted existing bucket." >&2
  exit 2
fi
mkdir -p "$WORK_ROOT/glm53" "$WORK_ROOT/mamba3"
GLM_DIR="$WORK_ROOT/glm53"
MAMBA_DIR="$WORK_ROOT/mamba3"
LOCAL_OUTPUT="$WORK_ROOT/IQ"
GLM_REV=abf42e6c0e53bc29a875511301e7b7fcd220315b
MAMBA_REV=bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Use official immutable donor weights remotely: only config/index are fetched.
hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$GLM_REV" --local-dir "$GLM_DIR"
hf download state-spaces/mamba3-mimo-1.5b \
  config.json pytorch_model.bin tokenizer.json tokenizer_config.json \
  --revision "$MAMBA_REV" --local-dir "$MAMBA_DIR"

python - "$CALIBRATION_DIR" <<'PY'
import sys
from iq_transfer.glm53_calibration import GLM53CalibrationSolution
from iq_transfer.complete_transplant import canonical_glm53_config
c = GLM53CalibrationSolution.load(sys.argv[1])
if (len(c.stages) != 24 or c.source_layers != 78 or
    c.calibration_mode != 'weight_only' or
    c.target_config_fingerprint != canonical_glm53_config().fingerprint):
    raise SystemExit('Existing WARM calibration is not the verified canonical 24-stage/32-expert weight-only baseline')
print('VERIFIED calibration: 24 donor-to-IQ stage maps; 256-capacity compiler may reuse it', flush=True)
PY

python -m iq_transfer.cli glm53-plan-compile-shards \
  --config "$GLM_DIR/config.json" --index "$GLM_DIR/model.safetensors.index.json" \
  --calibration "$CALIBRATION_DIR" --routed-experts 256 \
  > "$WORK_ROOT/required_shards.txt"
echo "Planning complete: $(wc -l < "$WORK_ROOT/required_shards.txt") GLM donor shards via HTTP Range."

python -m iq_transfer.cli glm53-compile \
  --checkpoint "$GLM_DIR" \
  --calibration "$CALIBRATION_DIR" \
  --mamba3-checkpoint "$MAMBA_DIR" \
  --output "$LOCAL_OUTPUT" \
  --routed-experts 256 \
  --checkpoint-revision "$GLM_REV" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --mamba3-revision "$MAMBA_REV" \
  --donor-license GLM-5.3 \
  --streaming-source

python - "$GLM_DIR" "$LOCAL_OUTPUT" "$RUN_ID" <<'PY'
import hashlib, json, sys
from pathlib import Path
src, root, run_id = map(Path, sys.argv[1:])
idx = json.loads((root/'model.safetensors.index.json').read_text())
m = json.loads((root/'transplant_manifest.json').read_text())
selected = m.get('selected_source_expert_ids')
if not isinstance(selected, dict) or len(selected) != 24:
    raise SystemExit('REJECT: missing 24-stage expert mapping')
sparse = [v for v in selected.values() if isinstance(v, list)]
if len(sparse) != 23 or any(len(v) != 256 or len(set(v)) != 256 or
                             set(v) != set(range(256)) for v in sparse):
    raise SystemExit('REJECT: compiled source expert selection does not retain all 256 routed experts at each sparse stage')
keys = idx.get('weight_map')
if not isinstance(keys, dict) or not keys:
    raise SystemExit('REJECT: compiled safetensors index invalid')
files = sorted(set(keys.values()))
if any(not (root / file).is_file() or (root/file).stat().st_size == 0 for file in files):
    raise SystemExit('REJECT: missing/empty indexed IQ safetensors shard')
for name in ('iq_config.json','transplant_manifest.json','model.safetensors.index.json'):
    if not (root/name).is_file():
        raise SystemExit(f'REJECT: missing {name}')
donor_index_bytes = (src/'model.safetensors.index.json').read_bytes()
donor_index = json.loads(donor_index_bytes)
source_keys = list(donor_index.get('weight_map', {}))
source_layers = sorted({int(k.split('.')[2]) for k in source_keys
                        if k.startswith('model.layers.') and k.split('.')[2].isdigit()})
if source_layers != list(range(78)):
    raise SystemExit('REJECT: official 78-layer source index is incomplete')
reference = {
  'schema_version': 1,
  'statement': 'all 78 original GLM layers remain externally referenced; only 24 are directly mapped into IQ',
  'source_repo': 'zai-org/GLM-5.3-BF16',
  'source_revision': 'abf42e6c0e53bc29a875511301e7b7fcd220315b',
  'source_index_sha256': hashlib.sha256(donor_index_bytes).hexdigest(),
  'source_index_tensor_count': len(source_keys),
  'original_source_layers': 78,
  'mapped_iq_stages': 24,
  'sparse_mapped_stages': 23,
  'mapped_routed_experts_per_stage': 256,
  'mapped_routed_expert_slots': 23*256,
  'additional_unmapped_teacher_layers': 78-24,
  'exact_full_donor_weights_locally_copied': False,
  'end_to_end_knowledge_retention_established': False,
}
(root/'original_glm_source_reference.json').write_text(json.dumps(reference,indent=2,sort_keys=True)+'\n')
print(f'COMPILED {len(files)} IQ shards; mapped all 5,888 selected slots; teacher retains all 78 original layer references', flush=True)
PY

# Promote only after full compile succeeds. Run output is immutable and unique.
python - "$LOCAL_OUTPUT" "$REMOTE_OUTPUT" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
source, dest = map(Path, sys.argv[1:])
if dest.exists():
    raise SystemExit('REJECT: remote destination already exists')
dest.mkdir(parents=True,exist_ok=False)
(dest/'INCOMPLETE').write_text('Upload in progress; not validated for inference\n')
count=0
for file in sorted(source.rglob('*')):
    if not file.is_file() or file.is_symlink():
        continue
    relative=file.relative_to(source)
    target=dest/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    with file.open('rb') as fin, target.open('xb') as fout:
        shutil.copyfileobj(fin,fout,8*1024*1024)
    if target.stat().st_size != file.stat().st_size:
        raise SystemExit(f'REJECT: incomplete uploaded file {relative}')
    count+=1
(dest/'COMPILE_COMPLETE.json').write_text(json.dumps({
   'stage':'compiled_only_not_inference_validated','uploaded_files':count,
   'original_weights_accessed_as_remote_ranges':True,'mapped_sparse_experts':5888
},sort_keys=True)+'\n')
(dest/'INCOMPLETE').unlink()
print('COMPILED_CHECKPOINT_UPLOADED',dest,flush=True)
PY
