#!/usr/bin/env bash
# Recover missing WARM maps into a full durable all-78 calibration snapshot.
# Run ONLY after private storage capacity >=600GiB has been confirmed.
set -euo pipefail
if [[ "${IQ_PROJECTION_DEVICE:-}" != cuda ]]; then
  echo "BLOCKED: calibration repair requires a CUDA worker, no silent CPU downgrade" >&2
  exit 3
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "BLOCKED: no CUDA on calibration worker"
PY
CAPACITY_GIB="${IQ_CONFIRMED_PRIVATE_STORAGE_GIB:-0}"
if [[ ! "$CAPACITY_GIB" =~ ^[0-9]+$ ]] || (( CAPACITY_GIB < 600 )); then
  echo "BLOCKED: verify >=600GiB free private durable storage before compute." >&2
  exit 3
fi
# Require actual provider-evidenced free capacity, matching the fresh/resume
# GPU workers; a self-declared 600GiB environment variable alone is not proof.
python scripts/full78_storage_preflight.py
: "${IQ_STORAGE_ATTESTATION:?Provider-verified capacity evidence required}"
REQUIRED_BYTES="$(python -c 'from scripts.full78_storage_preflight import required_private_bytes;print(required_private_bytes())')"
python scripts/glm53_full78_gate.py storage \
  --attestation "$IQ_STORAGE_ATTESTATION" \
  --required-bytes "$REQUIRED_BYTES"
if [[ "${IQ_CONFIRM_NO_OTHER_WRITER:-}" != 1 ]]; then
  echo >&2 "BLOCKED: independently verify no other full78 calibration writer is active"
  exit 3
fi
if [[ "${IQ_CLOUD_WORKER:-}" != 1 || ! -d /hf-bucket/legacy/visionboxx-macha/calibration ]]; then
  echo "requires cloud worker and original immutable 24 WARM maps" >&2
  exit 2
fi
: "${HF_TOKEN:?private Hub token required for explicit bucket sync}"
RUN_ID="${RUN_ID:-glm53-full78-20261024-c}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{4,79}$ ]]; then
  echo "Invalid RUN_ID" >&2
  exit 2
fi
DONOR_REV=abf42e6c0e53bc29a875511301e7b7fcd220315b
ROOT="/tmp/glm53-repair-$RUN_ID"
DEST="hf://buckets/kofecheckmate/iq-glm53-all256/calibration78/$RUN_ID"
if [[ -e "$ROOT" ]]; then
  echo "Refusing dirty calibration repair scratch directory" >&2
  exit 2
fi
# Reuse the CUDA Torch wheel from the cloud image; do not reinstall it.
source scripts/setup_glm53_gpu_worker.sh
python scripts/check_glm53_gpu_stack.py --require-cuda \
  --min-vram-gib "${IQ_MIN_GPU_VRAM_GIB:-20}" --smoke --pip-check
mkdir -p "$ROOT/donor"
# Build a local overlay; saved 28 later stages are referenced read-only and
# 15 source layers reuse SHA-verified original maps without a ~10 GiB copy.
# NO partial bucket files are modified by this step.
EXISTING="/out/calibration78/$RUN_ID"
PARTIAL_IQ="/out/IQ/runs/$RUN_ID"
if [[ ! -f "$EXISTING/glm53_calibration.json" || ! -d "$PARTIAL_IQ" ]]; then
  echo >&2 "BLOCKED: expected partial full78 calibration and checkpoint are absent"
  exit 3
fi
python scripts/reconcile_glm53_partial.py \
  --partial-calibration "$ROOT/calibration" \
  --existing-calibration "$EXISTING" \
  --original-calibration /hf-bucket/legacy/visionboxx-macha/calibration \
  --partial-checkpoint "$PARTIAL_IQ" \
  --expected-old-fingerprint 9cba3cfd5aded22306a1c3f52f9096b70c96fb18a87929daeafe7e4c47542611 \
  --link-overlay

hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$DONOR_REV" --local-dir "$ROOT/donor"
echo "CALIBRATION78_COMPUTING_ONLY_MISSING_SOURCE_LAYERS"
python -m iq_transfer.cli glm53-bootstrap-calibration \
  --checkpoint "$ROOT/donor" \
  --output "$ROOT/calibration" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --checkpoint-revision "$DONOR_REV" \
  --donor-license GLM-5.3 \
  --routed-experts 256 --full-78 \
  --repair-overlay \
  --warm-device cuda --streaming-source
# The overlay is ephemeral, and symlinks must NEVER be uploaded. Materialize
# only after the full 78 stages exist and enough local scratch is available.
python - "$ROOT/calibration" <<'PY'
import os, shutil, sys
from pathlib import Path
root=Path(sys.argv[1])
links=[p for p in root.iterdir() if p.is_symlink()]
need=sum(p.stat().st_size for p in links)
free=shutil.disk_usage(root).free
if free < need + 3*1024**3:
    raise SystemExit(f"BLOCKED_INSUFFICIENT_SCRATCH_TO_MATERIALIZE need={need} free={free}")
print(f"CALIBRATION_OVERLAY_MATERIALIZING_LINKS={len(links)}_BYTES={need}",flush=True)
for p in links:
    target=root/(f".{p.name}.materialize.partial")
    if target.exists():
        raise SystemExit(f"unsafe preexisting scratch artifact: {target}")
    with p.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src,dst,length=4*1024*1024)
        dst.flush()
        os.fsync(dst.fileno())
    # The preceding source hash checks and later 866-file receipt prevent
    # silently adopting changed original 24-stage weights.
    p.unlink()
    os.replace(target,p)
print("CALIBRATION_OVERLAY_MATERIALIZED_WITHOUT_SYMLINKS",flush=True)
PY
python - "$ROOT/calibration" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
raw=json.loads((root/'glm53_calibration.json').read_text())
records=raw['stages']
assert raw['source_layers']==78 and len(records)==78
assert [x['source_layer'] for x in records]==list(range(78))
names=set(['glm53_calibration.json'])
maps=['residual_map','attention_input_map','moe_input_map',
      'compressed_kv_map','latent_map','dense_intermediate_map']
for record in records:
    for typ in maps:
        name=record.get(typ)
        if name:
            names.add(name+'.json')
            names.add(name+'.safetensors')
    if record.get('expert_usage'):
        names.add(record['expert_usage'])
for nm in ('lexical-input','final-output'):
    names.add(nm+'.json')
    names.add(nm+'.safetensors')
missing=[nm for nm in names if not (root/nm).is_file() or (root/nm).stat().st_size==0]
if missing:
    raise SystemExit(f'MISSING_LOCAL_WARM_FILES {missing[:12]} count={len(missing)}')
(root/'EXPECTED_FILES.json').write_text(json.dumps(
   {n:(root/n).stat().st_size for n in sorted(names)},sort_keys=True))
print('ALL78_LOCAL_CALIBRATION_FILES_VERIFIED',len(names),flush=True)
PY
python scripts/glm53_calibration_sha256.py make --root "$ROOT/calibration"
echo "CALIBRATION78_EXPLICIT_BUCKET_SYNC_NO_DELETE_AND_NO_OVERWRITE"
# Explicit upload/sync instead of relying on transient mounted-FUSE flush.
hf buckets sync "$ROOT/calibration" "$DEST" --ignore-existing
python - "$ROOT/calibration" "$RUN_ID" <<'PY'
from pathlib import Path
import json,sys
from huggingface_hub import list_bucket_tree
root=Path(sys.argv[1])
run=sys.argv[2]
local=json.loads((root/'EXPECTED_FILES.json').read_text())
prefix=f'calibration78/{run}/'
remote={v.path[len(prefix):]:v.size for v in list_bucket_tree(
  'kofecheckmate/iq-glm53-all256',prefix=prefix
) if v.type=='file' and v.path.startswith(prefix)}
missing={name:sz for name,sz in local.items() if remote.get(name)!=sz}
if missing:
    raise SystemExit(f'BUCKET_CALIBRATION_INCOMPLETE {len(missing)} first={list(missing)[:12]}')
print('ALL78_REMOTE_CALIBRATION_FILES_AND_SIZES_VERIFIED',len(local),flush=True)
PY

echo 'IMPORTANT: launch an independent read-only worker to verify calibration_sha256.json and all 866 SHA256 hashes before any paid GPU weight transfer.'
