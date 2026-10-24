#!/usr/bin/env bash
# Recover missing WARM maps into a full durable all-78 calibration snapshot.
# Run ONLY after private storage capacity >=600GiB has been confirmed.
set -euo pipefail
CAPACITY_GIB="${IQ_CONFIRMED_PRIVATE_STORAGE_GIB:-0}"
if [[ ! "$CAPACITY_GIB" =~ ^[0-9]+$ ]] || (( CAPACITY_GIB < 600 )); then
  echo "BLOCKED: verify >=600GiB free private durable storage before compute." >&2
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
mkdir -p "$ROOT/donor"
hf download zai-org/GLM-5.3-BF16 config.json model.safetensors.index.json \
  --revision "$DONOR_REV" --local-dir "$ROOT/donor"
echo "CALIBRATION78_REGENERATING_ALL_MAPS_LOCALLY"
python -m iq_transfer.cli glm53-bootstrap-calibration \
  --checkpoint "$ROOT/donor" \
  --output "$ROOT/calibration" \
  --repo-id zai-org/GLM-5.3-BF16 \
  --checkpoint-revision "$DONOR_REV" \
  --donor-license GLM-5.3 \
  --routed-experts 256 --full-78 \
  --reuse-weight-only /hf-bucket/legacy/visionboxx-macha/calibration \
  --warm-device cpu --streaming-source
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
echo "CALIBRATION78_EXPLICIT_BUCKET_SYNC_NO_DELETE"
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
