#!/usr/bin/env bash
# Run on a provisioned high-memory cloud worker, NOT the 8 GiB-VRAM workstation.
# Reuse existing immutable 24-stage calibration; compile 32 experts/stage in bulk.
# No weight-only recalibration, no four-expert probes, no overwrite of old IQ runs.
set -euo pipefail

# Fail closed on a developer workstation: this job writes large IQ shards
# and fetches multi-GB Mamba/calibration artifacts to the WORKER filesystem.
# GLM donor shards themselves remain remote, served via HTTP 206 ranges.
if [[ "${IQ_CLOUD_WORKER:-}" != "1" ]]; then
  echo "Refusing local bulk run. Launch on a provisioned cloud worker and set IQ_CLOUD_WORKER=1 there." >&2
  echo "The full GLM-5.3 BF16 checkpoint is NOT required locally or on the worker." >&2
  exit 2
fi

: "${HF_TOKEN:?Set HF_TOKEN on the authenticated cloud worker}"
: "${RUN_ID:?Set a unique RUN_ID (for example bulk32-20261024-a)}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{4,79}$ ]]; then
  echo "RUN_ID must be 5..80 URL-safe characters" >&2
  exit 2
fi

HF_ROOT="${HF_ROOT:-hf://buckets/kofecheckmate/mocha/legacy/visionboxx-macha}"
GLM_REPO="zai-org/GLM-5.3-BF16"
GLM_REVISION="abf42e6c0e53bc29a875511301e7b7fcd220315b"
MAMBA_REPO="state-spaces/mamba3-mimo-1.5b"
MAMBA_REVISION="bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
ROUTED_EXPERTS="${ROUTED_EXPERTS:-32}"
if [[ "$ROUTED_EXPERTS" != 32 && "$ROUTED_EXPERTS" != 64 && "$ROUTED_EXPERTS" != 128 && "$ROUTED_EXPERTS" != 256 ]]; then
  echo "ROUTED_EXPERTS must be 32, 64, 128 or 256" >&2
  exit 2
fi
WORK_ROOT="${IQ_TRANSFER_ROOT:-/tmp/iq-bulk-${RUN_ID}}"
GLM_DIR="$WORK_ROOT/glm53"
MAMBA_DIR="$WORK_ROOT/mamba3"
CALIB_DIR="$WORK_ROOT/calibration"
OUTPUT_DIR="$WORK_ROOT/IQ"
OUTPUT_URI="$HF_ROOT/IQ/runs/$RUN_ID"

if [[ -e "$WORK_ROOT" ]]; then
  echo "Refusing existing WORK_ROOT: $WORK_ROOT" >&2
  exit 2
fi

command -v hf >/dev/null || { echo "Hugging Face CLI 'hf' required" >&2; exit 2; }
mkdir -p "$GLM_DIR" "$MAMBA_DIR" "$CALIB_DIR"
echo "Source calibration: $HF_ROOT/calibration"
echo "New checkpoint: $OUTPUT_URI"
echo "Capacity: $ROUTED_EXPERTS experts per routed stage"
echo "This streams per-layer donor ranges; it does not load the full GLM model."

# The full original 753GB donor is not downloaded; only immutable index + config.
hf download "$GLM_REPO" config.json model.safetensors.index.json \
  --revision "$GLM_REVISION" --local-dir "$GLM_DIR" --token "$HF_TOKEN"

# Uses immutable existing calibrated stage maps: do not recompute WARM.
hf sync "$HF_ROOT/calibration" "$CALIB_DIR"

python - "$CALIB_DIR" "$ROUTED_EXPERTS" <<'PY'
from pathlib import Path
import sys
from iq_transfer.glm53_calibration import GLM53CalibrationSolution
from iq_transfer.complete_transplant import canonical_glm53_config

root = Path(sys.argv[1])
n = int(sys.argv[2])
cal = GLM53CalibrationSolution.load(root)
if len(cal.stages) != 24 or cal.source_layers != 78:
    raise SystemExit("ERROR: calibration does not cover all 24 IQ stages / 78 source layers")
expected = canonical_glm53_config(routed_experts=n).fingerprint
original = canonical_glm53_config().fingerprint
if cal.target_config_fingerprint != expected and not (
    cal.calibration_mode == "weight_only" and cal.target_config_fingerprint == original
):
    raise SystemExit("ERROR: calibration fingerprint incompatible with requested expert capacity")
print("Validated 24-stage original GLM WARM calibration", flush=True)
PY

# 'mamba-foundation/' bucket contains a compiled overlay rather than the
# donor checkpoint required by --mamba3-checkpoint, so fetch the pinned donor.
hf download "$MAMBA_REPO" \
  config.json pytorch_model.bin tokenizer.json tokenizer_config.json \
  --revision "$MAMBA_REVISION" --local-dir "$MAMBA_DIR" --token "$HF_TOKEN"

python -m iq_transfer.cli glm53-plan-compile-shards \
  --config "$GLM_DIR/config.json" \
  --index "$GLM_DIR/model.safetensors.index.json" \
  --calibration "$CALIB_DIR" --routed-experts "$ROUTED_EXPERTS" \
  > "$WORK_ROOT/required_shards.txt"
echo "Planned official donor shards: $(wc -l < "$WORK_ROOT/required_shards.txt")"

# No partial run is ever described as complete; the compiler writes
# independent per-stage shards and the index only after every stage succeeds.
python -m iq_transfer.cli glm53-compile \
  --checkpoint "$GLM_DIR" \
  --calibration "$CALIB_DIR" \
  --mamba3-checkpoint "$MAMBA_DIR" \
  --output "$OUTPUT_DIR" \
  --routed-experts "$ROUTED_EXPERTS" \
  --checkpoint-revision "$GLM_REVISION" \
  --repo-id "$GLM_REPO" \
  --mamba3-revision "$MAMBA_REVISION" \
  --donor-license GLM-5.3 \
  --streaming-source

python - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
index_file = root / "model.safetensors.index.json"
manifest_file = root / "transplant_manifest.json"
config_file = root / "iq_config.json"
for f in (index_file, manifest_file, config_file):
    if not f.is_file():
        raise SystemExit(f"ERROR: incomplete bulk checkpoint, missing {f.name}")
index = json.loads(index_file.read_text())
manifest = json.loads(manifest_file.read_text())
files = set(index.get("weight_map", {}).values())
missing = sorted(name for name in files if not (root / name).is_file())
if not files or missing:
    raise SystemExit(f"ERROR: incomplete weight-map coverage; missing {missing[:20]}")
if len(manifest.get("source_layer_map", {})) != 24:
    raise SystemExit("ERROR: checkpoint lacks 24-stage source mapping")
print(f"Bulk compile complete: {len(files)} shard files, 24 stages, no missing tensors")
PY

# Only a COMPLETE checkpoint reaches the durable remote run.
hf sync "$OUTPUT_DIR" "$OUTPUT_URI"
echo "COMPLETE checkpoint synced to: $OUTPUT_URI"
echo "Follow with full-model load, forward, and generation validation."
