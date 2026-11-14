#!/usr/bin/env bash
set -euo pipefail

# One-shot complete IQ inference entrypoint for HF H200 Jobs:
# bash scripts/hf_full_inference.sh /mounted/iq-checkpoint --max-new-tokens 16
# (This is NOT a separate GPU kernel smoke test.)
if (( $# < 1 )); then
  echo >&2 "Usage: bash scripts/hf_full_inference.sh CHECKPOINT_DIR [generation arguments]"
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

checkpoint_dir=$1
shift

# Fail before expensive source compilation if a mounted checkpoint is absent.
python scripts/check_iq_checkpoint_files.py "$checkpoint_dir"

# Fail before costly GPU runtime installation when the full256 checkpoint has no map.
if python -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if c["stable_moe"]["num_experts"]>=256 else 1)' "$checkpoint_dir/iq_config.json"; then
  if [[ -z "${IQ_DEVICE_MAP:-}" ]]; then
    echo >&2 "Full-256 IQ inference requires IQ_DEVICE_MAP pointing to the verified HF-mounted hybrid plan."
    exit 2
  fi
fi
if [[ -n "${IQ_DEVICE_MAP:-}" ]]; then
  python scripts/check_iq_device_map.py "$checkpoint_dir" \
    --device-map "$IQ_DEVICE_MAP" \
    --tokenizer-dir "${IQ_TOKENIZER_DIR:-$checkpoint_dir}" \
    --require-tokenizer
fi

# QuACK/CuTe recurrent step() is a separate backend and is not yet certified
# on the pinned TVM-FFI/CUDA 12.8 stack. Full-prefix MIMO recomputation uses
# the original TileLang rank-4 forward kernel; never downgrade to SISO or CPU.
exec bash scripts/hf_container_venv.sh \
  python scripts/test_complete_iq.py --checkpoint "$checkpoint_dir" "$@"
