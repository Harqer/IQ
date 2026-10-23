#!/usr/bin/env bash
set -euo pipefail

# Entry point for Hugging Face container jobs. Reuse the image's CUDA-enabled
# PyTorch while installing IQ's model runtime into an isolated venv directory.
if (( $# == 0 )); then
  echo >&2 "Usage: bash scripts/hf_container_venv.sh COMMAND [ARG...]"
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
# Scripts executed by path have scripts/ (not the repository root) in sys.path.
# Keep IQ imports available both to runtime verification and the chat server.
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

venv_dir="${IQ_VENV_DIR:-/opt/iq-venv}"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  python -m venv --system-site-packages "$venv_dir"
fi
# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

# Build upstream CUDA extensions only for the H200/Hopper architecture.
# Explicitly provided image overrides are audited by hf_runtime_contract.py.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# Dependency installation only. No standalone GPU test: the command
# must load IQ and perform real inference, not just run a health check.
bash scripts/install_model_runtime.sh
exec "$@"
