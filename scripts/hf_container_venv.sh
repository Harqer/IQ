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

venv_dir="${IQ_VENV_DIR:-/opt/iq-venv}"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  python -m venv --system-site-packages "$venv_dir"
fi
# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

# Dependency installation only. No standalone GPU test: the command
# must load IQ and perform real inference, not just run a health check.
bash scripts/install_model_runtime.sh
exec "$@"
