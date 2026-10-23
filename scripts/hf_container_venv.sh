#!/usr/bin/env bash
set -euo pipefail

# Entry point for Hugging Face container jobs. Reuse the image's CUDA-enabled
# PyTorch while installing IQ's model runtime into an isolated venv directory.
if (( $# == 0 )); then
  echo >&2 "Usage: bash scripts/hf_container_venv.sh COMMAND [ARG...]"
  exit 2
fi

venv_dir="${IQ_VENV_DIR:-/opt/iq-venv}"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  python -m venv --system-site-packages "$venv_dir"
fi
# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

# Fail before booting the inference server if a Mamba-3 CUDA JIT is broken.
IQ_VERIFY_CUDA_KERNEL=1 bash scripts/install_model_runtime.sh
exec "$@"
