#!/usr/bin/env bash
# Source this inside an already-authorized HF GPU worker (not on the user's host).
# Installs ONLY the offline GLM transfer stack; preserves the image's CUDA Torch.
set -euo pipefail

if [[ "${IQ_CLOUD_WORKER:-}" != "1" ]]; then
  echo >&2 "Refusing GPU worker dependency setup outside IQ_CLOUD_WORKER=1"
  return 2 2>/dev/null || exit 2
fi

IQ_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default to a writable user cache: HF workers need not run as root.
IQ_GLM53_VENV="${IQ_GLM53_VENV:-$HOME/.cache/iq-glm53-transfer-venv}"
if [[ ! -f "$IQ_GLM53_VENV/bin/activate" ]]; then
  python -m venv --system-site-packages "$IQ_GLM53_VENV"
fi
# shellcheck disable=SC1090
source "$IQ_GLM53_VENV/bin/activate"

# The preinstalled CUDA-enabled Torch must survive dependency installation
# unchanged. Build the worker image with its correct CUDA wheel beforehand.
BEFORE_TORCH="$(python - <<'PY'
import torch
if not torch.version.cuda:
    raise SystemExit("CPU-only torch detected in GPU worker")
print(torch.__version__ + "|" + str(torch.version.cuda) + "|" + torch.__file__)
PY
)"
python -m pip install --disable-pip-version-check -r "$IQ_REPO_ROOT/requirements-transfer.txt"
AFTER_TORCH="$(python - <<'PY'
import torch
print(torch.__version__ + "|" + str(torch.version.cuda) + "|" + torch.__file__)
PY
)"
if [[ "$BEFORE_TORCH" != "$AFTER_TORCH" ]]; then
  echo >&2 "CUDA Torch was replaced during dependency install; refusing transfer"
  return 3 2>/dev/null || exit 3
fi
# Check only IQ's reachable dependencies, not unrelated image build tools.
python "$IQ_REPO_ROOT/scripts/check_glm53_gpu_stack.py" --pip-check
echo "IQ_GLM53_CUDA_TORCH_PRESERVED=$AFTER_TORCH"
