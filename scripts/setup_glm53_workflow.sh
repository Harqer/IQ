#!/usr/bin/env bash
# Install/check the GLM-5.3 transfer workflow's runtime dependencies once.
# Deliberately avoids reinstalling PyTorch/CUDA or downloading donor weights.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${IQ_VENV:-$HOME/IQ-integrate/.venv-iq-client}"
PY="${IQ_PYTHON:-$VENV/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "Missing Python venv interpreter: $PY" >&2
  echo "Set IQ_VENV to the correct venv path, or IQ_PYTHON to its Python executable." >&2
  exit 2
fi

"$PY" - <<'PY'
import sys
if sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing global pip install: select the existing .venv-iq-client interpreter.")
print("Using isolated Python:", sys.executable)
PY

cd "$REPO"
echo "[1/4] Install declared transfer dependencies (no PyTorch reinstall)"
"$PY" -m pip install -r requirements-transfer.txt
echo "[2/4] Check dependency consistency"
"$PY" -m pip check
echo "[3/4] Import every GLM weight-transfer entrypoint"
"$PY" -m scripts.check_glm53_workflow_env
echo "[4/4] Run range-probe and selective shard-planner unit tests"
"$PY" -m unittest discover -s tests -p 'test_glm53_remote_range_probe.py' -v
"$PY" -m unittest discover -s tests -p 'test_glm53_selective_shard_plan.py' -v
echo "GLM-5.3 transfer Python environment ready; no donor weights were downloaded."
