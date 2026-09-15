#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements-mamba-v2.txt
python -m pip install --no-build-isolation \
  "git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa"

python - <<'PY'
import mamba_ssm
import fla
import torch
print("torch", torch.__version__)
print("mamba_ssm", getattr(mamba_ssm, "__version__", "source"))
print("fla", getattr(fla, "__version__", "source"))
print("cuda", torch.cuda.is_available())
PY
