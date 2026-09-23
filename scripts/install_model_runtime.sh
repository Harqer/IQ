#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

MAMBA3_COMMIT="e9594ce1c732d97440f0332fdc43170a2294dbfa"

python -m pip install -r requirements-model.txt

MAMBA_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@${MAMBA3_COMMIT}"

python - <<'PY'
import torch

from iq_model import (
    MAMBA3_UPSTREAM_COMMIT,
    inspect_mamba3_mimo_runtime,
    require_mamba3_mimo_runtime,
)

if MAMBA3_UPSTREAM_COMMIT != "e9594ce1c732d97440f0332fdc43170a2294dbfa":
    raise SystemExit("IQ Mamba-3 pinned commit does not match installer")

info = inspect_mamba3_mimo_runtime(torch.device("cuda"))
require_mamba3_mimo_runtime(torch.device("cuda"))

print(
    "Mamba-3 MIMO runtime verified:",
    {
        "version": info.installed_version,
        "source_commit": info.source_commit,
        "mimo_kernel_available": info.mimo_kernel_available,
        "device_capability": info.device_capability,
    },
)
PY
