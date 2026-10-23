#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

MAMBA3_COMMIT="e9594ce1c732d97440f0332fdc43170a2294dbfa"

# Do not mutate the container's global Python installation. HF images can
# create /opt/iq-venv with --system-site-packages to reuse CUDA-enabled Torch.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo >&2 "Activate a Python virtual environment before installing IQ."
  echo >&2 "For HF containers: python -m venv --system-site-packages /opt/iq-venv"
  echo >&2 "Then: source /opt/iq-venv/bin/activate"
  exit 2
fi
python - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("IQ installation requires an active Python venv")
PY

# Upstream Mamba-3 pins TileLang 0.1.8. TVM-FFI >=0.1.10 breaks its
# NestedLoopChecker at kernel compile time. Constrain every pip operation,
# including the Mamba source build; never rely on a launcher-only override.
export PIP_CONSTRAINT="$repo_root/requirements-mamba3-constraints.txt"
python -m pip install -r requirements-model.txt

MAMBA_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@${MAMBA3_COMMIT}"

python - <<'PY'
from importlib.metadata import version

import torch

from iq_model import (
    MAMBA3_UPSTREAM_COMMIT,
    inspect_mamba3_mimo_runtime,
    require_mamba3_mimo_runtime,
)

expected = {"tilelang": "0.1.8", "apache-tvm-ffi": "0.1.9"}
for package, pinned in expected.items():
    actual = version(package)
    if actual != pinned:
        raise SystemExit(
            f"Incompatible Mamba-3 dependency {package}={actual}; required {pinned}"
        )

if MAMBA3_UPSTREAM_COMMIT != "e9594ce1c732d97440f0332fdc43170a2294dbfa":
    raise SystemExit("IQ Mamba-3 pinned commit does not match installer")

info = inspect_mamba3_mimo_runtime(torch.device("cuda"))
require_mamba3_mimo_runtime(torch.device("cuda"))

print(
    "Mamba-3 MIMO dependencies verified (imports; full IQ inference still required):",
    {
        "version": info.installed_version,
        "source_commit": info.source_commit,
        "mimo_kernel_available": info.mimo_kernel_available,
        "device_capability": info.device_capability,
        "tilelang": version("tilelang"),
        "apache-tvm-ffi": version("apache-tvm-ffi"),
    },
)
PY

# No standalone GPU kernel test or separate paid job is launched here.
# The next H200 container invocation must perform actual IQ inference
# (checkpoint load + generate), which exercises the MIMO prefill path.
