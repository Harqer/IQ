#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
MAMBA3_COMMIT="e9594ce1c732d97440f0332fdc43170a2294dbfa"

# Always reuse the CUDA-enabled PyTorch and Triton packaged with the exact HF
# devel image. A venv is mandatory; --system-site-packages reuses Torch
# without an unreviewed wheel reinstall.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo >&2 "Activate the HF Python venv (scripts/hf_container_venv.sh) first"
  exit 2
fi

snapshot=$(mktemp -t iq-runtime-before.XXXXXX)
constraints=$(mktemp -t iq-constraints.XXXXXX)
trap 'rm -f "$snapshot" "$constraints"' EXIT

# These are metadata/device checks, NOT a standalone GPU inference test.
python scripts/hf_runtime_contract.py before --snapshot "$snapshot"

# Preserve any caller-provided pip constraints and then protect the container's
# exact Torch/Triton wheels. Hard-pin the documented TileLang/TVM-FFI pair.
if [[ -n "${PIP_CONSTRAINT:-}" ]]; then
  if [[ ! -f "$PIP_CONSTRAINT" ]]; then
    echo >&2 "Existing PIP_CONSTRAINT is missing: $PIP_CONSTRAINT"
    exit 2
  fi
  cat "$PIP_CONSTRAINT" >> "$constraints"
fi
cat requirements-mamba3-constraints.txt >> "$constraints"
python scripts/hf_runtime_contract.py constraints >> "$constraints"
export PIP_CONSTRAINT="$constraints"

# Pin both layers of the CUDA software stack; pip must error on an incompatible
# requirement instead of silently replacing Torch, Triton, or TVM-FFI.
python -m pip install "setuptools>=61.0.0" wheel packaging ninja
python -m pip install -r requirements-model.txt

# Upstream Mamba-3 requires a source build against the SAME installed CUDA
# PyTorch ABI. Prevent build isolation from replacing it.
MAMBA_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@${MAMBA3_COMMIT}"

python -m pip check
python scripts/hf_runtime_contract.py after --snapshot "$snapshot"

# No kernel-only tests, GPU benchmarks, or standalone paid job here.
# The HF launcher must proceed to full IQ checkpoint load and text generation.
