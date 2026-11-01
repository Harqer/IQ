#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
# Scripts executed by path have scripts/ (not the repository root) in sys.path.
# Keep IQ imports available both to runtime verification and the chat server.
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
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

# Build the original pinned Mamba-3 source. Apply exactly two narrowly-scoped,
# fail-closed compatibility changes: package metadata for TileLang/CuTe and
# a masked tail read for V in the fixed-length native MIMO TileLang kernel.
# Verify every changed file against this allowlist before pip can build.
source_dir=$(mktemp -d -t iq-mamba3-source.XXXXXX)
trap 'rm -f "$snapshot" "$constraints"; rm -rf "$source_dir"' EXIT
git clone --quiet --filter=blob:none https://github.com/state-spaces/mamba.git "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$MAMBA3_COMMIT"
git -C "$source_dir" submodule update --init --recursive --quiet
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$MAMBA3_COMMIT" ]]; then
  echo >&2 "Mamba3 source revision differs: $actual_commit"
  exit 2
fi
python scripts/patch_iq_mamba3_dependencies.py "$source_dir"
python scripts/verify_iq_mamba_patch_paths.py "$source_dir"
MAMBA_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "$source_dir"
python scripts/patch_iq_mamba3_dependencies.py \
  --record-installed "$source_dir" --source-commit "$MAMBA3_COMMIT"

# The container inherits unrelated build tools (notably spin) whose Click
# range can conflict with Hugging Face Hub. Validate IQ's full transitive
# *inference* dependency graph instead of every package in the base image.
python scripts/verify_iq_dependencies.py
python scripts/hf_runtime_contract.py after --snapshot "$snapshot"

# No kernel-only tests, GPU benchmarks, or standalone paid job here.
# The HF launcher must proceed to full IQ checkpoint load and text generation.
