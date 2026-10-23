# Hugging Face H200 inference runtime — source-grounded compatibility

## Scope and policy

These choices are based on upstream source declarations and published
compatibility notes; they **do not certify** that complete IQ generation works.
No isolated CPU/GPU jobs or kernel-only H200 runs are needed before the next
**full IQ inference** attempt. Never start a paid Hugging Face GPU job without
explicit approval. The inference job must actually generate tokens, not merely
load the model or return HTTP 200 from `/health`.

## Recommended H200 environment

| Component | Configuration | Source / limitation |
| --- | --- | --- |
| NVIDIA H200 | Hopper SM90 | NVIDIA Hopper compatibility guide |
| Container image | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` | Previously used by IQ on HF H200; preserves the known image and its CUDA-enabled PyTorch |
| NVIDIA driver | Host must support CUDA 12.8 | H200 compute capability is SM90; actual HF host driver is managed by HF and cannot be asserted without a job |
| Python | 3.12, `/opt/iq-venv` using `--system-site-packages` | Reuses the GPU-enabled PyTorch image while isolating new packages |
| Mamba-3 source | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | IQ-pinned revision |
| TileLang | `==0.1.8` | Pinned upstream Mamba-3 dependency |
| Apache TVM-FFI | `==0.1.9` | Documented fix for TileLang 0.1.8's `_NestedLoopCheckVisitor` crash |
| Triton | `>=3.5.0` (PyTorch-compatible) | Upstream Mamba-3 requirement; do not override the image's resolved Torch stack arbitrarily |
| QuACK | `>=0.3.4` | Upstream Mamba-3 requirement; QuACK's published hardware documentation requires CUDA toolkit 12.9+ |
| NVIDIA CuTe DSL | QuACK transitive dependency | Used by Mamba-3 `step()`; NOT documented compatible with TVM-FFI 0.1.9 in all versions |

**Retain CUDA 12.8 for the next full IQ inference attempt.** H200 supports
CUDA 12.8 (SM90), and this exact base image previously reached checkpoint
load and server startup. The demonstrated exception was in the TileLang/
TVM-FFI compiler integration, not a CUDA 12.8 GPU architecture failure.
An upstream Mamba-3 issue documents the TVM-FFI 0.1.9 workaround and shows
MIMO forward-kernel passes without requiring a CUDA 13 upgrade.

QuACK/CuTe's separately published CUDA toolkit 12.9+ requirement applies
to its *incremental step()* path. That decode path remains unsupported
in this configuration; do **not** enable it or claim it works on CUDA
12.8. If incremental decoding is later implemented, document and resolve
its distinct CUDA/toolkit/TVM-FFI requirements before adopting a new
image.

## Important: MIMO prefill and incremental decoding are NOT equivalent

Pinned upstream `mamba3.py`:
- Runs **MIMO full-sequence prefill/forward** using TileLang.
- Uses Triton for supporting operations.
- Uses **CuTe DSL and QuACK** for `step()` token decoding.
- Explicitly documents its CuTe `step()` as *tested only on H100*, not H200.
- Has an upstream documented error: cached `forward()` passes a 3D tensor
  into a `step()` function expecting 2D.
- The TVM-FFI 0.1.9 bound avoids the documented TileLang 0.1.8 compiler
  exception, but upstream reports newer CuTe/QuACK code can require TVM-FFI
  APIs from later releases. **No source-supported single-version solution
  for both paths has been established for IQ's pinned revision**.

For the next full inference job, use IQ's existing **full-prefix recomputation**
path without invoking the unverified incremental Mamba `step()` path.
This is slower, but does not depend on CuTe recurrent decoding. Do not
advertise incremental decoding as production-ready until upstream resolves
the incompatibility or IQ ships a separately reviewed implementation.

MegaCpp's separate H200 TileLang 3D-to-2D shared-memory fix repairs a
TMA/layout-lowering error, **not** the previously observed
`_NestedLoopCheckVisitor._inst` exception. No evidence currently shows
IQ's pinned kernel has MegaCpp's particular TMA layout failure; do not
copy the shape/layout rewrite into IQ without a matching source path.
MegaCpp's H200 MIMO training evidence supports Hopper viability but
does not validate IQ's 4096-width, 128-state configuration.

The repository installer exports `PIP_CONSTRAINT` to enforce the
documented TileLang/TVM-FFI pair for the entire installation.
`scripts/hf_container_venv.sh` creates/activates a Python venv and
installs IQ's runtime without starting any standalone GPU test.
It executes the caller's inference command directly.

## Only approved next H200 run: full inference

A full inference run must:
1. Use the pinned repository revision, documented CUDA-capable image,
   and the existing virtual-environment entrypoint.
2. Load IQ model weights and tokenizer.
3. Submit a real prompt to IQ and verify returned generated tokens,
   not just server readiness.
4. Capture the stack trace if generation fails and stop the job promptly.
5. Exercise incremental CuTe decoding **only after** the dependency/API
   incompatibility is resolved by a documented upstream source.
6. Never substitute a kernel microbenchmark for end-to-end generation.

The full `/generate` request body and launcher are deployment-specific
and are not available from this repository; use the actual serving
entrypoint/schema instead of inventing them.

## Sources

- Mamba-3 pinned source:
  https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py
- Mamba-3 pinned dependencies:
  https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/pyproject.toml
- TileLang / FFI regression:
  https://github.com/state-spaces/mamba/issues/907
  https://github.com/state-spaces/mamba/issues/986
- Decode shape mismatch:
  https://github.com/state-spaces/mamba/issues/1014
  https://github.com/state-spaces/mamba/issues/1022
- CuTe `step()` nondeterminism report:
  https://github.com/state-spaces/mamba/issues/1024
- QuACK package requirements:
  https://github.com/Dao-AILab/quack/blob/v0.6.5/README.md
  https://github.com/Dao-AILab/quack/blob/v0.6.5/pyproject.toml
- Previously used PyTorch CUDA 12.8 container:
  https://hub.docker.com/r/pytorch/pytorch/tags
- H200 / Hopper compute compatibility:
  https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html
- MegaCpp's H200 TileLang and TMA experience (separate compiler issue):
  https://megacpp.com/blog/mamba3-kernel-journey/
  https://megacpp.com/blog/mamba3-parallel-performance/
- Hugging Face Jobs Docker image rules:
  https://huggingface.co/docs/hub/jobs-images
