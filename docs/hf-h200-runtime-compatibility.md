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

## Native image safety, dependency pinning, and same-model fallback

The HF image `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` provides
the baseline CUDA-enabled PyTorch **and** its corresponding Triton package.
This is a base-image assumption supported by earlier job startup logs, not
proof of full IQ generation. Official PyTorch publishes Torch 2.10 with a
CUDA 12.8 wheel; the exact HF GPU host driver is checked at job startup.

`scripts/hf_runtime_contract.py` is an **inference-job startup check**,
not a standalone GPU job or kernel execution test. It verifies:
- the selected image's installed Torch 2.10.0 and CUDA 12.8 runtime;
- the matching CUDA 12.8 `nvcc` from the *devel* image;
- H200 SM90, BF16 support, and availability of a CUDA device;
- an active venv inheriting image-native GPU packages;
- the physical import paths/versions of Torch and Triton **before and
  after** pip dependency resolution;
- Mamba-3's pinned source commit, TileLang 0.1.8, TVM-FFI 0.1.9, and
  QuACK package metadata.

The installer adds exact Torch/Triton version constraints based on the
image **before installing anything**. Dependency resolver conflicts fail
explicitly instead of silently replacing a GPU Torch wheel with CPU or
another CUDA build. `pip check` is required after installing IQ. Keep
the installed CUDA compiler and PyTorch CUDA minor versions aligned for
the source-built Mamba extension.

**Fallback policy:** A missing/unusable CuTe recurrent decoding backend
falls back to IQ's existing full-prefix recomputation, preserving the
same rank-4 MIMO weights and TileLang forward computation. This is
slower but is not an architecture change. That safe fallback is the
default for `scripts/test_complete_iq.py`, which passes the complete
prefix on every generation step. It is **not an automatic recovery from
TileLang compilation failure**. Missing CUDA, SM90, a valid MIMO kernel,
or compatible compiled packages are hard failures. Switching to CPU,
SISO, or mismatched CUDA wheels would silently change/corrupt the intended
runtime and is not a supported fallback.

For one authorized full IQ inference job, with checkpoint files already
mounted into the container, use:

```bash
cd /app/IQ
# The HF job should check out the exact current main commit, not 37447ea.
bash scripts/hf_full_inference.sh /mounted/iq-checkpoint --max-new-tokens 16
```

The HF job must supply `git`, `python -m venv`, and a working C/C++
build toolchain: the earlier launcher installed `git` and
`python3-venv` with `apt-get`. For a reproducible full-inference
run, check out a specific reviewed `main` commit, mount the checkpoint,
and then call the entrypoint; the scripts do not silently clone or
change their own Git revision.

The runner checks checkpoint shard/index/tokenizer *presence* before
installing model dependencies, enters the inherited-image venv, checks
dependency compatibility, and invokes the **complete IQ model generator**.
It does not start any separate benchmark job or kernel smoke test.
The full checkpoint loader then checks tensor names and shapes. The HF
checkpoint mount path must be provided by the deployment; do not infer it.

An external `/launcher/serve_iq.py` from a prior job is **not** tracked
in this repository and is not automatically updated by these changes.
Only use that launcher after separately inspecting its checkout commit,
checkpoint mapping, decode policy, and actual `/generate` call.

### Out-of-scope fallbacks

CuTe/QuACK decoding may demand a newer CUDA toolkit and TVM-FFI than
this TileLang 0.1.8 setup permits. Full-prefix MIMO avoids the CuTe
`step()` requirement and must not be misrepresented as verified CuTe
compatibility. A separately sourced and verified CuTe backend or an
alternative full MIMO kernel could be introduced in the future via an
explicit operator-selection policy, but is **not** currently present.
No amount of dependency metadata inspection proves actual H200 kernel
JIT compilation or that the entire model generates valid tokens.

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

The repository installer builds an isolated `PIP_CONSTRAINT` file from
its documented TileLang/TVM-FFI pair **and the base image's Torch/Triton
versions**, while preserving caller constraints.
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
- PyTorch 2.10 official CUDA 12.8 wheels:
  https://pytorch.org/get-started/previous-versions/
- NVIDIA CUDA 12.8 release and driver compatibility notes:
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
