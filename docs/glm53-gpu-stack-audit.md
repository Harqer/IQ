# GLM-5.3 -> IQ: CUDA / library compatibility audit (2026-10-24)

This document audits the **offline full-78 weight-transfer worker**, not the
separate Mamba-3 inference/decode engine. It does not certify model quality.
Baseline: `feat/glm53-gpu-transfer-guards`, pinned GLM/Mamba donor revisions.

## Current findings

| Area | Source of truth | Finding | Resolution |
| --- | --- | --- | --- |
| CUDA / Torch | `scripts/start_glm53_full78_gpu_from_verified_calibration.sh` and `requirements-model.txt` | No worker-level compatibility gate before the large transfer | Read-only `check_glm53_gpu_stack.py` with tiny device math smoke |
| Dependencies | `requirements-transfer.txt` | Compiler packages are declared, but HF worker image is not validated and there is no isolated setup contract | `setup_glm53_gpu_worker.sh` reuses image Torch and installs only transfer dependencies |
| CUDA precision | `iq_transfer/glm53_gpu_projection.py` | FP32 CUDA products and adaptive row tiles implemented; BF16 source tensors become CPU FP64 arrays in the MoE path | Retain working algorithm until real numerical/throughput benchmark; forbid silent CPU fallback |
| Transfer I/O | `iq_transfer/http_range_source.py` | Bounded HTTP Range source exists, but reads are synchronous | Profile bytes/s, retry counts, GPU wait before adding prefetch |
| Matrix caching | `iq_transfer/glm53_incremental_moe.py` | Reuses per-stage inverses; `ProjectionOps` caches a RHS basis, but ignores `cache_left` | Profile cache hits/transfer times before optimizing |
| Mamba kernels | `scripts/install_model_runtime.sh` / `requirements-mamba3-constraints.txt` | Pinned TileLang/TVM-FFI and source commit apply to *inference* only | Do not install CUDA Mamba kernels into the offline transfer compiler |
| HF fresh vs resume | `scripts/start_glm53_full78_gpu_from_verified_calibration.sh`, `scripts/resume_glm53_full78_mounted_cloud.sh` | Fresh run accepts separate `CALIBRATION_RUN_ID`, resume previously assumed it equals `RUN_ID` | Resume now requires the original calibration ID explicitly |
| GPU CI | `.github/workflows/glm53-transfer-safety.yml` | CPU-only runner does not exercise GPU math; CUDA-marked tests skip | Run offline contract tests in CI; measure CUDA performance during full transfer, not in standalone paid benchmark runs |

## Recommended environment

- Run transfer on an explicitly selected NVIDIA CUDA container with a tested
  preinstalled CUDA PyTorch wheel, e.g. PyTorch **2.10.0 + cu128** from an
  exact tagged image. Do not let `pip install` silently replace it.
- Use L4/L40S/H100/H200 depending on **measured** throughput and durable
  storage requirements. RTX 2060 / T4 (SM 7.5) can run FP32 projection tests,
  but do not advertise them as native BF16 Tensor Core compute. A100 (SM 8.0),
  L40S (SM 8.9), H200 (SM 9.0) support native BF16.
- Offline transfer requires `torch`, `numpy`, `safetensors`,
  `huggingface_hub`, `requests`, and `packaging`. `transformers==5.17.0`
  applies to activation capture / full model runtime, not to every tensor read.
- `mamba-ssm`, TileLang, Triton, CuTe and TVM-FFI are **inference/runtime**
  kernel requirements. The offline compiler must not build them just to
  multiply GLM projection matrices.
- No ONNX Runtime / Coral Edge TPU / TensorRT dependencies should be added to
  the offline converter without a reproducible benchmark proving benefit.

## Full-78 GPU worker setup

The *authorized* HF worker must provide CUDA PyTorch in its container and
`IQ_CLOUD_WORKER=1`; start/resume/repair scripts then source the isolated setup:

```bash
source scripts/setup_glm53_gpu_worker.sh
python scripts/check_glm53_gpu_stack.py \
  --require-cuda --min-vram-gib 20 --smoke --pip-check
```

The dependency validator checks only the requirements reachable from IQ's transfer libraries, avoiding unrelated base-image packages (e.g. `spin`) that can make a blanket `pip check` fail incorrectly.

The setup uses `python -m venv --system-site-packages` and records the
pre/post Torch version, compiled CUDA runtime and installation path; an
unexpected Torch change fails closed. The preflight reports package versions,
SM version, GPU VRAM, native BF16 capability, CUDA build, nvcc and driver
information. It performs only a tiny FP32 GEMM plus a BF16 storage round-trip.
Pass does not demonstrate Mamba-3 kernel execution or full IQ generation.

Use `--model-runtime` to **also** check the pinned transformers requirement;
use `--require-bf16` if the selected phase truly needs native BF16 computation.
Do not force BF16 for FP32 weight transformation alone.

## Required gates before a paid transfer

1. All 866 full-78 calibration artifacts hash-verified from a separate worker;
   donor revisions and target config fingerprint fixed.
2. Fresh output namespace, sufficient independently verified private storage,
   no concurrent writer and current provenance receipt.
3. GPU preflight green, then begin the **actual complete 78-stage compilation**.
   Validate donor tensors inline as processed; prohibit a redundant one-expert
   download or standalone paid microbenchmark. Compare GPU vs reference
   numerical behavior using CPU offline fixtures and bounded inline sampling
   without separate jobs or stopping production progress.
4. Capture actual stage wall-time, HTTP bytes/requests, PCIe transfers, GEMM
   time, peak RSS/VRAM, and dollars per verified expert **during the real
   compilation**. Use stage-level metrics; no stand-alone per-expert
   benchmarking, no experimental one-expert job loops. Optimize subsequent
   batches using those measurements without repeating completed work.
5. Full compilation must prove all 78 GLM source layers and 75 x 256 routed
   experts were mapped. The original Mamba foundation only supplies 32
   pretrained recurrent slots; remaining recurrent slots need validation.
6. Functional acceptance is **separate**: frozen GLM expert vs projected
   expert/router/attention outputs on held-out activations, checkpoint inference,
   text canaries and held-out language quality. Existing `mocha` model failed
   coherent generation and must never be used as a production reference.

The unfinished `glm53-full78-20261024-c` bucket run lacks the full calibration
hash receipt, run provenance and completion record. Do not adopt its partial
shards without revalidating all dependencies and hashes.

## Official references

- PyTorch CUDA semantics and precision:
  https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch CUDA wheel matrix:
  https://pytorch.org/get-started/previous-versions/
- NVIDIA compute capability matrix:
  https://developer.nvidia.com/cuda/gpus
- PyTorch memory profiling:
  https://docs.pytorch.org/docs/stable/torch_cuda_memory
- Mamba source and current kernel requirements:
  https://github.com/state-spaces/mamba
- HF Jobs and persistent volumes:
  https://huggingface.co/docs/huggingface_hub/en/guides/jobs

**Execution policy:** One authorized GPU session should make continuous
progress on all 78 layers, with bounded stages and resumability. Internal
per-expert shard writes preserve crash recovery; they are not separate jobs.
The current implementation is still a sequential expert loop; multi-expert
GPU batching and overlapping I/O remain performance enhancements to audit,
not verified features. Do not claim batching is implemented by these changes.

**Status:** Offline audit and fail-closed preflight code only. No GPU time,
donor-weight downloads, end-to-end benchmark, or model-quality improvement
is claimed.
