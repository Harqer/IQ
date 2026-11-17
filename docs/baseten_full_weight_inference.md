# Full checkpoint IQ inference on Baseten (NO tiny/random model)

## Status
- The real checkpoint is stored in `hf://buckets/kofecheckmate/mocha` (82 safetensors shards; index declares 33,846,255,884 bytes).
- The actual checkpoint config contains 80 physical layers: Mamba-3 MIMO 32 (rank=4), CSA 12, HCA 12, Stable LatentMoE 24 (32 routed experts, top-k=8), 4,096 hidden size, vocab 128,256.
- Its manifest reports `calibration_mode=weight_only` and prior language-quality failures, so a successful forward does **not** establish coherent generation.
- Baseten account authorization, available credits, a compatible H100/H200 runtime, persistent checkpoints, and GPU inference are **not yet verified**. Do not claim a deployment has run.

## Deploy the whole model, not a substitute

Use Baseten Dedicated Inference with the official Truss **custom Docker HTTP server** mode, not a stock vLLM engine or a hosted GLM API; IQ's custom 80-layer operators and pinned Mamba-3 MIMO kernels are required.

- Start with an image that includes CUDA/PyTorch 2.10 and CUDA 12.8 development headers (e.g. existing project target `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel`) and an available compatible GPU with sufficient VRAM (H100 80 GB or better; subject to runtime verification).
- Ensure the exact repository source from branch `feature/baseten-full-iq-otel-20261117` is present inside the image or container. Do not silently use the older `main` server/loader.
- Import ALL 82 weight shards plus `iq_config.json`, `transplant_manifest.json`, `model.safetensors.index.json`, **`SHA256SUMS` and `COMPLETE.json`** from `hf://buckets/kofecheckmate/mocha` to a mounted **read-only** checkpoint directory. HF bucket URIs are **not equivalent** to normal HF model repository IDs; verify Baseten BDN ingestion/mount method before assuming `weights: hf://...` supports buckets.
- Mount tokenizer assets separately at `/tokenizer` with exact special IDs and vocab size 128,256 (see `checkpoint_preflight.py`). The cached checkpoint does not include tokenizer files at root.
- Set `IQ_REQUIRE_CHECKPOINT_SHA256=1` (the Baseten starter and complete test runner enforce it). This gate checks *all bytes* via streaming SHA-256, each safetensors header, tensor name/shape/dtype/offset coverage, the index digest, and NaN/Inf weight tensors before GPU upload. The gate is CPU/file-I/O only and must execute on a mounted full checkpoint; remote metadata checks alone do not substitute for it.
- Set `IQ_CHECKPOINT_DIR=/checkpoints`, `IQ_TOKENIZER_DIR=/tokenizer`, `IQ_TRACE_DIR=/tmp/iq-traces`, `IQ_FULL_TRACE=1`, and optionally `IQ_TRACE_SYNC=1`.
- Start `bash scripts/baseten_full_iq_start.sh`. It refuses missing shards; calls pinned runtime setup, then serves `scripts/serve_iq_hf.py` with strict full checkpoint loading.
- In the Truss custom server configuration, map server port 8000, predict endpoint `/chat`, readiness/liveness `/health`. For each inference request, log `X-Baseten-Request-Id` to correlate with traces.

Do not put Hugging Face/Baseten/OTLP secrets into git. Use the service's secure secret storage; the endpoint must remain authenticated.

## What tracing covers

`iq_observability/full_trace.py` emits JSONL lines under `IQ_TRACE_DIR` and standard JSON logs on stderr. Without an OTLP collector it still records local logs; if `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` is configured and the OpenTelemetry Python exporter is installed, it also exports OTLP spans.

Critical boundaries: checkpoint metadata and per-shard load; full 80-layer schedule/count; all nested Mamba-3, CSA/HCA and MoE modules; embeddings, norm, reasoning and output head; forward start/stop and CUDA peak/allocated/reserved memory; tensor RMS/max/finite; MoE expert activation/assignments; logits max/min/entropy/top5; generated token progress and exception location.

OpenTelemetry SDK packages (only required when OTLP export requested): `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`.

**Important:** logging 80 layers x generated tokens with synchronization is intentionally intrusive and may reduce throughput. Set `IQ_TRACE_SYNC=0` for lower overhead only after timing interpretation is documented. Don't export raw user prompts, weights or access tokens. Runtime tracing does not prove weight-transfer behavior or language coherence.

For a single run (full checkpoint, real GPU, real weights), once the mounted artifacts and CUDA runtime are ready:

```bash
bash scripts/hf_container_venv.sh python scripts/test_complete_iq_traced.py \
  --checkpoint /checkpoints --tokenizer-dir /tokenizer --max-new-tokens 16 \
  --trace-dir /tmp/iq-traces --profile-cuda
```

The run must produce non-empty generated text, finite logits, 24 MoE layer outputs, and all 80 layer completion spans for **each token**. Report quality separately, including any repeated-subword/incorrect-answer canary results.

**OTLP verification:** `python -m unittest tests.test_iq_otlp_export` sends a real protobuf span and event to a local loopback receiver without Baseten or a GPU. Configure `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://YOUR-COLLECTOR/v1/traces` for actual backend exports; without it, durable local JSONL remains available. The test does not prove a production collector is connected.

**Independent CPU-only full integrity preflight (before paid GPU):** run `python scripts/verify_full_iq_integrity_cpu.py /path/to/full/mocha` after mounting all 82 shards, index, COMPLETE.json, and SHA256SUMS. This hashes every shard and validates every tensor header without importing Torch. It pins the original index SHA-256 (`0154a99903ea86daea357f855ee24ae10e430e2adda022f2a6843476eaaa6128`) and the original SHA256SUMS SHA-256 (`705b9127a8575acc5aba9e48dafa58ddfbb6d7616917711bffcd4f52478e25e2`) in source; changed buckets fail closed until deliberately reviewed.

**Billing guard:** obtain a live Baseten credit/balance estimate and confirm GPU availability before pushing/deploying; do not auto-create billed resources without approval.
