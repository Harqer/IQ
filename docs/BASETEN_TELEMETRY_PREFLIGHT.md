# IQ → Baseten: telemetry-first, zero-GPU preflight

**Branch:** telemetry/baseten-iq-preflight-20261117 (based on the GLM provenance audit branch).

**This preflight does not start GPUs, deploy to Baseten, invoke models, download weights, or mutate any checkpoints.** It cannot certify generation quality.

## Critical checkpoint distinction

The existing kofecheckmate/iq-glm53-all256 compiler and HF server target a Mamba-derived 128256-ID tokenizer and 4096-wide recipient. The newer GLM-first hybrid preserves the official 154880-ID GLM tokenizer and 6144-wide residual stream. They are **not interchangeable**. The current GLM-first hybrid is not trained or full-inference validated and deliberately rejects active cached generation.

## Offline preflight

Stage a read-only local copy of the intended checkpoint, separately verified original GLM tokenizer, and pinned per-shard SHA-256 manifest. Use CPU only:

    python -m pip install -r scripts/requirements_iq_telemetry.txt
    python scripts/baseten_iq_preflight.py \
      --checkpoint /readonly/iq-glm53-all256 \
      --tokenizer-dir /readonly/glm53-tokenizer \
      --expected-glm-tokenizer-sha256 PINNED_OFFICIAL_FULL_ID_MAP_SHA256 \
      --expected-shards 79 \
      --sha256-manifest /readonly/shards-sha256.json \
      --telemetry console

The hash manifest is JSON of the form:

    {"shards": {"model-mamba-000.safetensors": "64-char SHA256 hex string"}}

Include **all** indexed shards. The digest manifest must be independently pinned from a trusted source; hashing unchecked bytes and creating a manifest from the same source proves only internal consistency, not authenticity.

Checks include: checkpoint foundation/provenance audit, indexed shard count and presence, safetensors header names/shapes/dtypes, independently pinned token-ID map identity, and streaming local shard hashes when supplied. These checks never load tensor values or call CUDA. Header verification is not proof of numerical values, nonfinite tensor data, attention/MLP transfer parity, trained adapters, recurrence cache correctness or coherent generation.

A nonblocked metadata result is explicitly called metadata_consistent_quality_unverified, **not** inference-ready. A blocker exits with code 2. The legacy all256 checkpoint is expected to fail GLM-first identity checks rather than be mistakenly deployed.

## OpenTelemetry / OTLP HTTP

scripts/iq_telemetry.py provides stage-scoped traces, duration fields, sanitized error-type events, and OTLP metric counters/histograms when a metric collector endpoint is configured. Approved stage scopes cover preflight, metadata, tokenizer, shard_inventory, tensor_headers, tensor_hashes, tensor_validation, transfer_validation, checkpoint_load, donor_load, adapter_load, model_load, request, tokenize, prefill, first_token, decode, postprocess and response.

Configure the **future Baseten service** with environment secrets (not Git source):

    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://YOUR_COLLECTOR/v1/traces
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://YOUR_COLLECTOR/v1/metrics
    OTEL_EXPORTER_OTLP_HEADERS=YOUR_PROVIDER_SPECIFIC_AUTH_HEADERS

The metrics endpoint is optional. If instead using OTEL_EXPORTER_OTLP_ENDPOINT, provide the collector base URL rather than the traces-specific path; the exporter appends per-signal routes. Use IQTelemetry(mode="otlp") in the service. The library rejects missing and insecure remote collector configurations before tracing begins; no credentials, raw exception text, prompts, token IDs or checkpoint paths are recorded.

Future **actual** GLM-first serving integration (not yet deployed):

    from scripts.iq_telemetry import IQTelemetry
    telemetry = IQTelemetry(mode="otlp")
    with telemetry.stage("model_load"):
        # load independently validated GLM weights, Mamba donor and trained adapters
        ...
    with telemetry.stage("request", prompt_tokens=12):
        with telemetry.stage("tokenize"):
            ...
        with telemetry.stage("prefill"):
            ...
        with telemetry.stage("decode", max_new_tokens=8):
            ...
    telemetry.flush()

Timing GPU operations requires opt-in synchronization at diagnostic boundaries: CUDA kernels are asynchronous and plain wall-clock spans do not measure kernel execution precisely.

## CPU tests

    python -m unittest discover -s tests -p test_baseten_iq_telemetry.py -v

The tests cover nested spans, failure and secret redaction, OTLP configuration, index/header errors, complete hash manifests, and rejection of Mamba-derived checkpoints in GLM-first mode. They never allocate GPU.

## Remaining release blockers

- Stage and independently authenticate the exact full checkpoint and GLM tokenizer revision.
- Verify the full token-ID map, independent shard hashes and correct tensor layouts.
- Establish pretrained donor, lexical/residual and GLM→Mamba transfer correctness.
- Validate actual full-model forward/numerical parity, finite outputs, and viable decoding/cache semantics on a properly provisioned runner.
- Wire the provided live stage hooks into the actual GLM-first serving entrypoint; no production-ready Baseten service exists yet.
- Authenticate and verify telemetry export to a collector before starting the inference trial.

**Do not start billable Baseten GPUs or publish an inference-ready model based solely on this preflight.**
