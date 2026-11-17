# Baseten for IQ: read-only account setup and later dedicated inference

**Status:** Platform selected; Baseten account not connected to this chat. There is no
accessible Baseten ChatGPT plugin in the current integration catalog. No cloud
deployment or model call has been made.

## Use the official Baseten management API without spending on GPUs

1. Create or select a Baseten API key under Baseten account settings.
   Keep the key in your shell environment or secret store, never in Git.
2. In the IQ checkout, run the offline inspector:

       python scripts/baseten_iq_readonly.py

   This prints the planned official management endpoints but sends **zero** requests.
3. To authorize safe metadata queries, set BASETEN_API_KEY locally, then run:

       python scripts/baseten_iq_readonly.py --inspect --instance-types

   The inspector makes **GET only** requests to Baseten's Management API:
   GET /v1/models and GET /v1/instance_types, with no deploy or inference actions.
4. For an existing model ID:

       python scripts/baseten_iq_readonly.py --inspect --model-id YOUR_MODEL_ID

   This additionally fetches GET /v1/models/{model_id}/deployments, which
   lets you inspect deployment names/statuses without activating GPUs.

Scripts print **filtered nonsecret metadata**, never API keys or raw error bodies.
The results do not establish that any endpoint is currently healthy without
an actual live inspection.

## IQ custom model deployment design (NOT deployed)

**Choose Truss custom Python Model class** for the GLM-first + complete Mamba-3
MIMO architecture, rather than treating it as an ordinary vLLM model. Baseten
supports a custom Model class with load() and predict(), and a config.yaml
specifying dependencies and GPU resources. Source:
https://docs.baseten.co/development/model/model-class

The eventual implementation must:
- Pin the exact GLM teacher, Mamba donor, tokenizer, adapter and code revisions;
  verify full tensor hashes and token-ID mapping before declaring readiness.
- Keep the previous all256 checkpoint separate: it is a Mamba-lexical hybrid
  and cannot serve as verified original GLM-first weights.
- On Model.load(), initialize the telemetry provider before loading the
  separately verified runtime. Instrument checkpoint, donor, adapter and
  final model-load stages. Fail startup upon any missing or mismatched tensor.
- On Model.predict(), validate input schema and bounds and trace tokenize,
  prefill, first token, decode, postprocess and response. Record bounded
  numerical diagnostics (memory bytes, NaN/Inf counts, latency), not prompts,
  raw tensor values, personal information or exception contents.
- Explicitly verify full-model forward/logit parity, finiteness, recurrence
  and KV/SSM cache behavior, adapter training state, and coherent output before
  enabling user traffic. No fake or default text-output path.
- After explicit compute authorization, select a GPU from actual model
  weight/memory needs. Do not assume a single GPU can hold the original
  753B-class teacher. Assign a bounded timeout and deactivate idle deployments.

## Separate Truss and custom OpenTelemetry exporters

For Baseten's built-in Truss request spans, official configuration requires:

    runtime:
      enable_tracing_data: true
    environment_variables:
      OTEL_EXPORTER_OTLP_ENDPOINT: https://COLLECTOR/v1/traces

That full URL is used by Truss's built-in exporter. Do **not** put
authentication headers or API keys in the YAML. Use Baseten secrets or a
trusted collector. Source: https://docs.baseten.co/observability/tracing

For the project's own Python OpenTelemetry SDK instrumentation, configure
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT to the full /v1/traces URL; metrics, if
enabled, use OTEL_EXPORTER_OTLP_METRICS_ENDPOINT with /v1/metrics. They have
independent exporters. Configure the collector/credentials and test delivery
before the first paid deployment.

## Deployment approval gate

No code in this branch calls a Baseten write endpoint. Do NOT run
baseten model push, activate, promote, or call hosted inference until:
1. Model-specific integrity and inference readiness have been verified.
2. A real custom serving runtime and test matrix pass CPU and actual GPU
   compatibility checks.
3. The user explicitly approves starting billable Baseten compute.

Baseten documentation confirms that model push allocates billable resources:
https://docs.baseten.co/development/model/build-your-first-model
