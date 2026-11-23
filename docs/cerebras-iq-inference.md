# IQ GLM-5.3 → Mamba-3: Cerebras inference integration

This repository integrates the **ordinary Cerebras Inference API** for account
verification and a **guarded** first request to a separately deployed IQ
Dedicated Inference endpoint. It does **not** claim that the public API can
execute arbitrary IQ architecture/weights from a Hugging Face Storage Bucket.

## What is already implemented

- `scripts/cerebras_iq_probe.py`: model discovery, fail-closed endpoint gate,
  one-request dedicated endpoint probe, structured sanitized JSON report.
- `iq_telemetry.py`: opt-in OpenTelemetry traces (local console or explicit
  OTLP endpoint). No prompt strings, token IDs, weights, access tokens, raw
  exception messages, or stack traces are sent.
- `scripts/test_complete_iq.py` and `iq_model/hybrid.py`: traces for actual
  checkpoint loading, first numerical-token forward pass, every operator type
  (Mamba-3, CSA, HCA, MoE), final AttnRes, and generation steps.
- `tests/test_cerebras_iq_probe.py`: offline safety/error tests.

No model weight was changed, and no Cerebras API request was sent by the
development or test workflow.

## Colab / Linux setup (regular Cerebras API key only)

1. Obtain an Inference API key from https://cloud.cerebras.ai.
2. For Colab, save it as a private Colab Secret named `CEREBRAS_API_KEY`
   and give notebook access. Never place the key in committed source.
3. Clone the IQ branch and install only the API client requirements:

```bash
git clone -b fix/iq-full256-mapped-inference-preflight-20261114 https://github.com/Harqer/IQ.git
cd IQ
python -m pip install -r requirements-cerebras-iq.txt
python -m scripts.cerebras_iq_probe --doctor
```

The notebook may load the Colab secret into the environment with:

```python
from google.colab import userdata
import os
os.environ["CEREBRAS_API_KEY"] = userdata.get("CEREBRAS_API_KEY")
```

After that, **API discovery only, no inference costs**:

```bash
IQ_OTEL_EXPORTER=console python -m scripts.cerebras_iq_probe
```

The output `status=api_connected` only proves the key works.
The list of models is **not** evidence that IQ is deployed.

## Real IQ deployment: separate provider prerequisite

Cerebras Dedicated Inference currently documents a **private-preview
Management API** for custom-weight upload, versioning, and deployment. It uses
a **separate Management API key**. The documented import path is S3; the IQ
checkpoint is stored in a Hugging Face Storage Bucket and is not an S3 bucket.

Their endpoints require a supported model architecture. Our architecture is
nonstandard:

- 235 physical stages: 79 Mamba-3 MIMO, 78 CSA/HCA, 78 Stable LatentMoE.
- 256 experts per MoE stage.
- Block AttnRes; 128,256-token current lexical interface.
- 63,164 indexed tensors in 20,283 SafeTensors shards; ~539.388 GB indexed.
- Hybrid model not yet inference validated; tokenizer is not packaged.

**Must be confirmed by Cerebras before any paid deployment:**

1. Compile and support the actual IQ Mamba-3 MIMO + CSA/HCA + Stable LatentMoE
   + Block AttnRes runtime; don't substitute a hosted GLM or another model.
2. Verify the supported checkpoint format and accepted device/runtime topology.
3. Provide access to private-preview Management API and a compatible
   Dedicated Inference endpoint, including pricing before spending.
4. Specify secure migration from HF Storage Bucket to the approved S3
   staging location, without modifying weights and with checksum verification.
5. Provide endpoint ID and provider-side deployment version/checkpoint
   fingerprint, which we can compare against the IQ manifest/index.
6. Confirm whether provider-side layer diagnostics/telemetry are exposed.

Documentation: https://inference-docs.cerebras.ai/dedicated/management-api

## Guarded first custom-endpoint request (ONLY after provider approval)

```bash
export CEREBRAS_IQ_ENDPOINT_ID="your-confirmed-dedicated-endpoint-id"
IQ_OTEL_EXPORTER=console python -m scripts.cerebras_iq_probe \
  --run-iq --confirm-deployed --allow-billed-request
```

The explicit flags prevent any accidental inference or spending. A successful
request is reported as `dedicated_endpoint_responded`: **not** proof of the
server's actual IQ weights. Correlate the Cerebras deployment version,
checkpoint fingerprint, and server-side traces to establish that.

## Actual first numeric-token IQ forward on a compatible runtime

Use the existing protected IQ CUDA entrypoint (NOT the Cerebras shared API),
with a read-only mounted checkpoint and verified GPU plan:

```bash
IQ_DEVICE_MAP=/mounted/iq-map/full256-hybrid-pattern-h200x8.metadata-plan.json \
IQ_OTEL_EXPORTER=console \
bash scripts/hf_full_inference.sh /mounted/checkpoint \
  --numeric-only --token-ids 1,2,3,4 --max-new-tokens 1
```

This path is only for compatible multi-GPU compute and is **not** launched by
the Cerebras client. It checks real model weights while streaming, including
tensor names, shapes and finite values, then executes real forward/logit checks.
A full model run has **not** yet succeeded.

## Trace export

- Default `IQ_OTEL_EXPORTER=none`: no tracing or export.
- `IQ_OTEL_EXPORTER=console`: traces locally to job output.
- `IQ_OTEL_EXPORTER=otlp` + `IQ_OTEL_ENDPOINT=https://YOUR-COLLECTOR/v1/traces`:
  explicitly exports to your collector. No endpoint, no network export.
- `python -m unittest discover -s tests -p test_cerebras_iq_probe.py -v`:
  run offline tests.

**Do not paste API keys into ChatGPT, logs, GitHub issues, or notebooks.**
