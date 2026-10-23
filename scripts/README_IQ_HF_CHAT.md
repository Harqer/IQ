# IQ on Hugging Face Jobs

IQ is loaded from a read-only HF bucket on H200, then accessed through the
authenticated HF Jobs port proxy via Python `InferenceClient`. This is **not**
Inference Providers and does not make IQ available via its bucket name.

## Install locally (active venv)

```bash
python -m pip install -r scripts/requirements_iq_client.txt
hf auth login  # only if hf auth whoami fails
```

## Preflight (no GPU or HF Job)

```bash
python scripts/iq_hf_inference.py
```

The preflight checks Hugging Face SDK API support, credentials and source files.

## First real inference (explicitly billed H200)

Only after changes are merged and local `main` matches `origin/main`:

```bash
python scripts/iq_hf_inference.py --run-h200 --prompt "What is 2 + 2?"
```

For interactive chat:

```bash
python scripts/iq_hf_inference.py --run-h200 --interactive
```

Type `/quit` or Ctrl+C to cancel. The script also attempts cancellation
when inference fails or SIGTERM is received. HF Jobs enforces a 30-minute
maximum lifetime if the client machine becomes unreachable.

The IQ server provides `GET /health`, legacy `POST /chat`, and
`POST /v1/chat/completions`, with `messages` and `max_tokens`
(up to 64). Streaming is not yet supported.

**Prerequisites:** HF token authorized for the namespace and bucket,
IQ checkpoint already stored in `hf://buckets/kofecheckmate/mocha`,
tokenizer mounted from `unsloth/Meta-Llama-3.1-8B`.
The server must actually generate non-empty text to pass; readiness
alone is not proof of inference success.
