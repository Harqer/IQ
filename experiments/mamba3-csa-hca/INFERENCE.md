# Phase 5e: opt-in inference through IQ's existing HTTP API

This is a **separate research backend**, not the 235-layer `IQHybridForCausalLM`
runner. It uses the **original pretrained** `state-spaces/mamba3-mimo-1.5b`
24-layer foundation and the accepted Phase 5e CSA/HCA attention checkpoint.

The model weights are never committed to Git or modified. The server pins the
attention checkpoint SHA-256 to
`56cedd804224ca7f5db85834a8beddd461f19795f4b27b1b3503e11b4e2624f6`,
checks source-model/placement/parent metadata, enforces exact attention state
keys and verifies the pretrained model's parameter checksum. The existing
`python scripts/serve_iq_local.py` endpoint remains untouched.

## Start on your Linux machine

```bash
cd ~/IQ
git pull --ff-only origin main
# Use your existing virtualenv with torch, transformers, huggingface_hub.
# First run requires the ORIGINAL pretrained model in the HF cache or a local path.
python scripts/serve_iq_mamba3_csa_hca.py \
  --attention-checkpoint ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5e_attention_candidate.pt \
  --mode hybrid --probe --prompt 'The capital of France is' --max-new-tokens 3
```

To avoid downloading the 1.5B foundation from HF again, specify
`--foundation-checkpoint /your/path/pytorch_model.bin` (must be the **original
Mamba-3 MIMO 1.5B** pretrained checkpoint, not GLM/IQ235 weights).

If the probe succeeds, start an HTTP server bound to loopback:

```bash
python scripts/serve_iq_mamba3_csa_hca.py --mode hybrid --port 8001
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"iq-mamba3-csa-hca-phase5e-hybrid", "messages":[{"role":"user","content":"What is the capital of France?"}], "max_tokens":3}'
```

Use `--mode baseline` to compare with the original pretrained Mamba-3 without
CSA/HCA residuals. For a direct raw text completion, use `--probe` with a
prompt as shown above. Both modes recompute the full prefix on CPU on every
new token; this is **not fast or streaming** and defaults to `127.0.0.1` only.

## Safety and research limits

- This Phase 5e attention checkpoint passed **4/16** small synthetic held-out
  fact predictions, not production accuracy. It is separate from IQ235.
- The CPU MIMO reference has **not** been numerically certified against the
  official CUDA TileLang implementation.
- The added CSA/HCA-style modules approximate rather than reproduce DeepSeek's
  compressed attention. No speedup is claimed.
- Do not expose this research server to the public internet.
- Checkpoint weights remain in `~/IQ/checkpoints/` and the OpenShift PVC, not
  Git history. The 24-layer foundation must be available locally or fetched.
