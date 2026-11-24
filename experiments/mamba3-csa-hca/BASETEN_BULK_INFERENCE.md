# Bulk Phase 5e generated-answer audit using Baseten inference credits

This provides a single **bulk** test pass, not 76 manual one-prompt experiments.
It compares the EXACT original `state-spaces/mamba3-mimo-1.5b`
foundation with/without the accepted Phase 5e CSA/HCA adapter, including
actual decoded responses on original/swapped source facts, unseen project
names, paraphrases and general prompts. The same original model and
checkpoint SHA-256 are mandatory on the remote service. Nothing is retrained.

**Why remote:** the local ~300-token CPU benchmark was SIGKILLed with status
137 before completing a generated answer. A 3-token short prompt succeeding
does not establish that the longer bulk workload fits in laptop RAM.

## 1. Stage the model outside Git

```bash
cd ~/IQ
git pull --ff-only origin main
source .venv/bin/activate

python scripts/build_mamba3_baseten_truss.py
```

This verifies the exact adapter SHA-256 before staging the 81 MiB file into
`~/.cache/iq/phase5e-baseten-truss/data/`; does **not** add model weights
to GitHub and does **not** deploy anything. The original pretrained
foundation is loaded remotely from Hugging Face and must pass the
checkpoint's original-foundation parameter-hash check.

## 2. Deploy the EXACT custom model

First verify your Baseten organization has available deployment access,
an L4 GPU in a region you can use, and enough remaining **free credits**.
Deployments can consume credits while running; do not presume a job is
free solely because there is a trial credit balance.

```bash
python -m pip install --upgrade truss

cd ~/.cache/iq/phase5e-baseten-truss

# Authenticates in a browser; do not paste credentials in this chat.
truss login --browser

# Only run after checking the available credit balance and L4 access:
truss push --watch
```

This uses custom Python Truss serving, **not** an unrelated built-in Mamba
model, vLLM, or TensorRT-LLM engine. The custom model requires an NVIDIA L4
with compatible CUDA PyTorch; successful deployment/GPU correctness is not
yet proven. The CPU research recurrence uses PyTorch GPU operations, NOT
official Mamba CUDA kernel parity.

Save the actual model ID/endpoint from the successful Baseten deployment.
The historical custom-model development endpoint format is
`https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict`.
For regional environments or current dashboard URL variations, use the
**exact predict URL provided by Baseten**, not a guessed endpoint.

## 3. Run the complete 76-response comparison from one CLI command

```bash
cd ~/IQ
source .venv/bin/activate
set +e  # Avoid closing your interactive shell on a failure.

# Read your key privately; it is never written into the report.
read -rsp "Baseten API key: " BASETEN_API_KEY; echo
export BASETEN_API_KEY

# Replace this with the REAL custom-model predict URL:
export IQ_PHASE5E_PREDICT_URL='https://model-REPLACE_WITH_YOUR_ID.api.baseten.co/development/predict'

python -u scripts/benchmark_iq_mamba3_csa_hca.py \
  --predict-url "$IQ_PHASE5E_PREDICT_URL" \
  --output-dir ~/iq-phase5e-baseten-bulk \
  --fresh-pairs 8 --regression-pairs 4 --paraphrase-pairs 4 \
  --general-prompts 6 --max-new-tokens 8 --threads 2 \
  2>&1 | tee ~/iq-phase5e-baseten-bulk.log
echo "BENCHMARK EXIT CODE: ${PIPESTATUS[0]}"

unset BASETEN_API_KEY
```

The benchmark prepares 38 prompts and requests baseline and hybrid responses
from the **same deployment**, for 76 actual autoregressive text generations.
It journals results after each response and emits a consolidated
`~/iq-phase5e-baseten-bulk/summary.json`, including *both-correct* original/
swapped evidence switching, first-word answer accuracy and failures grouped
for subsequent batch fixes.

An interrupted run can resume with the **same arguments** plus `--resume`.
For remote failures (bad auth, wrong model, bad checkpoint, etc.), it stops
immediately without making dozens of failing credit-consuming requests;
the journal is retained for a subsequent full-batch resumption. A separate
output directory avoids conflicting with the earlier failed local CPU run.

## Regression checks (local, no inference credits)

Run the complete contract suite together, not prompt by prompt:

```bash
cd ~/IQ
python -m pytest -q \
  tests/iq_local/test_mamba3_csa_hca_inference.py \
  tests/iq_local/test_mamba3_bulk_audit.py \
  tests/iq_local/test_mamba3_generated_http.py \
  tests/iq_local/test_mamba3_baseten_bulk.py
```

**Research limitations:** neither the local nor GPU-mode reference has
official Mamba CUDA kernel parity certification; the attention is CSA/HCA-
inspired, not a faithful DeepSeek reproduction; and 76 tiny synthetic
generations cannot establish production reliability. Previously inspected
Phase 5e held-outs are labelled regression-only. Do not retune weights on
the new benchmark's revealed answers.

When finished, deactivate the Baseten development deployment if you
do not need it, to avoid continued credit consumption.
