# Modal $30-credit experiment: full IQ 235-layer Phase A

**Target:** Exact full 235-layer IQ architecture with 79 Mamba-3 MIMO, 78 CSA/HCA,
78 Stable LatentMoE with 256 experts each, and the pinned GLM-5.3 tokenizer
(154880 IDs). This is the **pre-injection architecture test**: all model weights
are test-initialized and the existing 128256-token checkpoint is **not** loaded.

Modal GPU access **requires a payment method on file**, even when Starter
includes $30 monthly compute credit. Successful `modal setup` verifies
authentication, **not** GPU billing authorization:
https://modal.com/docs/guide/gpu

## Mandatory cost controls before testing

Set both in the Modal dashboard **Usage & Billing**:

- **Workspace budget:** $30 or less gross usage; limits spending *before* credits
- **Workspace spend limit:** $0 net charges, if supported by your account;
  limits out-of-pocket charges *after* credits

See https://modal.com/docs/guide/budgets .
8 × H200 GPUs cost 8 × $0.001261/s = **$0.010088/s**, $0.60528/min
in GPU charges, plus CPU/memory and any billed build time. The script sets
a **480-second function timeout**, 180-second container startup timeout,
one single-use container, zero minimum containers, and no detached run.
These timeouts alone are not a dollar-based spending cap.

The model's large expert count means loading can be slow, and a code/build
failure might consume credits without generating a token. The native Mamba-3
package is installed using a pinned Git VCS URL, retaining PEP-610 source
revision metadata; then its installed Python TileLang kernel receives a
verified one-line tail correction. Both source revision and tail guard are
checked at model startup. The application
and model are **not yet validated on Modal**.

## Authenticate and validate (no GPU allocation)

```bash
cd "$HOME/Downloads/IQ-glm53-student"
python3 -m venv "$HOME/.venvs/modal"
"$HOME/.venvs/modal/bin/python" -m pip install -U modal
"$HOME/.venvs/modal/bin/modal" setup
"$HOME/.venvs/modal/bin/modal" billing summary

python3 -m unittest discover -s tests -p 'test_iq235_modal.py' -v
python3 -m scripts.iq235_preflight
```

## Start the actual 8-H200 full-IQ test — billed usage

Do this **only after confirming the workspace budget and spend limits**:

```bash
"$HOME/.venvs/modal/bin/modal" run deploy/modal_iq235_phase_a.py
```

Do **not** use `modal run --detach` or `modal deploy`. One ephemeral App and one
single-use Function will load the 235-layer model from staged source, use the
GLM tokenizer, attempt one complete token forward pass, record per-layer
OpenTelemetry spans to the console, write evidence to
`artifacts/iq235-modal/architecture_only.json` **only on success**, and
exit. A failed run does not prove model execution. No checkpoint weight
injection occurs in this phase.

If errors occur, preserve the Modal log and code SHA. The remaining Phase B
checkpoint-remapping and tensor-integrity code must be completed separately
before another GPU run.
