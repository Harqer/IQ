# Baseten full 24-block GLM-tokenized IQ student: test, shut down, inject, retest

This directory is shipped **through GitHub**, not a chat ZIP. It does not
modify `IQ-baseten-full` or the separately stored 235-layer checkpoint.

## Phase A (no donor weight injection)

The Truss config launches all 24 original Mamba-3 MIMO blocks by structure,
uses 154,880-ID GLM tokenizer, random test embeddings/head and test-initialized
backbone parameters. It executes each physical block for every generated token.
Outputs will be nonsensical; this is a real architecture-execution probe, not a
reduced model or pretrained checkpoint test.

```bash
cd "$HOME/Downloads/IQ-glm53-student"
git fetch origin
git merge --ff-only origin/refactor/iq-glm53-student-consolidation-20261117
python3 -m unittest discover -s tests -p 'test_iq_cloud_phase.py' -v
python3 -m compileall -q iq_cloud deploy/baseten_student/model scripts/iq_baseten_probe_once.py
baseten model push --dir deploy/baseten_student --dry-run
```

**Stop after dry-run** to inspect successful validation and check live H100
capacity/credit allowance. `--dry-run` does not deploy. Do not execute a
paid push without explicit authorization.

When authorized to deploy and after a successful dry-run:
```bash
baseten model push --dir deploy/baseten_student --wait
```

Set `BASETEN_API_KEY`, `BASETEN_MODEL_ID`, and `BASETEN_DEPLOYMENT_ID`
privately, using actual values returned by deployment (never paste secrets).
Then execute:
```bash
python3 -m scripts.iq_baseten_probe_once --expect-phase architecture_only
```

The script records `artifacts/iq-cloud-probes/architecture_only.json`,
then **requests deactivation even when inference fails**, and polls for
`DEPLOYMENT STOPPED: INACTIVE`. If this confirmation is absent, manually
deactivate via Baseten immediately.

## Phase B (only after A is stopped)

Stage pinned `zai-org/GLM-5.3-BF16` tokenizer, embedding and output-head
shards and `state-spaces/mamba3-mimo-1.5b` pretrained backbone weights.
Set `IQ_PHASE=injected` and `IQ_GLM_ROOT` to a valid mounted lexical root
in a separate deployment. The existing `scripts/iq_glm53_mamba3.py` loader
checks Mamba SHA-256 and lexical tensor geometry. **The Phase B mount is not
configured yet**, so the injection deployment must not be pushed before its
real artifacts and mounts are validated.

Use the same one-shot script with `--expect-phase injected`, then compare
schema and tokenizer fingerprints from both runs.

No local GPU is required. Baseten compute can be billed; no deployment is
started by adding these files.
