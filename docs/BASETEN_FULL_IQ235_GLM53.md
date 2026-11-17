# Full IQ 235-layer GLM-tokenized Baseten experiment

## Verified target

- Full IQ architecture: 235 physical layers, consisting of **79 Mamba-3 MIMO,
  21 CSA, 57 HCA and 78 Stable LatentMoE** physical blocks
- 256 experts per MoE block, top-k 8, Block AttnRes, reasoning recurrence
- Original source: `hf://buckets/kofecheckmate/iq-glm53-all256/IQ/runs/glm53-full78-local-20261025a`
- Source config and compile-status snapshot: `configs/iq235_checkpoint_source.json`
  and `configs/iq235_source_compile_status.json`
- Phase A config changes only `model.vocab_size` **128256 -> 154880**;
  tokenizer is pinned to GLM-5.3 BF16 revision
  `abf42e6c0e53bc29a875511301e7b7fcd220315b`
- Original saved checkpoint has no packaged tokenizer and is marked
  `inference_ready: false`. **Do not treat it as a GLM-tokenized model.**

The 24-layer student is a separate experimental branch and is NOT deployed
by `deploy/baseten_full_iq`.

## Stage A — test the complete architecture BEFORE weight injection

`iq_cloud/full_iq235.py` constructs the full IQ physical stack on **eight
H200 GPUs**, preserving all 235 physical layers. It uses randomly initialized
neural weights, real pinned GLM-5.3 tokenizer IDs, and the actual Mamba-3 MIMO,
CSA/HCA, Stable LatentMoE, AttnRes and reasoning code. No CPU forward fallbacks,
expert-count reduction, reduced-depth shortcut or borrowed pretrained weights.

Each generated token must traverse all 235 layers. The response includes
schema/tokenizer SHA-256, generated GLM IDs, model geometry, and a true
`all_layers_executed` flag based on registered forward hooks. OpenTelemetry
records loading, each real physical layer and the request.

**The cloud execution is still unverified.** The model is large; the eight-H200
deployment has substantial cost. The multi-device code needs actual CUDA runtime
validation and may expose failures that CPU structural tests cannot find.

### No-GPU preflight (safe)

Run in a worktree on `feat/iq235-glm53-baseten-architecture-20261117`:

```bash
python3 -m unittest discover -s tests -p 'test_iq235_full_architecture.py' -v
python3 -m scripts.iq235_preflight
python3 -m compileall -q iq_cloud/full_iq235.py deploy/baseten_full_iq/model scripts/iq235_baseten_probe_once.py
baseten model push --dir deploy/baseten_full_iq --dry-run
```

The dry run validates upload/build request structure without creating an
inference deployment. Baseten's `H200:8` SKU and availability require validation.

### Paid Stage A (ONLY after successful dry run and cost approval)

```bash
baseten model push --dir deploy/baseten_full_iq --wait --deployment-name iq235-glm53-preinjection
```

Capture the real model/deployment IDs immediately. Store `BASETEN_API_KEY`,
`BASETEN_MODEL_ID`, and `BASETEN_DEPLOYMENT_ID` privately; never paste secrets.

```bash
python3 -m scripts.iq235_baseten_probe_once
```

This verifies a full response, saves `artifacts/iq235-baseten/architecture_only.json`,
then invokes Baseten's deactivation endpoint for **that exact deployment** even
when the prediction fails. It polls until both `INACTIVE` and zero active
replicas are confirmed. If deployment startup/build fails before the script
can begin, recover deployment IDs and execute:

```bash
python3 -m scripts.iq235_baseten_probe_once --stop-only
```

There is no guarantee that shell interruption, power loss or a crash before
capturing deployment IDs can automatically stop the cloud deployment. Confirm
the deployment status in Baseten after every run.

## Stage B — weight injection, then SECOND Baseten inference

Only after Stage A reports the actual full 235-layer pass and deactivation:

1. Preserve the pinned phase-A topology.
2. Map original source checkpoint's 128256-ID lexical tensors to the intended
   154880-ID GLM tokenizer boundary using explicit GLM lexical source tensors.
   Their 6144-wide source and IQ's 4096-wide recipient require an approved,
   deterministic adaptation and independent integrity checks.
3. Inject the full original 235-layer weight inventory with strict key, shape,
   source, dtype, finite-value and shard coverage validation; do not claim
   `COMPILE_COMPLETE` implies complete tensor integrity.
4. Create a separately named phase-B deployment, run the same full-model
   forward pass and compare schema, tokenizer and output reports.
5. Deactivate and verify zero active replicas.

**Phase B is NOT yet implemented or deployable.** No weight injection,
training, paid GPU, or Baseten account changes occurred in this commit.
