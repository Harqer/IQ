# GLM-5.3 weight expansion with frozen Mamba-3

Status: source coverage planning and opt-in expanded checkpoint compiler.
No new weights have been transferred yet.

## Actual original checkpoint

Source: `hf://buckets/kofecheckmate/mocha`.
The source-layer map selects 24 of GLM-5.3's 78 layers:
`0,3,6,10,14,17,22,23,26,30,34,37,42,44,46,50,54,57,62,64,66,70,74,77`.

The GLM donor has 3 dense and 75 sparse stages, with 256 routed
experts in each sparse stage. IQ has 32 Mamba, 24 context and 24 MoE
physical stages, with 32 experts per selected MoE stage.

- Mapped: 24/78 GLM stages (1 dense + 23 sparse)
- Omitted: 54 GLM stages (2 dense + 52 sparse)
- Original routed expert slots: 23 × 32 = 736 / 19,200
- Expanded candidate at 64: 23 × 64 = 1,472 / 19,200
- Expanded candidate at 128: 2,944 / 19,200
- Expanded candidate at 256: 5,888 / 19,200

These counts measure parameter source coverage, not language quality.
The old checkpoint does not record exact selected donor expert IDs;
do not claim a new compile preserves each of the previous 32 experts
until donor selection and serialized weights are compared.

## New tools

- `scripts/plan_glm53_weight_extension.py`: fetches GLM and IQ
  metadata, checks the source-layer map, and produces a JSON coverage plan.
- `canonical_glm53_config(routed_experts=64)`: opt-in configuration.
  Valid counts are 32, 64, 128, or 256. All other geometry is frozen.
- `iq_transfer.cli glm53-bootstrap-calibration --routed-experts 64`:
  generate calibration compatible with the candidate configuration.
- `iq_transfer.cli glm53-plan-compile-shards --routed-experts 64`:
  calculate exactly which donor shards the candidate compile needs.
- `iq_transfer.cli glm53-compile --routed-experts 64`:
  build into an EMPTY, NEW output directory, after donors and
  matching calibration are available. The original checkpoint remains
  unchanged. New manifests include selected donor expert IDs.
- `scripts/verify_mamba_frozen.py`: compare hashes of all 32 Mamba
  checkpoint shards and the pretrained lexical/global shard before
  treating a candidate as preservation-safe. Identical weights do not
  guarantee identical activations in the hybrid model.

## Run the CPU-only metadata audit

```bash
python scripts/plan_glm53_weight_extension.py --routed-experts 64
python -m unittest discover -s tests -p 'test_glm53_weight_extension.py' -v
python -m unittest discover -s tests -p 'test_glm53_expert_capacity.py' -v
python -m unittest discover -s tests -p 'test_glm53_mamba_preservation.py' -v
```

The planning script uses Hugging Face metadata only; no GPU job.
To use local copies, specify `--donor-config`,
`--transplant-manifest` and `--iq-config`.

## Explicit boundary: unmapped GLM layers

Adding experts to the 24 existing GLM-derived stages does NOT transfer
the other 54 GLM layers. Mapping all 78 source layers would require
new physical context+MoE slots, and hence a different model topology.
It would also violate the current Mamba-dominance condition with only
32 Mamba physical slots. Such an expansion must be designed, validated,
and budgeted separately; do not silently rewrite the production
architecture or claim mathematically exact donor preservation.

Likewise, increasing expert capacity may greatly increase model size,
memory use, inference latency and compile work. No new model should
be promoted without tokenizer checks, Mamba hash parity, first-token
logit/activation diagnostics and held-out language evaluation.

No compute allocation, training job, model download or checkpoint
overwrite is performed by these tools.
