# IQ cross-architecture initialization: execution contract

**Recipient:** the frozen IQ hybrid architecture on `feat/glm53-cross-architecture-transfer`.
**Pretrained foundation:** official Mamba-3 MIMO 1.5B, revision
`bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec`.
**Owner-provided Hugging Face Bucket:** `hf://buckets/kofecheckmate/mocha`.

## Actual weight ownership

| IQ module | Source | Transfer rule | Verified state |
| --- | --- | --- | --- |
| Token embedding, final norm, LM head | Mamba-3 MIMO | exact 2048→4096 replication with head scaling | implemented by `mamba3-direct` |
| Recurrent MIMO mixer and pre-norm | Mamba-3 MIMO | replica-space operator transport, rank 4, 24 donor layers → 32 IQ slots (8 exact identities) | implemented by `mamba3-direct` |
| CSA/HCA, DSA indexer | GLM-5.3 BF16 | MLA/DSA operator transformations + calibrated maps | implemented in `glm53_compile.py`; full real-donor transfer not yet validated |
| Stable LatentMoE | GLM-5.3 BF16 | expert usage selection, latent maps, MoE routing transform | implemented in `glm53_compile.py`; capability retention not proven |
| Block AttnRes, halting, reasoning verifier | IQ-native | controlled initialization + training | **not** borrowed from Mamba or GLM |

**Shape equality is necessary but insufficient:** before copying, enforce operator semantics, basis, checkpoint revision, layer position and causal behavior. A matching 4096×4096 tensor from an unrelated model is not a valid replacement.

## Execute the actual Mamba→IQ foundation step on Kaggle

The owner has confirmed `mocha` contains `config.json` and
`pytorch_model.bin` at its root. Those **original donor weights** are not
already IQ-transformed weights.

In a Kaggle notebook, enable Internet, use an authenticated HF CLI session as
`kofecheckmate`, and clone the relevant branch. From the repo root after
installing `requirements-model.txt`, run:

```bash
python scripts/transfer_mamba_bucket_foundation.py
```

The runner:
1. Copies **only** `config.json` and `pytorch_model.bin` from the bucket root.
2. Requires the pinned donor SHA-256, not just matching tensor shapes.
3. Runs the existing `iq_transfer.cli mamba3-direct` compiler.
4. Validates 24 transplanted layer shards, eight identity slots, global weights and provenance.
5. Uploads to `hf://buckets/kofecheckmate/mocha/iq-mamba-foundation/` without `--delete`.
6. Downloads **every** published shard and checks SHA-256 against the local output.

No GLM weights are required for this phase. This runner publishes the **Mamba
foundation overlay**, not the complete runnable IQ checkpoint; do not represent
it as finished model assembly or validated generation.

## Full IQ checkpoint (subsequent phase)

Use `glm53-bootstrap-calibration` with real or explicitly labeled
weight-only calibration, and then `glm53-compile` from
`iq_transfer/cli.py`. The canonical recipient retains Mamba ownership of
the lexical and recurrent layers. GLM-5.3 BF16 contributes transformed
attention/DSA/MoE only. It must pass full loader-key coverage, causal-prefix
alignment, held-out activation error, checkpoint round trip and generation
evaluation before promotion.

Reference implementations:
- `iq_transfer/mamba3_direct.py`
- `iq_transfer/glm53_calibration.py`
- `iq_transfer/glm53_compile.py`
- `tests/test_mamba3_direct.py`
- `tests/test_mamba_bucket_foundation.py`

Hugging Face Buckets command semantics:
https://huggingface.co/docs/huggingface_hub/guides/buckets
