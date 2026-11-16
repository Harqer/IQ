# IQ GLM-5.3 foundation: audit and correction contract

This is a **read-only architectural audit**. It does not authorize launching GPUs,
editing existing checkpoint buckets, or assuming that tensor loading proves
language-model quality.

## What the Mamba-3 MIMO reference really does

The official `state-spaces/mamba3-mimo-1.5b` is a separately **pretrained
causal LM** with 24 Mamba-3 MIMO mixer + gated-MLP blocks, hidden size 2048,
and tied input/output embeddings. It was trained on 100B FineWeb-Edu tokens
using the **Llama 3.1 tokenizer**. It **did not transplant Llama Transformer
weights** into its recurrent blocks. Llama was a tokenizer reference, not the
model's weight donor.

Its Hugging Face repository exposes `config.json` and `pytorch_model.bin`,
but no tokenizer files. The model card loads its tokenizer from
`meta-llama/Llama-3.1-8B`.

References:
- https://huggingface.co/state-spaces/mamba3-mimo-1.5b
- https://arxiv.org/abs/2603.15569
- https://huggingface.co/zai-org/GLM-5.3-BF16/blob/main/config.json
- https://github.com/jxiw/MambaInLlama (a **different**, distilled-Transformer-to-Mamba methodology)

## The current GLM all256 artifact is Mamba-founded, not GLM-founded

Audited paths: `iq_transfer/complete_transplant.py`,
`iq_transfer/glm53_compile.py`, `iq_transfer/mamba3_direct.py`,
`iq_model/hybrid.py`, and `scripts/run_glm53_mamba_foundation_transfer.sh`.

| Provenance | Current all256 | GLM-5.3 donor |
|---|---:|---:|
| Vocabulary size | 128256 | 154880 |
| Hidden width | 4096 | 6144 |
| Physical layers | 235 | 78 Transformer layers |
| Original lexical / output weights | Mamba-3 | GLM-5.3 |
| Original recurrent mixers | Mamba-3 | None |
| Transformed attention + MoE weights | GLM-5.3 | GLM-5.3 |

`glm53_compile.py` writes Mamba-derived `embed_tokens.weight`,
`norm.weight`, and `lm_head.weight` into `model-global.safetensors`.
It transforms GLM context and MoE into the recipient hidden basis. Its
manifest method is `mamba3_foundation_plus_glm53_capability_weight_transfer`.
The Mamba donor is being used as a **pretrained lexical and recurrent
foundation**, not only an architectural reference.

The current all256 compile marker explicitly records:
`FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED`,
`inference_ready=false`, `tokenizer_packaged=false`, and
`end_to_end_knowledge_retention_proven=false`. Calibration is
`weight_only`; 13 shard headers were sampled, not every tensor validated.

**Concrete build bug:** `run_glm53_mamba_foundation_transfer.sh` requests
`tokenizer.json` and `tokenizer_config.json` from the Mamba-3 donor
repository, which does not contain them. This needs a separate tokenizer
source. It is not proof of bad token IDs in the existing artifact.

**Misleading validation path:** `scripts/iq_hf_inference.py` currently mounts
`kofecheckmate/mocha`, not the all256 checkpoint; its earlier incoherent
output is evidence about mocha only.

## Distinguish three separate failures

1. **Token-ID mismatch:** changing IQ's tokenizer alone to GLM's is
   invalid: GLM IDs and EOS IDs exceed the current 128256-entry IQ lexical
   table. The existing Llama tokenizer may be compatible with the *Mamba*
   embeddings and LM head, provided the complete ID mapping matches.
2. **Representation mismatch:** GLM experts/attention were trained on
   GLM's residual basis; transformed 6144→4096 weights cannot be presumed
   function-equivalent on Mamba-produced hidden states.
3. **Architectural/function loss:** Mamba's official donor has mixer + gated
   MLP per block, whereas the hybrid transfer redistributes mixer layers and
   inserts GLM modules and IQ controllers. Shape/key coverage is not
   functional equivalence.

## User-selected direction: GLM is the capability foundation

A new build—not a relabel or silent edit of all256—should:

1. Start with an independently validated original GLM-5.3 teacher,
   including its **own tokenizer, 154880-token embedding/head and 6144-wide
   residual stream**.
2. Insert Mamba-3 MIMO recurrent branches **without copying Transformer QKV
   into SSM state weights**. For a function-preserving starting point, add
   zero-gated residual branches while initially retaining original GLM
   attention/MLP behavior. This starts equivalent to GLM, not magically more
   efficient.
3. Train/distill the Mamba additions and/or replace attention incrementally
   against verified GLM teacher states and logits **in GLM token space**.
   Layerwise alignment, teacher-student KL and language loss, and held-out
   quality tests are needed. Mamba recurrent kernels still require correct
   stateful decoding implementation.
4. Remove/reduce GLM attention **only after** measured quality retention.
   Deployment memory and serving costs remain high until then.

The Mamba-3 paper's hybrid experiments trained hybrid layers jointly; the
paper does **not** validate untrained transfer of independently pretrained
models. `MambaInLlama` addresses a similar *distillation* problem, but it
uses Mamba-2, not proof that GLM→Mamba-3 works weight-only.

## Safe audit command

```bash
python scripts/audit_iq_foundation_contract.py /path/to/readonly/all256-run --intent glm53
python -m unittest discover -s tests -p 'test_iq_foundation_contract.py' -v
```

The checker reads only local JSON metadata and index entries, and **must fail**
on existing all256 because its foundation does not match GLM-first intent and
it is not inference-validated. Passing metadata checks never proves coherent
generation.

**Do not** swap tokenizer files alone, change all256's config/manifest to
claim GLM provenance, launch Baseten, or overwrite the existing bucket.
Any new GLM-first compiler must output to a new immutable location with
explicit tokenizer/lexical source revision, tensor manifest, training
provenance, and validated generation-quality status.