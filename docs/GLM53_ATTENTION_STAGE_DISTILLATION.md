# Stagewise GLM-5.3 MLA/DSA → IQ CSA attention alignment

## Authoritative operator and why direct copying is blocked

- [NVIDIA GLM5Bridge](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html)
- [Hugging Face GLM MLA/DSA operator source at Transformers v5.15.0](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
- [NVIDIA creating a custom Bridge/provider](https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html)

IQ combines Mamba-3 with its own CSA operator in each of 78 fused blocks.
Source GLM-5.3 instead uses MLA + DSA with distinct Q/K/V expansions,
RoPE, indexer weights and width 6144. IQ's CSA is width 4096 with a
different output compressor; no 1:1 physical parameter map exists.
Do not use `AutoMapping` or tensor reshaping as "pretrained transfer".

## Implemented by this branch

- `scripts/evaluate_glm53_attention_teacher.py` now requires captured
  `token_ids` alongside `source_layer_input`, and emits token IDs with
  the **official** HF attention outputs and selected DSA indices.
- `iq_transfer/glm53_attention_alignment.py` requires paired GLM teacher
  attention outputs and IQ CSA inputs from the *same token sequences*,
  a real fitted source→target coordinate map with held-out diagnostics,
  **disjoint** train and validation token sequences, and finite values.
  Its objective is differentiable MSE against measured source outputs
  mapped into IQ's residual width, with the **best held-out state** retained.
- `scripts/distill_glm53_attention_stage.py` loads real read-only
  source/target safetensors capture sets and verifies source file hashes,
  donor config identity, indexer owner, token IDs, source stage and
  a fitted coordinate map. It loads an existing IQ CSA branch or requires
  explicit `--start-random`; it will not silently interpret random
  initialization as transferred GLM weights. It writes only the newly
  trained CSA state safetensors plus a provenance/checksum/quality manifest.
- CPU tests exercise the real CSA gradient and optimizer; a separate
  pinned `transformers==5.15.0` CI job also exercises gradient fitting
  from the *actual official HF GlmMoeDsaAttention operator* using a small
  synthetic **test-only** parameterization, which does not claim source
  pretrained quality.

## Essential correction: GLM and IQ have different tokenizers

GLM-5.3 uses a **154,880-token** vocabulary while the current IQ
Nemotron lexical configuration uses **131,072 tokens**. Requiring equal
`token_ids` between the two models is **not a valid default** and
usually cannot work. Hugging Face fast tokenizers provide
`return_offsets_mapping=True` with `(char_start, char_end)` for each
token; see [official tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer).
This does not establish semantic equivalence across different token
segmentation; it enables a **measurable, deliberately lossy supervised
alignment**.

`iq_transfer/glm53_attention_capture.py` provides
`fast_tokenizer_text_provenance(tokenizer, texts)` and
`capture_attention_input(module, ...)`. Use the *actual* official GLM
and IQ tokenizers and pass precisely their returned token IDs as each
model's forward input. The capture pre-hook records genuine
`self_attn`/`attention` input tensors, tokenizer-specific token IDs,
character spans and the same raw UTF-8 text SHA-256 for both models.

The stage evaluator copies the source's `token_offsets` and
`text_sha256` to the official teacher output. The distillation CLI
aligns teacher outputs onto IQ token positions with
`completed_source_token_char_overlap_v1`: a GLM source token may
contribute to an IQ token only when it **ends by the IQ token's end**
and overlaps that IQ token's character span. Tokens with no completed
source overlap (including special tokens or incompatible segmentation)
are masked out of MSE and are reported as lower supervision coverage.
If coverage is below `--min-span-coverage` (default 0.5), training
fails rather than inventing teacher targets.

Because GLM and IQ token indices differ, **cross-tokenizer DSA top-k
recall is explicitly undefined**, not zero or 100%. Source/target
DSA selection parity needs an independently measured span-aware
evaluation and cannot be certified by matching integer indices.

A `--shared-tokenizer` override exists only for genuinely identical
lexical tokenization and exact ID sequences. Never enable it merely
because two models processed the same raw text.

## Real input format and invocation

No GPU is needed for the small CPU test or to validate existing captures.
Processing a real 4096-wide CSA with long contexts will be expensive and
may require accelerated hardware later, but **there is no cloud account or
payment prerequisite to implement these correctness gates**.

First evaluate/capture the *official BF16 source* for a full-indexer GLM
layer on a concrete token sequence. An input safetensors file must contain:

- `source_layer_input`: genuine GLM pre-attention hidden states
  `[B, S, 6144]`, float.
- `token_ids`: the exact input token IDs `[B, S]`, int32/int64.

```bash
python -m scripts.evaluate_glm53_attention_teacher \\
  --donor "$GLM_BF16_DIR" --layer 6 \\
  --source-activations "$REAL_TRAIN_GLM_CAPTURE" \\
  --output /scratch/iq/glm-attn-train-layer006

python -m scripts.evaluate_glm53_attention_teacher \\
  --donor "$GLM_BF16_DIR" --layer 6 \\
  --source-activations "$REAL_HOLDOUT_GLM_CAPTURE" \\
  --output /scratch/iq/glm-attn-val-layer006
```

For the **same original raw input text**, independently capture IQ's
`norm_out` (the actual input to CSA) from the target training run,
writing `iq_attention_input` `[B,S_iq,4096]`, **IQ's own** token IDs,
`token_offsets` and matching `text_sha256` to safetensors files.
The IQ and GLM token sequences can and typically will differ in length,
ID values and segment boundaries. Merely projecting GLM inputs and calling
them IQ activations is **not** a valid pretraining evaluation.
Capture corresponding source/target residual activation pairs and fit
a separate `CoordinateMap` using the existing
`iq_transfer.glm53_calibration`/transport code. That map must have
true **validation** diagnostics, rather than fake values.

Then execute one stage:

```bash
python -m scripts.distill_glm53_attention_stage \\
  --layer 6 \\
  --train-teacher-dir /scratch/iq/glm-attn-train-layer006 \\
  --train-iq-capture "$REAL_TRAIN_IQ_CAPTURE" \\
  --validation-teacher-dir /scratch/iq/glm-attn-val-layer006 \\
  --validation-iq-capture "$REAL_HOLDOUT_IQ_CAPTURE" \\
  --coordinate-map "$FITTED_GLM_IQ_RESIDUAL_MAP_PREFIX" \\
  --start-random --seed 2026 \\
  --steps 200 --lr 0.0001 \\
  --output /scratch/iq/distilled-csa-layer006
```

Replace `--start-random` with `--initial-student /path/to/verified/csa.safetensors`
when using an existing IQ student. This command **only** distills one
attention module; the 200-step count is a CLI example, not an endorsed
quality threshold or measured converged training run.

The output reports initial/best held-out MSE and cosine, the student
indexer's **causal teacher top-k recall when the tokenizer is identical**
(or `null` across tokenizers), and an actual **future-token
perturbation test** for causal output leakage. This is measured on held-out
activations with the real student module. It also records initialization,
provenance/checksums, and a
`trained_CSA_stage_requires_causal_and_quality_acceptance` status.
It does not claim source operator equivalence, pretrained weight
transplantation, checkpoint-ready full-model training, or GLM quality.

## Remaining acceptance gates before complete IQ integration

1. Record genuine official donor **and** IQ activations aligned by
   token sequence and exact stage, with held-out data and model revisions.
2. Validate teacher/student DSA selection, **causal** future-token
   invariance, packed and cached step semantics, and gradient parity.
   Output MSE alone is **not** sufficient to approve a weight transplant.
3. Package all 78 accepted attention stages into a deterministic
   model-wide IQ conversion manifest with exact Mamba and MoE transfer
   ownership and no missing/unverified tensors.
4. Implement IQ's custom actual Megatron `ModelProvider`, model layers,
   bidirectional `MegatronMappingRegistry`, and native TP/PP/EP full-model
   checkpointing according to NVIDIA's new-model guide. The current
   IQ EP-only saver is not the Bridge training checkpoint format.

**No user Mocha bucket objects were accessed, transformed or written.**
