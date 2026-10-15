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

For the **same exact input token sequences**, independently capture IQ's
`norm_out` (the actual input to its CSA module) from the target training
run, writing `iq_attention_input` `[B,S,4096]` plus identical `token_ids`
to separate safetensors files. Merely projecting GLM inputs and calling
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
indexer's **causal teacher top-k recall**, and an actual **future-token
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
