# IQ: GLM-5.3 BF16 → pretrained Mamba-3 MIMO (MOHAWK recovery)

This is the production transfer direction. It intentionally **does not use Phi-4-mini**,
the legacy ELU linear-attention experiment, or a replacement SSM.

## Source of truth

- GLM donor: `zai-org/GLM-5.3-BF16` at an immutable commit.
- Recurrent foundation: `state-spaces/mamba3-mimo-1.5b` at
  `bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec`.
- Target architecture: `canonical_glm53_config()` in
  `iq_transfer/complete_transplant.py`, with the real `IQHybridForCausalLM`.
- Transfer: `scripts/run_glm53_mamba_foundation_transfer.sh`,
  `iq_transfer/glm53_compile.py`, and `iq_transfer/mamba3_direct.py`.
- Current training primitives: `iq_training.train_step`,
  `iq_training.build_optimizer`, and `iq_training.save_checkpoint`.
- Durable artifact destination defined by the transfer branch:
  `hf://buckets/Visionboxx/macha`.

The Mamba donor owns tokenization, embedding/output weights and recurrence;
GLM contributes architecture-aware CSA/HCA and MoE capability initialization.
A trained teacher is never silently reinterpreted as Mamba-3 weights.

## What this change implements

1. `iq_transfer/mohawk.py`: explicit three-stage loss contracts. Stage 1
   accepts an effective token-mixing *operator*, not hidden activations. Stage 3
   KL fails closed if the verified tokenizer hashes differ.
2. `iq_transfer/alignment.py:align_glm_states_to_mamba_tokens`: converts real,
   per-document GLM prefix states to Mamba-token positions using UTF-8 byte
   boundaries; the last completed GLM token must end no later than each student
   token. It also requires matching content hashes and a *held-out validated*
   GLM residual-output → IQ residual-output coordinate map.
3. `iq_training/mohawk.py:IQGLM53MohawkStage2`: attaches to the actual IQ MoE
   physical-layer output, optimizes hidden-state alignment, and reuses
   `iq_training.train_step` / existing optimizer/checkpointer. Mamba
   embeddings/head and recurrence are frozen by default. Adjacent recurrent
   weights can be explicitly unfrozen for a controlled ablation.
4. The canonical trainer accepts `teacher_aligned_hidden` and an **explicit**
   `teacher_alignment_mask`. Invalid/partially tokenized prefixes cannot
   silently contribute to loss.

### Stage-2 semantics

Select a GLM source layer using the real correspondence in
`GLM53CalibrationSolution.stages`. The corresponding IQ **MoE physical layer**
is the anchor because that is where IQ completes the context-plus-MLP stage.
Capture the **real post-MLP GLM residual output** and **real post-MoE IQ residual
output** on matching, licensed calibration documents and fit a dedicated
`CoordinateMap` using separate fit and held-out validation splits.

Do **not** reuse the existing `residual_in` map as proof that
`residual_out` maps correctly. A map with no held-out diagnostics is rejected.

For each un-packed document, persist the original UTF-8 content hash,
both tokenizer versions, per-token byte offsets and causal activations.
Run `align_glm_states_to_mamba_tokens` on each document, then form
`[batch, IQ-token, 4096]` targets and boolean valid-prefix masks. Do
not mix documents or leak future bytes across packed-document boundaries.

Once **real, verified** batches and IQ checkpoints have been materialized on
the GPU worker, the training primitive is:

```python
from iq_training import (
    IQGLM53MohawkStage2, TrainStepConfig, build_optimizer, train_step,
)

stage2 = IQGLM53MohawkStage2(
    student,
    source_glm_layer=selected_calibration.source_layer,
    target_physical_layer=selected_calibration.moe_physical_layer,
    alignment_manifest_sha256=verified_alignment_manifest_sha256,
)
optimizer = build_optimizer(stage2)
metrics = train_step(
    stage2,
    optimizer,
    [{
        "input_ids": real_iq_input_ids,
        "attention_mask": real_iq_attention_mask,
        "teacher_aligned_hidden": causally_aligned_and_projected_glm_hidden,
        "teacher_alignment_mask": verified_causal_prefix_mask,
    }],
    TrainStepConfig(precision="bf16"),
)
```

These variables denote artifacts that must be loaded and verified by the
production dataset preparation stage; **no placeholder artifacts are supplied
or fabricated by this branch**. The manifest hash is recorded on the adapter,
but its presence alone is not a source-data integrity check; the future
artifact loader must verify hashes on actual bytes before calling `train_step`.

## Remaining work before a paid GPU training job

- Extend per-document GLM capture to retain tokenizer byte offsets and raw
  record IDs, rather than only concatenated/flattened activation matrices.
- Measure true IQ post-MoE `residual_out` under Block AttnRes and fit a
  validated `residual_out` map, separate from input calibration.
- Add an atomic, hash-verified target artifact writer/loader and connect it to
  the pinned Mamba-token batch reader.
- Run smoke training with verified real batches, finite gradients and a
  save/resume round trip. Compare held-out loss against frozen/transplant-only
  baselines. Do not enable teacher-forcing on reasoning recurrence without
  explicit causal `reasoning_context_lengths`.
- Only promote native MOHAWK Stage 1 when a faithful **Mamba-3 MIMO effective
  mixer** and comparable GLM MLA/DSA mixing target have been derived and
  validated. An ELU linear-attention proxy is not Mamba-3.
- Only promote Stage 3 after a validated cross-tokenizer distribution alignment
  or another formally specified teacher-target objective. Equal vocabulary
  tensor dimensions do not make GLM and Mamba token IDs equivalent.
- GPU and durability gate: verify native rank-4 MIMO kernels on the selected
  GPU, load actual compiled donor/recipient checkpoints, and persist output
  through authorized remote storage. The legacy GCP script targets the retired
  NIF architecture and incorrectly labels `a3-highgpu-8g` as H200.
- The Hugging Face Jobs account currently returned HTTP 402 Payment Required
  on the attempted bucket-mount preflight. No training job has started.

Run the CPU-only unit checks for this integration with:

```bash
python -m unittest tests.test_glm53_mohawk -v
```

The unit tests validate loss gradients, frozen-parameter coverage, source
byte-prefix causality, invalid maps, and correct integration with the existing
trainer. They do **not** establish MOHAWK quality on the actual 8x H200 target.

## References

- MOHAWK / Phi-Mamba: https://github.com/goombalab/phi-mamba
- Mamba-3 official runtime: https://github.com/state-spaces/mamba
- Current IQ capability-transfer PR: https://github.com/Harqer/IQ/pull/38
