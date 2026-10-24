# GLM-5.3 → IQ: complete functional transfer, staged and verifiable

**Current status:** the immutable `hf://buckets/kofecheckmate/mocha`
checkpoint was built with **weight-only calibration**, and its first chat
inference produced incoherent text. Strict tensor coverage and an HTTP 200
are not evidence of retained GLM language capability. It is **not** a
complete functional transfer.

The donor is 744B total / 40B active, whereas the student is about 16.9B
stored parameters. A full *literal* copy of donor weights into a 16.9B
architecture is mathematically impossible; the intended objective is
measurable capability retention via calibrated transformation + learning.
Do **not** silently expand IQ to 744B or declare "complete" from saved
safetensors alone.

## What the repository already implements

- `iq_transfer/glm53_compile.py`: 24 remapped GLM stages into the 80-layer
  Mamba-3+GLM hybrid. 12.4B of approximately 16.9B student parameters are
  derived from the donor's context/MoE family.
- `iq_transfer/glm53_calibration.py`: supports weight-only bootstrap and
  activation-refined maps; mounted `mocha` is **weight_only**.
- `iq_transfer/alignment.py`: causal UTF-8 byte alignment for the
  different GLM and IQ tokenizations.
- `iq_training/mohawk.py`: actual Stage-2 trainable IQ context/MoE
  residual alignment; refuses missing aligned teacher targets.
- `scripts/iq_hf_inference.py`: authenticated HF Job with H200 cancellation
  and a real generation smoke gate.

## Added this iteration (preparations, not completed training)

1. `scripts/capture_glm53_teacher_data.py`: **dry-run by default**, cost
   acknowledgment required with `--execute`; records provider-returned
   model ID, request ID, and text. Does **not** attest the host's revision.
2. `iq_training/behavior_transfer.py` and
   `scripts/prepare_glm53_behavior_data.py`: take **actual** GLM
   responses, re-tokenize using **IQ's tokenizer**, mask question tokens,
   disjointly split teacher examples, hash the files, and reject truncation.
3. `scripts/train_iq_glm53_behavior.py`: bounded response-CE correction of
   one selected context+MoE pair and final norm with held-out validation.
   Saves **only a candidate delta**, never overwriting `mocha`.
4. `iq_transfer/stage2_artifacts.py`: builds real causal GLM hidden-state
   targets from teacher token byte offsets, IQ token offsets, and a held-out
   validated `residual_out` coordinate map. Rejects future-token leakage.
5. `scripts/train_iq_glm53_stage2.py`: uses existing
   `IQGLM53MohawkStage2` on separate train/validation aligned captures and
   saves a candidate correction delta.
6. `iq_training/transfer_overlay.py`: in-memory checkpoint overlays
   require parent-index hash, tensor hashes, allowed layer names, matching
   shapes and finite values. Candidate status remains **unpromoted**.
7. The H200 launcher can mount a **read-only candidate-delta bucket** for
   a future, explicitly approved generation check. No paid job is launched
   by installing or testing this code.

## 1. CPU-only test and input-validation commands

Use an environment with `torch`, `numpy`, `safetensors`,
`transformers`, and `huggingface_hub` installed.

```bash
python -m unittest discover -s tests -p 'test_glm53_behavior_transfer.py' -v
python -m unittest discover -s tests -p 'test_glm53_stage2_artifacts.py' -v
python -m unittest discover -s tests -p 'test_glm53_transfer_overlay.py' -v
```

These are CPU-level tests with tiny mathematical fixtures. They do not prove
the 744B teacher was run, that GLM activations were captured, or that IQ
generates coherent text.

## 2. Collect real teacher response examples

Create `prompts.jsonl` with **real chosen prompts**, one JSON object with
exactly a nonempty `prompt` property per line. Use multiple distinct prompts;
the output needs separate train/validation examples. Avoid secrets/PII.

```bash
# Default is metadata-only, no API call:
python scripts/capture_glm53_teacher_data.py \
  --prompts prompts.jsonl --out glm53-responses.jsonl \
  --teacher-revision abf42e6c0e53bc29a875511301e7b7fcd220315b

# Only after you approve API spending and have a valid Z.ai credential:
# export ZAI_API_KEY=...  # Set privately; never paste keys into chat/logs
# python scripts/capture_glm53_teacher_data.py [same flags] --execute
```

This uses Z.ai's documented OpenAI-compatible
`https://api.z.ai/api/paas/v4/chat/completions` endpoint, with
provider model `glm-5.3`. **That hosted model is not authenticated as the
exact pinned BF16 weight snapshot.** Treat its outputs as behavioral
supervision, not tensor/operator-equivalent target logits.

Prepare IQ-side supervised tokens (requires access to the **same**
Llama-3.1 tokenizer previously mounted at `/tokenizer`):

```bash
python scripts/prepare_glm53_behavior_data.py \
  --input glm53-responses.jsonl \
  --tokenizer-dir /path/to/iq-tokenizer \
  --output-dir ./iq-glm53-behavior-data \
  --max-tokens 256
```

**Do not directly KL-diverge GLM logits against IQ logits.** Their
vocabularies, token IDs and boundaries are different. The existing
`iq_transfer/mohawk.py` correctly rejects that approach.

## 3. Train a candidate IQ response correction (requires separate authorized CUDA)

First validate the files and source checkpoint **without CUDA**:

```bash
python scripts/train_iq_glm53_behavior.py \
  --checkpoint /mounted/mocha \
  --data /mounted/iq-glm53-behavior-data \
  --tokenizer-dir /mounted/iq-tokenizer \
  --out-dir /mounted/new-run \
  --dry-run
```

Remove `--dry-run` **only** on explicitly budgeted CUDA compute.
Do not run the full IQ student training on an 8 GB GPU. Runtime Mamba-3
backward compatibility and real memory budget still require validation.
Outputs are versioned **candidate correction layers**, not a newly
certified general language model.

## 4. Stage-2 hidden alignment — required for actual operator retention

Capture paired GLM-5.3 and IQ document activations on **genuinely identical
UTF-8 bytes** with per-model token byte spans. Fit a GLM
`residual_out` → IQ `residual_out` map on fit records, retaining
held-out validation diagnostics. Use
`iq_transfer.stage2_artifacts.write_stage2_example` to construct
separate train and validation artifact directories for an exact pair of
compiled source/physical target layers.

The upstream GLM activations are **not present in the currently mounted IQ
bucket**, nor can they be reconstructed from the transplanted IQ weights.
GLM BF16 is vastly larger than one H200; a teacher capture requires a
suitable distributed setup or a trusted activation-export service.

The existing trainable adapter can then be used:

```bash
python scripts/train_iq_glm53_stage2.py \
  --checkpoint /mounted/mocha \
  --train-alignment /mounted/alignment-train \
  --validation-alignment /mounted/alignment-validation \
  --out-dir /mounted/stage2-candidate \
  --dry-run
```

Remove `--dry-run` only with authorized training compute. Repeated
Stage-2 alignment across mapped source/recipient layers, and end-to-end
teacher-response CE adaptation, will likely both be needed; exact
retention is **not guaranteed** by either.

## 5. Evaluate and promote, never overwrite source

Upload a candidate-delta directory to a **separate** HF bucket, containing
its manifest and safetensors file, then (on an explicitly approved H200 run)
use the canonical launcher with

```bash
python scripts/iq_hf_inference.py \
  --run-h200 --accept-unvalidated-checkpoint \
  --candidate-delta-bucket owner/new-candidate-bucket
```

This performs real QA and cancels on failure. Do not use
`--interactive` until the quality canary passes.

**Promotion gate:** require multiple unseen text probes, real validation
cross-entropy improvement, measurable held-out reasoning/coding retention,
provenance consistency, no mask leakage, and reproducible training
checkpoints. Only then produce a new versioned IQ model artifact. One
successful arithmetic response is only a smoke test.

## Verified source documentation

- Official GLM-5.3 744B-A40B release:
  https://github.com/zai-org/GLM-5
- GLM BF16 donor:
  https://huggingface.co/zai-org/GLM-5.3-BF16
- Official Mamba-3 1.5B donor:
  https://huggingface.co/state-spaces/mamba3-mimo-1.5b
- Hugging Face Jobs persistent bucket mounts:
  https://huggingface.co/docs/huggingface_hub/main/en/guides/jobs
- Z.ai API:
  https://docs.z.ai/guides/capabilities/mcp-call

**Paid compute was not initiated by this implementation.**
