# GLM-5.3 BF16 -> IQ: Comprehensive Weight- and Capability-Transfer Portfolio

Status: research and execution contract, not a claim that every strategy has been implemented.
Donor: official \`zai-org/GLM-5.3-BF16\`, pinned revision \`abf42e6c0e53bc29a875511301e7b7fcd220315b\`.
Recipient: \`canonical_glm53_config()\` and \`IQHybridForCausalLM\`; official pinned Mamba-3 MIMO remains the recurrent/lexical foundation.

## Operating objective

Use **all technically applicable, evidence-supported transfer strategies** in controlled,
reproducible experiments. Combine methods when composable; compare mutually exclusive
weight representations as ablations and optionally ensemble/distill their strengths.
Optimize final IQ quality at a stated runtime/VRAM/storage/training budget, not a
single matrix reconstruction score. Do NOT silently replace IQ's Mamba-3,
CSA/HCA, Stable LatentMoE, recurrent reasoning or native controller operators
with donor Transformer blocks.

Two separate notions of retention must be measured:

1. **Exact donor tensor retention:** unmodified hash-verified BF16 weights still
   exist in a runtime/persistent expert path (expensive), versus lossy factors,
   adapters, or shared-weight representations (not exact retention).
2. **Behavioral knowledge retention:** causal language-model outputs, routing,
   hidden trajectory alignment, tasks, reasoning, long-context behavior.

No finite 2,048D state can losslessly encode every arbitrary 6,144D vector.
The observed basis-specific subspace floor is not a theorem about the best
learned encoder, a higher-width sidecar, or final model competence.

## Architecture-specific slots

| Region | Donor source | Canonical IQ recipient | Key caveat |
| --- | --- | --- | --- |
| Tokens, lexical | GLM native 154,880-vocab / 6,144-width assets | pinned Mamba-3 MIMO lexical foundation, 128,256 vocabulary / IQ 4,096 | token IDs are not compatible; never copy GLM logits into Mamba-token KL naively |
| Recurrent backbone | GLM attention/hidden dynamics | 32 pretrained Mamba-3 operators | preserve true Mamba-3 MIMO kernels and recurrence; learned dynamics can't be overwritten with attention tensors |
| Context | GLM MLA/DSA, indexer, norm and KV semantics | 24 CSA/HCA context operators | transfer validated operators, cache/indexer behavior and token mixing, not name-based tensors |
| Experts and router | GLM 256 routed experts, top-8, shared expert | 24 Stable LatentMoE layers, 32/64/128/256 selectable experts, top-8 | separate donor-ID coverage, router remap, shared expert, latent codec, and relative compute |
| Expert width | donor 6,144D and 2,048 intermediate | 4,096D residual and 2,048D expert latent, 2,048 intermediate | WARM's 6,144 -> 2,048 reduction is only one path |
| Native IQ extras | no direct equivalents | block AttnRes, reasoning recurrence/critic/halting, native mechanisms | recipient-native initialization, train with suitable teacher targets |

The mapped donor source layer for currently tested stage 1 is GLM layer 3;
the recipient MoE physical layer is 5. Never infer arbitrary positional
correspondences between all 78 donor layers and IQ's 24 MoE stages.

## Strategy inventory: pursue in parallel, consolidate only after evidence

Status labels: **existing primitive** means code exists but does NOT imply
an end-to-end validated production transplant. **experiment** means a measured
read-only prototype. **new candidate** requires an implementation/benchmark.

| ID | Family | Candidate method(s) | Implementation entry / proposed module | Primary measured gate |
| --- | --- | --- | --- | --- |
| T01 | Exact copy | bit-identical compatible weights, scalars/norms, partial slices and hash-proven provenance | \`iq_transfer/apply.py\`, \`mamba3_direct.py\` — existing primitives | exact semantics, shape, source/target basis and tensor hash |
| T02 | Weight subcloning | activation-/gradient-/weight-importance channels; TransMamba-style seeded initialization | \`glm53_direct.py\` — existing primitive | real held-out block outputs vs unbiased controls |
| T03 | Coordinate transfer | WARM, separate input/output bases, rotations, orthogonal Procrustes, pseudoinverse; teacher/student paired-state regression | \`warm.py\`, \`transport.py\`, \`glm53_expert_informed_basis.py\` — existing + experiment | nontrivial held-out routed RMSE and rank/spectrum |
| T04 | Output/input representation | jointly learned nonlinear codecs, residual/error paths, optional 4,096D sidecar or wider offloaded expert state | new candidate | capacity, training instability, decode overhead, teacher-space fidelity |
| T05 | Frozen donor graft | untouched BF16 expert FFN with IQ 2,048->6,144 input adapter, 6,144->2,048 output adapter; optional 4,096D path | new candidate | exact donor tensor hashes, adapter held-out/generalization, runtime footprint |
| T06 | Low-rank individual experts | standard SVD; SVD-LLM truncation-aware whitened factorization; low-rank + important sparse residual | new candidate | matched rank, activation-weighted loss, real routed token fidelity |
| T07 | Cross-expert sharing | MoE-SVD shared factors, MoBE basis experts, SLBF, expert-specific sparse/delta residual | new candidate | total bank storage, specialization, rare experts, throughput, held-out quality |
| T08 | Expert consolidation | usage-aware selection, balanced clustering/merging/splitting, Sub-MoE functional clusters, MergeMoE output fitting | \`glm53_moe.py\` usage-selection existing; others new | no systematic rare-expert collapse, remapped top-k behavior and functionality |
| T09 | Routing | temperature-, top-k-, probability-, load-, co-occurrence-, preference-matrix matching; teacher->student expert ID map | \`glm53_routing_replay.py\`, \`glm53_routed_fidelity.py\` — existing probes | real token hit-rate, weighted output KL/MAE, load balance, 0-coverage guard |
| T10 | Attention/context | MLA/DSA weights/operator transport, learned context adapters, sparse-indexer equivalence, attention->Mamba mixer initialization | \`glm53_dsa.py\`, \`mamba3_init.py\` — existing primitives | faithful effective token-mixing operator, matched contexts, causal checks |
| T11 | Architecture transfer | MOHAWK mixer alignment, layer/hidden alignment, Mamba in the Llama stepwise attention replacement, TransMamba WSAB | \`mohawk.py\`, \`iq_training/mohawk.py\` existing stages; new schedules | Mamba recurrence preserved; trained-block and E2E gains |
| T12 | Distillation | blockwise feature/gradient regression, routed-expert matching, intermediate teacher trajectories, offline teacher demonstrations, sequence-level logits | existing trainer primitives; additional loaders/objectives needed | held-out teacher/student loss and generation; no fabricated alignments |
| T13 | Different tokenizers | UTF-8 document/byte-boundary causal alignment, context-aware cross-tokenizer probability mapping, OT or likelihood scoring | \`alignment.py\`, \`mohawk.py\` hard guard existing; additional losses candidate | proven causal token alignment; never raw ID-level KL |
| T14 | Structured reduction | depth/width/head pruning, progressive shrinking, Wanda/SparseGPT, protected important channels + recovery | new candidate | task-level retention and actual target-hardware speedups |
| T15 | Training-efficient recovery | LoRA/DoRA-style residuals, alternating adapter and expert fine-tune, progressive unfreezing, QAT, rank allocation | new candidate | measured gains and trainable-parameter costs |
| T16 | Numerical serving | BF16 reference, FP8/int8/int4 candidates, AWQ/GPTQ-style activation-aware quantization, mixed-precision protected experts | new candidate | match functional/task thresholds, measure real native kernels |

The above is an extensible portfolio, **not a universal exhaustive list**.
New published methods must be screened for semantics, official implementations,
data license, hardware feasibility, and evidence before being added.

## Composition, not one winner

Try complete chains under a single experiment harness, including:

- **Preservation-first:** T05 frozen donor expert + T09 router alignment + T12
  teacher-guided adapter training -> optionally T07/T16 only after fidelity.
- **Dense information recovery:** T02 initialization + T06 activation-aware
  factors + T04 residual sidepath + T15 correction -> T12.
- **Full expert-bank route:** T08 functional clusters + T07 share factors
  + preserve donor-ID specialization via residuals + T09 routing + T12.
- **Mamba conversion:** official pretrained Mamba-3 lexical/recurrent weights
  + T10 attention/context teacher targets + T11 MOHAWK stages + T13
  tokenizer-safe T12 final behavioral distillation.
- **Frozen reference / teacher:** preserve exact donor FFNs as an optional
  offline sidecar to supervise progressively smaller compiled IQ operators.

**Never concatenate mutually incompatible weight maps into the same slot.**
A slot gets one executable representation at a time; combined methods may
provide initialization, correction deltas, teachers, auxiliary branches,
or staged training. Ensemble candidates are a separate deployment budget.
Checkpoint provenance records exact initialization plus all subsequent changes.

## Shared experiment protocol

For every method and composition, require one manifest with:

\`\`\`json
{
  "schema_version": 1,
  "donor_repository": "zai-org/GLM-5.3-BF16",
  "donor_revision": "abf42e6c0e53bc29a875511301e7b7fcd220315b",
  "recipient_config_fingerprint": "<verified-fingerprint>",
  "source_layer": 3,
  "recipient_stage": 1,
  "recipient_physical_moe_layer": 5,
  "method_ids": ["T02", "T06", "T15"],
  "donor_ids": [77, 87, 246, 131],
  "training_ids": [77, 87, 246],
  "heldout_ids": [131],
  "real_capture_sha256": "<hash-of-actual-tensor>",
  "router_topk_sha256": "<hash-of-official-router-topk>",
  "token_alignment_manifest_sha256": null,
  "weight_hashes": {},
  "training_seed": 0,
  "param_budget": null,
  "activation_memory_budget_bytes": null,
  "output_artifacts": {},
  "evaluation_report_sha256": null,
  "checkpoint_promoted": false
}
\`\`\`

The angle-bracket fields are **schema examples**, not real hashes/artifacts.
Production runners must reject placeholders rather than treat this JSON
example as valid configuration.

Evaluation ladder, always with controlled ablations:

1. Structural semantics and tensor SHA256; exact tensor budget; CPU smoke;
   numerical parity and finite gradients.
2. Donor-block fidelity on identical tokens and teacher top-k-selected
   routed tokens. Compare both original and candidate subspace floors,
   projected-input oracle, within-subspace error and rare-expert behavior.
3. Held-out by token, **document** and **donor expert**; store sample counts
   and confidence intervals. Two routed examples for donor 131 are not a
   reliable generalization estimate.
4. Full routed-MoE outputs including teacher router probabilities/top-k
   weights and shared expert; context residual delta; mixed stage interactions.
5. IQ block-level training with frozen pretrained Mamba-3 lexical and
   recurrent paths unless explicitly ablated.
6. End-to-end student training and validation: language perplexity,
   code/math/reasoning, chat preference/instruction following, long-range
   recall, latency, peak memory, checkpoint size, context/kernels.
7. Statistically compare improvements against initialization-only, WARM,
   random-seed and matched-parameter alternatives; retain Pareto-optimal
   candidates and compatible composites.

Use an unmodified validation split for candidate selection, and reserve a
separate final test set. A heldout **expert** is not automatically heldout
**data**. Report both macro (per-expert equal) and micro (routed-frequency)
statistics. Avoid misleading per-token normalizations.

## Findings motivating new approaches

The stage-1 16-column WARM candidate improved routed-token isolated FFN
relative RMSE on 723 real teacher tokens (donor IDs 77:58 routed, 87:71,
246:15, 131:2), but final relative RMSE remained ~0.74-0.78.

The **candidate-specific** output subspace floor accounted for about
83%-87% of squared FFN output error across these four donor samples;
BF16 weight-roundtrip deviation was about 0.001-0.0014 relative RMSE.
Thus the next priority is **more expressive representations** and true
teacher/student training, rather than more 16-column basis rotations.
These values are not evidence of end-to-end model accuracy loss.

## Resource and operational constraints

- The workstation's RTX 2060 SUPER has ~8GiB VRAM with 16GiB RAM plus
  swap. Limit local runs to streamed donor tensors, CPU small-matrix
  experiments, tiny bounded batches, one donor/expert/layer at a time,
  and small adapter training. Do not attempt to load 743B BF16 teacher.
- Use pinned official GLM BF16 tensor ranges (HTTP 206, bounded memory),
  canonical Mamba-3 weights and tokenizer on the recipient.
- Retain the verified official HF calibration object(s) unchanged; write new
  candidate results to separate, immutable, hashed directories. Never
  overwrite canonical WARM, worktree edits, or previously saved experiment.
- GPU/TPU workers are needed for larger collective expert-bank updates,
  multi-stage distillation and real end-to-end benchmarks. Estimate cloud
  cost and memory before dispatch.
- This document by itself **does not** execute, validate or claim success
  for any new algorithm; execution requires code, data, and benchmarks.

## Research anchors (adapt to IQ; do not assume reproduction)

- MOHAWK, Transformer -> SSM: https://arxiv.org/abs/2408.10189
- Mamba in the Llama: https://github.com/jxiw/MambaInLlama
- TransMamba WSAB / subcloning: https://arxiv.org/abs/2502.15130
- MoE-SVD: https://proceedings.mlr.press/v267/li25az.html
- Sub-MoE: https://arxiv.org/abs/2506.23266
- Shared Low-rank Basis Factorization (October 2026): https://arxiv.org/abs/2610.09342
- Mixture-of-Basis-Experts: https://arxiv.org/abs/2508.05257
- SVD-LLM: https://arxiv.org/abs/2403.07378
- Wanda: https://arxiv.org/abs/2306.11695
- DWA-KD tokenizer-aware objectives: https://aclanthology.org/2026.findings-eacl.181/
- Cross-Tokenizer Likelihood Scoring (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/40d50fe20d5e6922dbb52d7da6de03f1-Abstract-Conference.html

## First execution wave (not a winner-take-all choice)

1. Capture diverse router-conditioned GLM hidden states with independently
   held-out documents. Extend provenance with document offsets and per-token
   router **weights**, including rare donor IDs.
2. Implement T05 frozen-weight adapters in a bounded four-expert CPU/GPU
   benchmark. Preserve donor tensor hashes, compare exact donor-space FFN
   outputs and measure memory cost.
3. Implement T06 activation-aware truncated SVD and T07 shared expert basis
   baselines at matched aggregate parameter budgets; check rare expert 131
   with sufficient additional routed tokens.
4. Implement T09 student-router mapping with teacher mixture targets, then
   T12 blockwise FFN and context+FFN training. Keep training and validation
   document sets separate.
5. Bring T11 and T13 end-to-end teacher/student paths online only after
   byte-aligned source/recipient batches and actual Mamba-3 MIMO kernel
   numerics are verified. Train each candidate and later their compatible
   combinations, selecting the Pareto frontier.
