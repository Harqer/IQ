# IQ: GLM-5.3 + Full Mamba-3 MIMO Integration — Requirements and Implementation Plan

**Status:** DESIGN / REQUIREMENTS ONLY. No model rewrite, training, inference deployment, or checkpoint replacement is authorized by this document.

**Objective:** Make GLM-5.3 the pretrained language/capability foundation and integrate the complete pretrained Mamba-3 MIMO architecture without silently changing GLM's tokenizer, embedding/output vocabulary, weight meaning, computational semantics, or inference behavior. Preserve full 256-routed-expert GLM capability. No claim of exact capability retention is permitted without measured evidence.

**Existing checkpoint:** hf://buckets/kofecheckmate/iq-glm53-all256/IQ/runs/glm53-full78-local-20261025a/ is **READ-ONLY LEGACY**. Its compiler explicitly uses Mamba pretrained lexical and recurrent weights and remaps GLM context/MoE into a 4,096-wide model. It is not a GLM-founded checkpoint and reports inference_ready=false and end_to_end_knowledge_retention_proven=false. Do not overwrite it, relabel it, or confuse it with a validated new result.

## A. Decisions and non-negotiable boundaries

1. **GLM-first lexical and computation ownership.** The first new student baseline must reproduce the official GLM-5.3 teacher, preserving its own tokenizer, 154,880 embeddings/output rows, 6,144 hidden dimensions, 78-layer order, normalization, original attention/DSA and MoE behavior, and all GLM residual pathways. The first success criterion is parity with GLM itself, **before** inserting any non-neutral Mamba output.
2. **Full Mamba-3 MIMO, not merely a label.** The reference Mamba donor has 24 blocks, each with a Mamba-3 MIMO recurrent mixer **and** a gated MLP. Keep its MIMO recurrent equations and state layout (MIMO rank 4, state size 128) and its gate/MLP computations. Do not confuse MIMO rank 4 with every-fourth-layer MLP placement. The reference donor used the Llama 3.1 tokenizer for its own training; **it did not use Llama Transformer weights**. That tokenizer is *not* the GLM-first student's tokenizer.
3. **Proposed safe integration:** first use an unchanged GLM block plus a zero-gated Mamba-3 MIMO residual branch, with input/output adapters 6,144 <-> 2,048 if reusing the official 2,048-wide Mamba-3 pretrained block. At zero gate, each branch contributes exactly zero and the baseline GLM function remains unaffected (subject to a reference forward-parity test). This introduces no claimed GLM-to-SSM weight transplantation. The adapters and contribution must then be trained/distilled. Do not replace original GLM attention/MLP until tests support replacing it.
4. **No hidden compression.** A 6,144 -> 4,096 projection and arbitrary expert/attention compression cannot be described as exact preservation. Any loss-making change is an explicit experimental variant with measured quality regression and its own checkpoint identifier.
5. **No premature GPU spend.** Metadata and fixture tests are CPU-only. Actual teacher forward/training/full inference require separately approved compute and capacity planning. This specification itself does not authorize paid jobs, Baseten calls, HF Jobs, or downloads of hundreds of GB of weights.
6. **Fail closed.** Missing source provenance, mismatched vocabulary, unrecognized tensor layout, absent parity evidence, corrupted shards, or incoherent generation must block deployment and inference-ready claims.

## B. Authoritative source contracts (verify and pin immutable revisions)

| Property | Official GLM-5.3 BF16 | Official Mamba-3 MIMO 1.5B |
|---|---|---|
| Source | zai-org/GLM-5.3-BF16 | state-spaces/mamba3-mimo-1.5b |
| Tokenizer owner | GLM-5.3 tokenizer | Llama 3.1 tokenizer (donor training only) |
| Vocabulary size | 154,880 | 128,256 |
| Model hidden width | 6,144 | 2,048 |
| Backbone | 78 GLM Transformer blocks | 24 blocks; Mamba mixer plus gated MLP in every block |
| Attention | 64 heads, QK 192 nonrotary + 64 rotary, V width 256, MLA/DSA | No Transformer attention in the standalone donor |
| MoE | First 3 blocks dense, remaining 75 sparse; 256 routed + 1 shared expert, top-8 | Gated MLP per block, donor intermediate width 3,824 |
| Other geometry | Dense intermediate 12,288; sparse intermediate 2,048; Q latent 2,048; KV latent 512 | MIMO rank 4; state size 128; expansion 2; head width 64; chunk size 16 |
| GLM norm/positional | RMSNorm epsilon 1e-5, RoPE theta 8,000,000, interleaved rotary | Verify RMSNorm/residual-in-FP32, RoPE fraction 0.5 and all MIMO head/state layouts from pinned source |
| Output projection | Untied, 154,880 outputs | Tied, 128,256 outputs in standalone donor |

Read official configs and code instead of treating this table as a substitute for immutable source references. Also verify GLM tokenizer EOS IDs (154820, 154827, 154829) and chat-template behavior. The config context-length field is **not** evidence that the integrated runtime supports that length.

References:
- https://huggingface.co/zai-org/GLM-5.3-BF16/blob/main/config.json
- https://huggingface.co/zai-org/GLM-5.3-BF16/blob/main/tokenizer_config.json
- https://huggingface.co/state-spaces/mamba3-mimo-1.5b/blob/main/config.json
- https://huggingface.co/state-spaces/mamba3-mimo-1.5b
- https://arxiv.org/pdf/2603.15569
- https://github.com/jxiw/MambaInLlama (related distillation methodology, **not** proof for GLM-5.3+Mamba-3)

## C. Requirements with auditable IDs

| ID | Requirement | Acceptance evidence |
|---|---|---|
| TOK-01 | GLM tokenizer belongs with GLM input embeddings and LM head; full token-to-ID mapping and vocabulary sizes are consistent. | Entire vocabulary map hash, 154,880 embedding/head rows and sampled encode/decode token-ID tests |
| TOK-02 | GLM BOS/EOS/PAD/control-token IDs and chat template are consistent with official GLM runtime. | Explicit multi-turn and stop-token tests; test IDs 154820, 154827, 154829 |
| GEO-01 | Preserve original GLM tensor dimensions for all unchanged layers. | Shape inventory of embeddings, 78 layers, MLA/DSA, dense/256+shared experts, output head |
| GEO-02 | Preserve full Mamba-3 donor block structure: 24 mixer + gated-MLP blocks, MIMO rank 4, recurrent state 128, donor layer order. | Independent reference layout and per-block parameter-state correspondence |
| GEO-03 | Any adapter has an explicit source/target basis, exact shape and rank; no unsupported invertibility claims. | Adapter shapes, singular/rank checks and effect-specific tests |
| SEM-01 | Preserve GLM RMSNorm placement/weights/epsilon, SiLU/SwiGLU, attention score/scaling, positional rotations and MLA/DSA indexing. | Matched module outputs and intermediate attention-score fixtures |
| SEM-02 | Preserve original GLM sigmoid/correction-bias router, top-8 expert IDs/weights, routed scaling, shared-expert contribution and FP32 routing conventions. | Routing parity fixtures, including ties and correction-bias edge cases |
| SEM-03 | Preserve Mamba-3 MIMO projection, recurrent update, state initialization, chunking and gating semantics, including its gated MLP. | Donor-reference full-sequence, chunked, streaming step equivalence on deterministic inputs |
| MAP-01 | Record exact source revision, source tensor name, recipient tensor name, shape, dtype, layout/permutation and hash for every copied or transformed tensor. | Complete machine-readable manifest and coverage report; unmapped or duplicate keys fail |
| MAP-02 | Never silently replace donor normalization with unit weights or change activations/routing behind a successful shape check. | Explicit semantic-diff checks and operation-specific tests |
| CON-01 | Unchanged GLM teacher graph and residual ordering must remain equivalent at the initial zero-gate baseline. | Matched hidden states and logits before/after inserting zero-gated branches |
| CON-02 | Mamba branches consume/return GLM residual tensors through explicit adapters; they do not rewrite GLM token embeddings or GLM attention into Mamba state weights. | Verified isolated branch interfaces and gradient/training isolation |
| TRAIN-01 | Learned replacements are trained/distilled against GLM teacher traces in **GLM token space**. | Pinned data, alignment manifests, objective definitions, reproducible training runs |
| TRAIN-02 | Architecture replacements are incremental, reversible and ablated; all prior baseline tests remain green. | Per-stage before/after comparison, rollback checkpoint, regression report |
| CKPT-01 | New artifacts are immutable/versioned and contain tokenizer, config, tensor index, all shards, model/code revisions, transform/training provenance. | Full hash verification, load-contract checks and complete shard inventory |
| RUN-01 | Stateful prompt ingestion and decoding use correct GLM KV-cache plus any Mamba recurrence/cache, with equivalent full/step/chunk results. | Differential decoding and multi-turn control-token tests |
| RUN-02 | No checkpoint is called inference-ready until load, numerical, stability, coherent generation, task-quality and reproducibility gates pass. | Signed report of all required gates and their measurable evidence |
| COST-01 | No paid compute starts automatically; data staging and GPU plan require explicit authorization. | Separate reviewed command/estimated hardware, runtime and spend ceiling |

**Transfer classification:** every mapping must be marked **EXACT** (literal unchanged tensor and operation), **EQUIVALENT** (mathematically justified reparameterization plus parity test), or **LEARNED/APPROXIMATE** (requires training and measured acceptance). Unknown mappings are **BLOCKED**. A numerically similar approximation may be usable, but it must never be reported as exact.

## D. Components and proposed module ownership

Use existing IQ modules where semantics match; avoid creating duplicate model runtimes. The following names are *proposed files*, not a claim that they already exist:

| Component / proposed boundary | Responsibility | Key dependencies |
|---|---|---|
| GLM reference loader and trace recorder | Load immutable GLM teacher, record per-layer states/logits/router/attention outputs | Official tokenizer/config/model code, pinned revisions |
| GLM-native student backbone | Same lexical tensors, original 78-layer ordering, residual graph, MLA/DSA and original MoE | Teacher parity before adding Mamba |
| Mamba-3 MIMO reference compatibility layer | Entire donor mixer + gated MLP, state init/step/chunk semantics, donor weight mapping | Pinned Mamba source; existing iq_model/state/mamba3.py |
| Hybrid insertion scheduler | Placement of up to 24 complete Mamba blocks in monotonic donor order without deleting original GLM blocks | Frozen schedule manifest; zero-gate behavior |
| Residual adapters and gates | Map GLM 6,144 to Mamba 2,048 and back; start with exact zero output contribution | Safe norms, trainable adapters, gated skip connections |
| Transfer ledger / tensor inspector | List all source and target tensors, dtype, shape, permutations, exact/cast/transformed provenance | Safetensors headers, source and recipient immutable manifests |
| Teacher/student alignment dataset builder | Align positions using GLM tokens; capture residual/attention/router/logit targets | GLM teacher, reproducible calibration text, privacy/licensing |
| Distillation trainer | Teacher KL on GLM logits, optional hidden-state/attention/router alignment and next-token loss | Stable recurrent backprop, controlled freezing/unfreezing |
| Incremental replacement controller | Select one replaceable attention path at a time after quality preservation | Per-stage gates, rollback snapshots |
| Checkpoint compiler & loader | Stream/validate shards; support offload/distributed loading; produce immutable artifacts | Correct GLM tokenizer, full source weights and manifest |
| Differential test harness | Compare teacher vs zero-gate student, module parity, state caches and generated output | Fixed seeds, deterministic fixtures, reference implementations |
| Serving/observability adapter | Readiness gates, streaming/token controls, traceable errors, cancellation and budget limits | Validated checkpoint, actual GPU-memory plan |

Proposed directory layout (adjust to repo conventions): model/glm-native, model/mamba3-reference, model/hybrid-gates, transfer/provenance, training/glm-distillation, tests/parity, scripts/checkpoint-preflight and docs. Do not rename or repurpose current canonical all256 compiler silently.

## E. Dependency-ordered subtasks and deliverables

| Phase | Subtask | Deliverable | Exit gate | Resources |
|---|---|---|---|---|
| P0 | Inventory & provenance of current IQ, GLM donor, Mamba donor, tokenizer artifacts and all256 bucket | Verified immutable source manifest, discrepancy report, storage/memory estimate | Every donor role and missing artifact accounted for | Metadata CPU, no inference |
| P1 | Freeze the target specification and acceptance matrix | GLM-first model topology and unambiguous tensor contracts; insertion schedule | Architecture accepted with no undocumented dimensional compression | Docs/CPU |
| P2 | Establish an authoritative GLM teacher and correct tokenizer | Deterministic teacher-only parity harness and reference fixtures | Correct GLM tokenizer/embedding/head, model forward and cached decoding validated | Mostly CPU fixtures; original teacher requires planned distributed compute |
| P3 | Independently validate the full Mamba-3 MIMO reference | 24-block weight inventory; MIMO rank/state/MLP and full-vs-step tests | Donor reference parity under pinned runtime; no missing MLPS | CPU small fixtures; optional GPU kernel validation |
| P4 | Build function-neutral hybrid baseline | GLM graph plus 6,144↔2,048 Mamba adapter/gate; 1-block pilot, then full scheduled 24 blocks | Zero-gated student matches GLM teacher hidden states/logits and stop behavior | Small fixtures first, full compare needs planned GPUs |
| P5 | Add explicit weight mapping / provenance | Bijective mapping for reused GLM tensors, donor Mamba tensors, adapters and any transformations | No unknown, duplicate, truncated or wrongly transposed tensor | CPU metadata, sampled tensors |
| P6 | Calibrate/train Mamba contributions | Pinned dataset and teacher targets; train adapters/gates or appropriate modules | Held-out perplexity/task-quality and per-layer error within predeclared thresholds | Training GPUs only after approval |
| P7 | Optional measured attention replacement | Reversible, layerwise reduction of original GLM attention only when proven acceptable | Every replacement passes parity/quality regression; no hidden capacity loss | Expensive; separate go/no-go |
| P8 | Compile **new** immutable checkpoint and serve | Full shard hashes, tokenizer package, verified stateful loader, distributed placement plan | All sanity, decode, coherence, quality and lifecycle tests pass | Distributed inference resources with explicit budget |
| P9 | Rollout / performance optimization | Benchmark throughput, memory, latency and multi-turn reliability | Correctness maintained and metrics reported | Approved provider/runtime |

**Critical dependency:** P2 (working GLM teacher) and P3 (verified Mamba reference) must succeed before claiming P4 full correctness. **No shortcut from P0 metadata to P8 inference readiness.**

## F. Test layers and gates

**Gate 0 — offline metadata (required before any weight load):**
- Exact tokenizer/model provenance; IDs and special token contract; expected lexical row count.
- Complete tensor-name coverage, shard inventory, hashes, dtype, shapes and ordering metadata.
- Verify 3 dense GLM blocks followed by 75 routed MoE blocks; all 256 expert indices, shared expert, routing biases, and 78 original block identities.
- Verify Mamba 24 complete blocks, including 24 gated MLPs, 24 mixers, rank-4/state-128 tensors.
- Reject current all256 as a GLM-founded artifact; maintain legacy checkpoint as read-only.

**Gate 1 — CPU deterministic semantic fixtures:**
- RMSNorm with real learned gamma, SiLU/SwiGLU, shared/routed expert aggregation, routing normalization/bias/scaling, attention masks/causality/score scaling, MLA/DSA projection and RoPE permutations.
- Mamba mixer recurrent reference: initial states, per-step updates, rank-4 projections, chunk boundaries, variable lengths and residual/MLP ordering.
- Negative tests: wrong vocab, wrong eos, wrong head permutation, swapped Q/K, wrong expert ids, wrong activation, random norm replacement, truncated safetensors, missing MLP, duplicate source mapping, non-zero initial residual gate.

**Gate 2 — original GLM teacher & zero-gated hybrid numerical parity:**
- On deterministic prompts, compare per-layer hidden inputs/outputs, attention intermediate scores, expert IDs/weights, final logits and stop conditions in identical token space and under controlled numerical precision.
- Zero-gated hybrid must preserve the original GLM response. Prefer exact agreement when operations are identical; otherwise record justified BF16 tolerance, max/mean error and reproducibility. Do not invent success thresholds post hoc.
- If an optimized reference uses different fused kernels, compare against a common unoptimized reference with tolerance justified by measured numerical drift.

**Gate 3 — trained hybrid regression:**
- Predeclare held-out tasks: next-token perplexity / log-loss, cross-entropy KL, reasoning, code, factual responses, long contexts, tool/call format, multilingual text, refusals and safety behavior.
- Track baseline and variant on the same input set; compare multi-turn generation and special-token stopping. No universal capability-preservation guarantee can be inferred from one smoke prompt.
- Train/test split and artifacts must be versioned; leakage checks required.

**Gate 4 — production decoder/inference:**
- Run fixed-prompt full-forward, autoregressive step, and chunked prefill/generate comparisons.
- Validate GLM attention caches, Mamba recurrent state caches, reset/continuation/sequence boundaries, streaming, batching, cancellation, resource cleanup, and finite-logit checks.
- Confirm the deployment target can fit all required resident/offloaded weights plus KV/SSM states and overhead. A single H200 cannot be presumed sufficient for GLM-5.3 or a full GLM-founded hybrid.
- Deployment readiness is blocked until coherent output and agreed quality criteria pass. HTTP 200 or a loadable checkpoint is not proof.

## G. Explicit risks / assumptions

- **Full GLM-native baseline is substantially larger than the existing ~539 GB IQ checkpoint.** Keeping all original GLM weights and adding Mamba cannot also promise the same checkpoint size; compression/attention replacement is an experimental quality trade-off.
- **GLM teacher access is compute-heavy.** First plan a legal, affordable GPU/offload path and exact estimated cost. CPU metadata inspection and low-dimensional synthetic fixtures are not equivalent to running a full 753B-class teacher.
- **Mamba pretrained 2,048-wide states are not automatically aligned with GLM 6,144-wide activations.** The explicit adapters, zero-gate baseline and training are there precisely because architecture alone does not establish semantic alignment.
- **A structurally complete hybrid is not necessarily an efficient hybrid.** Initially retaining GLM attention and adding 24 Mamba blocks increases compute. Remove/replace expensive GLM operations only after proven quality retention.
- **No head/activation/norm/projection or expert shape should be altered silently.** If a design requires such changes, separate it into a clearly labeled approximate variant with its own measurements.
- **Tokenizer artifacts are not supplied by the standalone Mamba donor repo.** Never pull tokenizer files from that repository or interpret Llama 3.1 vocabulary as evidence of Llama neural weights.
- **Prior mocha gibberish does not by itself establish all256's generation quality.** All256 remains unvalidated until it has its own results.

## H. Phase-0 approval checklist (planning-only completion)

- [x] Confirmed official GLM and Mamba model geometry from Hub metadata.
- [x] Confirmed current all256 provenance and non-readiness marker.
- [x] Defined GLM-first foundation, full Mamba MIMO block topology and zero-gate invariant.
- [x] Specified transfer classifications and four invariant categories.
- [x] Defined component boundaries, dependencies, tasks, and explicit test gates.
- [ ] Inventory complete GLM and all256 source tensor headers/hashes and any missing checkpoint files.
- [ ] Obtain pinned exact source revisions and reproducibility manifests for all required artifacts.
- [ ] Select feasible teacher runtime, hardware capacity and explicit budget (no automatic launch).
- [ ] Establish quantitative parity/quality thresholds **before** expensive training.
- [ ] Implement and validate the new GLM-first graph and actual checkpoint compiler.
- [ ] Prove trained hybrid quality and inference readiness.

**End of requirements document.** Implementation is deliberately deferred until the specified prerequisites and no-spend constraints are satisfied.
