# IQ v2 — Production Implementation Plan

Status: canonical implementation plan for the trainable IQ architecture.

This document replaces the old Gemma/NIF training assumptions. `SHADOW_TRANSFER.md` remains the source of truth for donor-independent transport math, and `IQ_WEIGHT_TRANSFER_IMPLEMENTATION_PLAN.md` defines the production multi-donor transfer sequence. This plan defines the recipient architecture, training runtime, data path, distributed execution, validation, and migration criteria.

## 0. Non-negotiable engineering rules

1. No mock tensor paths, string-returning model layers, constant energies, fake checkpoint loaders, placeholder credentials, fake cloud resources, or TODO-backed production paths.
2. Every model component must implement a real forward pass, backward-compatible autograd path, shape validation, serialization, and unit tests before being used in training.
3. Every custom optimized kernel must have a numerically equivalent PyTorch reference implementation. The optimized implementation is enabled only after forward/backward parity tests pass.
4. The canonical training implementation is PyTorch + NVIDIA Megatron-Core + Transformer Engine. Mojo is used for targeted custom kernels and MAX inference after parity is established.
5. Existing donor transport in `iq_transfer/` is reused. New donors implement `DonorInspector`; do not add model-specific graft architectures.
6. Finished work is committed, merged, and continued from `main`; do not leave competing architecture branches or duplicate source-of-truth files.
7. Model configuration, tokenizer identity, donor identity, dataset manifest, git SHA, optimizer state, RNG state, and data cursor are checkpointed together.
8. No secrets in source or model prompts. Runtime credentials come from the execution environment / workload identity / secret manager.

## 1. Runtime split

### Training

- PyTorch
- Megatron-Core for tensor, pipeline, context, expert, and data parallelism
- Transformer Engine for fused Transformer primitives and FP8/MXFP8 where numerically validated
- distributed checkpointing with atomic manifests
- `torch.compile` only after eager correctness tests pass

### Kernel optimization

- PyTorch reference implementation first
- Triton/Transformer Engine where an implementation exists
- Mojo `max.torch.CustomOpLibrary` only for IQ-specific kernels after parity and profiling demonstrate value

### Inference

- initial: PyTorch inference using the exact training modules
- production: MAX graph / MAX Serve with Mojo custom ops for IQ-specific kernels
- `iq_harness.ModelBackend` adapter exposes generation to the skills/agent runtime

## 2. Initial recipient scale

Keep the first IQ recipient close to the current repository's intended scale so architecture effects are not confounded with parameter scaling.

```yaml
hidden_size: 4096
num_blocks: 32
num_attention_heads: 32
num_kv_heads: 8
head_dim: 128
norm: rmsnorm
activation: swiglu
initial_context: 32768
max_context_target: 131072
exec_latent_dim: 512
exec_ring_groups: 8
max_inner_steps: 4
max_outer_steps: 4
min_inner_steps: 1
min_outer_steps: 1
mamba3_state_size: 128
mamba3_head_dim: 64
mamba3_mimo_rank: 4
global_attention_period: 4
hybrid_fusion: parallel_gated_residual
executive_geometry: euclidean
```

These are implementation defaults, not immutable research constants. Changes require a config version bump and a reproducible ablation.

## 3. Package layout

```text
iq_model/
  config.py
  model.py
  outputs.py
  embeddings.py
  position.py
  norm.py
  attention/
    nsa.py
    differential.py
    global_attention.py
    global_memory.py
    kernels.py
  state/
    mamba3.py
    hybrid_fusion.py
  mlp/
    swiglu.py
    moe.py
    router.py
  recurrence/
    inner_loop.py
    outer_controller.py
    halting.py
    spectral_depth.py
  latent/
    coconut.py
    soft_thinking.py
    concept_mapper.py
  energy/
    scalar_energy.py
    hamiltonian_ring.py
    diagnostics.py
  geometry/
    euclidean.py
    hyperbolic.py
  adapters/
    dora.py
  objectives/
    language.py
    mtp.py
    transfer.py
    energy.py
    halting.py
    moe.py

iq_training/
  train.py
  config.py
  optimizer.py
  scheduler.py
  precision.py
  parallel.py
  checkpoint.py
  state.py
  callbacks.py
  metrics.py

iq_data/
  manifest.py
  tokenizer.py
  packing.py
  fim.py
  contamination.py
  loader.py

iq_eval/
  language.py
  code.py
  reasoning.py
  long_context.py
  transfer.py
  recurrence.py

configs/
  model/
  training/
  experiments/
```

`iq_transfer/` remains separate and feeds transported parameters/activation targets into `iq_model` and `iq_training`.

## 4. Input, tokenizer, and embeddings

### Stage-1 tokenizer

Use the Phi proof donor tokenizer for the first Phi->IQ experiment so embedding/LM-head transport and calibration-token alignment are well-defined. The tokenizer is frozen for the complete Phi proof.

Required tokenizer metadata in every checkpoint:

- tokenizer files/hash
- vocabulary size
- BOS/EOS/PAD ids
- FIM control-token ids
- chat/control token ids if used

If FIM tokens do not exist, add a fixed reserved set exactly once, resize embedding/unembedding matrices, initialize only the new rows, and persist the resulting tokenizer as an IQ tokenizer version. Never silently mutate vocabulary after training begins.

### Embedding layer

- trainable token embedding matrix
- optional mode/type embedding for code/FIM/tool/control modes
- no positional information baked into token embeddings
- input embedding and LM head may be untied; tying is an explicit config option

Transport policy:

- exact vocabulary match: exact/coordinate transport
- partial vocabulary match: string/token-id alignment for shared tokens, learned lexical projection for unmatched rows
- different donor tokenizer: no raw row-copy assumption; use functional/lexical alignment

## 5. Position and computation-depth encoding

Use two independent coordinates.

### Token position

`xPos/RoPE-family` positional transform is applied only to Q/K. The implementation exposes a single positional interface so RoPE/xPos/YaRN can be selected without changing attention code.

### Reasoning depth

Outer and inner recurrence receive a continuous spectral depth encoding independent of token position:

```text
s(r) = [sin(w_1 r), cos(w_1 r), ..., sin(w_k r), cos(w_k r)]
```

Optional state conditioning is bounded:

```text
w_i(z) = w_i0 * (1 + alpha * tanh(a_i^T z)), alpha <= configured maximum
```

The spectral encoding is projected into the executive latent and/or residual stream through learned gated projections. Token position and reasoning depth are never conflated.

## 6. Core block and hybrid execution

IQ v2 is explicitly a **Mamba-3 + Transformer hybrid**. Mamba-3 is never treated as a replacement for attention. The two paths solve different memory problems:

- Mamba-3: linear-time recurrent state propagation, long-horizon state tracking, and constant-size decode state.
- Transformer attention: exact content-addressable retrieval, induction, code-symbol lookup, and needle-in-context recall.
- NSA: efficient local/selected/compressed token addressing on ordinary inner blocks.
- periodic global attention: explicit full-context anchor blocks for high-recall retrieval.
- Differential Attention: outer-loop controller attention over compressed/global memory, not a replacement for token-level attention.

The default topology is parallel fusion within each inner block plus a periodic global Transformer anchor:

```text
x
 -> RMSNorm
 -> +----------------------+-------------------------+
    | Mamba-3 state path   | Transformer path        |
    | complex/MIMO SSM     | NSA or global attention |
    +----------------------+-------------------------+
                    |
             bounded residual fusion
                    |
                 residual
                    |
                RMSNorm
                    |
              dense SwiGLU
                    |
        executive-latent injection
```

Every fourth block is a global-attention anchor by default; the three intervening blocks use NSA. The ratio is configurable and must be evaluated on code dependency retrieval, multi-needle recall, long-context perplexity, KV-memory use, recurrent-state memory, and tokens/sec.

Use independent bounded residual gates rather than a zero-sum mixer:

```text
m = Mamba3(norm(x))
a = Attention(norm(x))
y = x + alpha_m * m + alpha_a * a
alpha_m = alpha_m_max * sigmoid(g_m)
alpha_a = alpha_a_max * sigmoid(g_a)
```

For the first Phi transport stage, `alpha_a` initializes to preserve the transported attention path and `alpha_m` initializes near zero. When a compatible Mamba-3 donor is available, the Mamba path can be donor-initialized and its gate can start at a nonzero calibrated value.

### Mamba-3 recurrent state path

Use the published Mamba-3 block semantics rather than a hand-written "linear projection" approximation:

- expressive SSM discretization
- complex-valued state update
- MIMO mode where supported
- recurrent state carried across decode tokens
- BF16 reference implementation first
- state parameters kept at the precision required for stable recurrence
- exact state reset/continuation semantics covered by tests
- chunked training must produce the same state transition as the reference unchunked path within tolerance

The production implementation may wrap the upstream Mamba-3 kernel initially; custom Triton/Mojo kernels are permitted only after forward/backward/state parity.

### Transformer token-retrieval path

Ordinary inner blocks use Native Sparse Attention. Periodic anchor blocks use explicit global Transformer attention. Both share the same positional interface and causal semantics. The global path is not removed even if Mamba long-context metrics improve; it is the model's exact addressable-memory mechanism.

### Native Sparse Attention

Implement three real paths:

1. local causal blocks/window
2. compressed/global summary blocks
3. selected high-value blocks/tokens

Required behavior:

- causal masking is exact
- selection indices are deterministic under fixed seeds
- no dense `N x N` fallback in the production sparse path
- reference dense-equivalent tests on small sequences
- attention statistics expose local/compressed/selected utilization
- optimized kernel must match the reference within dtype-specific tolerances

Q/K/V/O transported from the donor initialize the compatible projections. NSA-specific compression/selection parameters are new trainable parameters.

### Outer/global execution path

Use Differential Attention over compressed token memory + executive state, not every raw token at every inner step.

Two attention maps are computed and combined using the differential parameterization. Stream 1 receives transported donor initialization. Stream 2 starts from the same transported basis plus a small reproducible trainable symmetry-breaking delta; differential-specific lambda parameters are initialized from the published formulation.

This module runs only in the outer refinement loop.

## 7. Feed-forward and routing

### Inner FFN

Dense SwiGLU:

```text
SwiGLU(x) = W_down( SiLU(W_gate x) * (W_up x) )
```

Transport gate/up/down independently through `iq_transfer`.

### Outer FFN

Phase 1 uses a dense SwiGLU outer FFN to establish the Phi proof without confounding MoE routing.

Phase 2 enables routed SwiGLU experts for the code-specialized/multi-donor stage. The implementation must support:

- top-k expert routing
- shared expert option
- expert capacity accounting
- token drop = disabled by default; overflow behavior explicit
- load-balancing auxiliary loss
- router z-loss
- expert parallelism through Megatron-Core
- grouped GEMM path where supported

Mellum/code-MoE donors can later initialize experts/router through exact/operator/functional transport depending on topology compatibility.

## 8. Recurrent control flow

IQ uses nested adaptive recurrence with shared weights.

### Inner loop

Purpose: local refinement and code/token computation.

- reuse the same block parameters across inner steps
- inject inner-step spectral encoding
- inject executive latent through a gated projection
- produce compressed memory for the outer controller

### Outer loop

Purpose: update the global execution hypothesis.

At each outer step:

1. run required inner refinement
2. compress token state
3. run Differential Attention against executive/global memory
4. update executive latent with the Hamiltonian controller
5. compute energy, convergence, and halting statistics
6. continue or halt

### Training-time halting

Training must remain differentiable and distributed/compile friendly.

- execute up to configured max steps
- compute halting probabilities per step
- use soft expected-state/ACT-style weighting during training
- after a sequence is effectively halted, later steps are masked/no-op for loss/state aggregation
- enforce minimum steps
- add ponder/compute loss to discourage unnecessary iterations

### Inference-time halting

Inference may physically exit early when all enabled criteria pass:

```text
p_halt >= threshold
relative_state_delta <= eps_state
absolute_energy_delta <= eps_energy
step >= min_steps
```

Record actual inner/outer step counts in telemetry.

## 9. Continuous latent reasoning

### Coconut-style latent recurrence

The model may recycle a continuous hidden/concept state without decoding a discrete token. This path is enabled only in reasoning phases and has an explicit maximum latent-thought budget.

### Soft-thinking concepts

A soft concept is a probability-weighted embedding mixture, preferably over a bounded top-k vocabulary set for efficiency:

```text
p = softmax(logits / temperature)
c = sum_{v in topk(p)} p_v * E_v
```

The top-k approximation must be compared against the dense reference on small vocabularies before production use.

### Experimental isolation

Coconut, soft-thinking, and IQ executive-latent reasoning are independently feature-flagged for ablation. No training run may silently enable all three without an experiment configuration that names the combination.

## 10. Hamiltonian executive controller

Replace `IsingGate` entirely. No fake scalar energy and no spin-copy operation remains. The useful legacy NIF idea is retained as a real differentiable energy-based executive controller, not as a quantum-spin simulation.

The executive state is intentionally small (`d_exec=512`) so energy-gradient dynamics are computationally tractable.

Define a scalar energy:

```text
E_theta(z, c) -> scalar
```

where `c` is compressed reasoning context.

Port-Hamiltonian update:

```text
g = grad_z E_theta(z, c)
dz = (J_theta - R_theta) g + B_theta u
z_next = z + dt * dz
```

Structural constraints:

```text
J = A - A^T
R = L L^T + eps I
```

Ring structure:

- split `z` into 8 groups
- primary learned interactions are self + nearest-neighbor + wraparound blocks
- optional bounded low-rank long-range correction
- masks enforce the ring topology at parameterization time

Because backpropagation through `grad_z E` creates higher-order derivatives, keep the energy network small and isolated. Benchmark memory/throughput with `torch.func.grad`/autograd and activation checkpoint this controller if needed.

### Energy must score generated reasoning

The energy model receives the actual recipient trajectory/state/output context. It must never score only the unchanged problem prompt.

Contrastive/ranking training uses successful vs failed/corrupted trajectories from the same task. Example objective:

```text
L_energy = softplus(E_positive - E_negative + margin)
```

Energy is also logged as a calibration signal for halting; halting does not rely on energy alone.

### Optional Riemannian executive geometry

Riemannian/hyperbolic geometry is retained only as an explicit executive/concept-space experiment. It is not applied to the transferred token backbone by default.

Baseline:

```text
z_exec in R^d
```

Experimental variant:

```text
z_exec in H^d
```

The hyperbolic implementation must provide numerically stable exp/log maps, distance, projection/retraction, mixed-precision guards, and Euclidean-vs-hyperbolic ablation parity. No curvature-dependent hand-designed attention factor is permitted.

## 11. Concept collapse / lexicalization head

Reasoning and language spaces are explicitly separated.

```text
[final token state, executive state, optional soft concept]
 -> ConceptMapper
 -> RMSNorm
 -> LM head
 -> vocabulary logits
```

The concept mapper is a real trainable gated residual MLP/projection. Initialize it near identity with the executive branch initially gated near zero so transported donor logits remain stable at step 0.

Ablation:

- direct LM head
- concept mapper

The mapper is retained only if it improves held-out reasoning/code quality without harming calibration or ordinary LM loss.

## 12. DoRA correction around transported weights

Implement a tensor-parallel-aware `DoRALinear` compatible with the selected parallel linear layer.

Transported donor weight is the frozen/slow base during early alignment. Trainable DoRA parameters correct architecture mismatch.

Required features:

- magnitude/direction decomposition
- configurable rank
- merge/unmerge for export
- TP-aware sharding rules
- no materialization of full TP weight during normal training
- state-dict conversion tests
- exact merge parity test

Stage-wise unfreezing removes the requirement that the transported base remain frozen forever.

## 13. Transfer integration

Reuse `iq_transfer`.

### Stage A — calibration

- fixed tokenizer/version
- fixed calibration corpus
- donor activations captured at selected layers/modules
- IQ random/structural recipient activations captured at corresponding points
- functional shadows generated with fixed measurement seeds

### Stage B — correspondence

- monotonic layer matching
- separate coordinate maps for residual, Q/K/V/O, gate/up/down spaces where required
- persist correspondence and map quality metrics

### Stage C — operator initialization

Transport:

- Q
- K
- V
- O
- gate
- up
- down
- compatible norm/embedding/LM-head state

Never directly transplant donor weights into the Hamiltonian controller.

### Stage D — correction

Initially freeze transported base and train:

- DoRA corrections
- NSA-specific parameters
- executive latent injection
- Differential Attention second stream/lambda
- Hamiltonian controller
- concept mapper
- halting controller

Then progressively unfreeze donor-derived blocks based on validation gates.

## 14. Training objectives

Losses are phase-specific; do not enable every loss from step 1.

### Base language/code

- next-token cross entropy
- FIM formatting/data objective
- multi-token prediction heads

### Transfer alignment

- functional-shadow observable loss
- paired representation/subspace loss
- optional operator/Jacobian diagnostic loss
- donor logit KL only where token spaces align

### Recurrence

- final task loss
- intermediate consistency loss where useful
- ACT/ponder compute regularization
- state-convergence diagnostics

### Energy

- positive/negative trajectory ranking
- energy calibration metrics
- conservative/dissipative diagnostics

### MoE

- load-balance loss
- router z-loss

No loss coefficient is hardcoded in model code. Every coefficient is explicit in a versioned training config and logged.

## 15. Optimizer

Delete/retire the current parameter-Gram-Schmidt `MuonOptimizer` after the new optimizer path is validated.

Production optimizer groups:

- Muon: eligible 2D matrix weights where the implementation and distributed sharding are valid
- AdamW: embeddings, norm scales, scalar/vector parameters, router scalars, halting parameters, energy scalars, Mamba recurrence parameters that require non-Muon handling, and parameters not supported by Muon
- GaLore: optional memory-constrained training mode only; never stack it implicitly on parameters already assigned to Muon. Its activation is explicit in the experiment config and must show a measured HBM benefit without unacceptable convergence regression.

Muon performs momentum + gradient/update orthogonalization; it does not periodically orthonormalize model parameters.

Required tests:

- optimizer reference parity on small matrices
- distributed shard parity
- checkpoint/resume parity
- no parameter-group overlap or omission

## 16. Precision

Correctness order:

1. BF16 reference training
2. FP8 on Hopper where TE supports the operation and loss curves match BF16 acceptance bands
3. MXFP8 on Blackwell only after dedicated numerical validation

Keep numerically sensitive operations in BF16/FP32 as needed:

- normalization statistics
- softmax/logsumexp reductions
- routing probabilities
- Hamiltonian energy accumulation
- selected optimizer state

No low-precision mode becomes default solely because it runs faster.

### Quantum/CUDA-Q isolation

CUDA-Q and external QPUs are not dependencies of the production forward, backward, optimizer, checkpoint, or inference path. They remain an isolated research backend for future experiments. No training launch may require QPU availability, and no quantum claim is used as evidence for model reasoning quality.

## 17. Distributed training

Use Megatron-Core composable parallelism.

### First 4096x32 recipient on 8x H200

Start with the simplest configuration that fits:

- DP/FSDP or distributed optimizer
- CP=2 for 32K context if activation memory requires it
- TP only if profiling shows a memory/throughput win
- PP=1 initially
- EP=1 until outer MoE stage

### Scale-out

Enable as needed:

- TP for wide linear layers
- CP for long context
- PP for deeper/larger recipients
- EP for MoE experts
- sequence parallelism when required by TP/EP

Every supported distributed topology receives a checkpoint round-trip and short loss-parity test.

## 18. Checkpoint format and reproducibility

Every checkpoint is an atomic directory/object prefix with a manifest containing:

```text
model shards
optimizer shards
scheduler state
training step / consumed tokens
data cursor / sampler state
CPU + CUDA RNG states
parallel topology
model config + config hash
tokenizer + tokenizer hash
donor checkpoint IDs/hashes
transport-map IDs/hashes
dataset manifest hash
git commit SHA
container/image digest
precision recipe
metrics snapshot
```

Rules:

- checkpoint completion marker written last
- incomplete checkpoint is never considered resumable
- restore verifies hashes/config compatibility
- resharing/resharding is explicit, tested, and versioned
- inference export uses safetensors + immutable model/tokenizer/config metadata

## 19. Data pipeline

No shell-created sample directories or implicit datasets.

A dataset manifest specifies every source with:

- URI/version
- license/provenance metadata
- content hash
- language/content type
- weight
- preprocessing version

Pipeline:

1. normalize text/code without destroying whitespace-sensitive languages
2. exact dedup
3. near-dedup
4. secret/credential filtering
5. generated/minified/vendor/binary filtering policy
6. benchmark contamination scanning
7. tokenize with frozen tokenizer
8. deterministic document packing
9. deterministic FIM transform using recorded RNG seeds
10. write immutable indexed shards

Code training records repository/file boundaries and can preserve related-file metadata for later repository-context training.

## 20. Code-specific training

### FIM

Implement PSM/SPM transformations in `iq_data/fim.py`. Preserve exact special-token semantics. Unit-test reconstruction of original code from transformed examples.

### MTP

Implement configurable future-token heads (initially small N such as 4). Heads share the lexical embedding/LM-head space where appropriate. MTP is training-time auxiliary supervision; inference can discard auxiliary heads unless speculative decoding experiments use them.

### Repository tasks

Later mid-training examples include:

- function completion
- cross-file symbol resolution
- bug repair
- test repair
- API migration
- code review/fix pairs
- issue -> patch -> tests

Repository context must use real dependency/file retrieval, not random concatenation.

## 21. Post-training for reasoning/coding

Sequence:

1. supervised instruction/code-repair tuning
2. executable RLVR/verifier training
3. agent/tool-use tuning using the existing `iq_harness`

Verifiable rewards may include:

- compile/typecheck success
- unit/integration tests
- lint/static-analysis invariants
- exact task answers for math/reasoning

Never reward merely producing tool calls or verbose reasoning.

## 22. Evaluation gates

### Language

- validation CE/perplexity
- calibration / ECE where applicable

### Code

- HumanEval+/MBPP-style function tasks
- repository-level repair benchmark
- multi-file dependency tasks
- compile/test pass rate
- pass@1 primary; pass@k diagnostic

### Reasoning

- multi-step math/logical tasks
- planning/backtracking tasks
- robustness to distractors

### Long context

- retrieval at multiple depths
- code dependency recall
- long-sequence perplexity/CE

### IQ-specific

- functional-shadow error
- donor capability retention
- coordinate-map error
- energy separation AUC/margin
- inner/outer step histogram
- accuracy vs executed recurrent steps
- premature-halt rate
- NSA branch utilization
- MoE expert load entropy/capacity
- concept-mapper gain vs direct head

### Promotion gate

A stage is not promoted because training loss decreased. Require:

- no regression beyond configured tolerance on donor-retention suite
- improvement on target reasoning/code suite or compute-efficiency target
- stable long-run training without NaN/Inf/divergence
- checkpoint/resume parity
- distributed topology test pass

## 23. CI / quality gates

### Per PR

- formatting/lint/typecheck
- unit tests
- small CPU forward tests
- 1-GPU forward/backward test where GPU CI exists
- deterministic seed test
- checkpoint serialization test
- secret scan
- dependency vulnerability scan

### GPU integration

- BF16 forward/backward numerical checks
- optimized-kernel vs reference parity
- gradient parity
- 2-GPU NCCL distributed loss parity
- resume-after-interruption test
- memory leak test

### Scheduled scale tests

- 8-GPU short training
- long-context CP test
- TP/EP topology tests when enabled
- checkpoint reshard test

No CI check is disabled to make the branch green.

## 24. Observability

Record at minimum:

- tokens/sec and samples/sec
- model FLOP utilization when measurable
- GPU HBM/SM utilization
- dataloader wait
- loss by objective
- grad norm globally and per subsystem
- parameter/update norm
- learning rate by group
- NaN/Inf counters
- recurrent depth histogram
- halt probability and convergence deltas
- energy distributions for positive/negative trajectories
- NSA sparsity/selection metrics
- MoE load/capacity/router entropy
- shadow/transport retention metrics

External tracking (W&B, TensorBoard, cloud monitoring) is configured by runtime config; no usernames/webhooks are committed.

## 25. Migration / deletion plan

Do not keep parallel legacy implementations after replacement is green.

### Replace

- `nif_sovereign/core/custom_llm_architecture.mojo`
- `nif_sovereign/core/physics_transformer_block.mojo`
- `nif_sovereign/modules/ising_gate.mojo`
- legacy neutrino-oscillation blocks on the production path; their useful recurrent/oscillatory role is replaced by Mamba-3
- `nif_sovereign/optimization/muon_optimizer.mojo`
- legacy Gemma/Neutrino/Ising architecture config fields
- `gcp_training_script.sh`
- `gcp_training_config.yaml`

### Keep only if still used and mathematically justified

- reusable Riemannian/geometry experiments, behind explicit research flags
- Mojo kernels that pass parity/performance gates

Migration rule: build and validate the new path first, migrate callers/tests/config, then delete the legacy path in the same implementation series. Do not leave both as competing architectures.

## 26. Implementation sequence

### Phase 0 — repository cutover

1. create `iq_model`, `iq_training`, `iq_data`, `iq_eval`
2. create typed/versioned config system
3. add dependency lock/container build
4. add CI for CPU + GPU-capable test layers
5. add immutable experiment/config IDs

Exit: empty framework is not enough. A real minimal dense decoder must run forward/backward and checkpoint/resume.

### Phase 1 — transported dense recipient

1. embeddings/tokenizer
2. RMSNorm
3. positional interface
4. GQA-compatible attention
5. dense SwiGLU
6. LM head
7. load Phi transport output
8. implement DoRA correction
9. train CE/FIM/MTP

Exit: Phi->IQ transported model trains and evaluates end-to-end without recurrent/Hamiltonian features.

### Phase 2 — Mamba-3 + Transformer hybrid execution

1. integrate the reference Mamba-3 state path with exact recurrent-state semantics
2. implement parallel bounded residual fusion
3. NSA reference implementation for ordinary Transformer branches
4. periodic global-attention anchor blocks
5. optimized sparse/state kernels only after reference parity
6. outer Differential Attention
7. compressed global memory
8. evaluate attention-anchor frequency (for example 8:1, 4:1, 3:1, 2:1 Mamba-heavy/sparse blocks to global anchors) on long-context coding and needle retrieval

Exit: Mamba state continuation, sparse attention, global retrieval, and fusion are numerically correct; the hybrid meets the donor-retention gate and demonstrates the intended memory/throughput tradeoff.

### Phase 3 — recurrence + halting

1. shared inner recurrence
2. executive latent injection
3. spectral reasoning-depth encoding
4. differentiable training-time halting
5. hard inference early exit

Exit: adaptive compute works, telemetry is correct, and task quality vs executed steps is measured.

### Phase 4 — Hamiltonian/energy controller

1. scalar energy network
2. structured ring J/R parameterization
3. differentiable port-Hamiltonian update
4. trajectory ranking dataset/objective
5. energy/convergence halting integration

Exit: energy separates successful/failed trajectories and improves target reasoning or compute efficiency against a conventional recurrent-controller control.

### Phase 5 — continuous thought + lexical collapse

1. Coconut latent recycling
2. soft-thinking concept path
3. concept mapper
4. independent and combined ablations

Exit: retain only mechanisms that improve coding/reasoning or reduce compute with statistically meaningful repeated runs.

### Phase 6 — MoE/code donor integration

1. outer routed SwiGLU MoE
2. EP/grouped GEMM
3. code donor inspector/functional transfer
4. expert/router transport where compatible
5. FIM/MTP code mid-training

Exit: code suite improves without unacceptable general/reasoning regression.

### Phase 7 — post-training + agent runtime

1. instruction/code repair SFT
2. executable RLVR
3. `iq_harness.ModelBackend` adapter
4. tools/skills/subagent evaluation

Exit: the model can perform tool-using coding workflows with verifiable outcomes.

### Phase 8 — scale donor and recipient

Only after the Phi proof passes `iq_transfer.scaling` gates:

- larger dense donor
- 30–70B donor
- 100B+ donor
- frontier MoE donor

First hold the IQ recipient architecture/size fixed to measure transfer scaling. Increase IQ recipient size only in a separate scaling study.

## 27. Definition of production-ready for first training launch

Training may begin only when all are true:

- real end-to-end forward/backward
- no mock model components on active path
- tokenizer and dataset manifests immutable
- Phi checkpoint can be inspected and transported
- transported state loads without missing/unexpected trainable weights except explicitly documented new IQ modules
- optimizer covers every trainable parameter exactly once
- BF16 1-GPU overfit test passes on a tiny real batch
- multi-GPU loss parity test passes
- checkpoint save/resume reproduces the next-step loss within tolerance
- NaN/Inf guards and metric logging active
- evaluation harness runs from a checkpoint
- container image is digest-pinned
- cloud job config contains no placeholder identities or credentials
- interruption/restart path tested

Only then launch the first meaningful Phi->IQ adaptation run.
