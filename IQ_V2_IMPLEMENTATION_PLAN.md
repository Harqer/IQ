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

Keep the first dense Phi recipient close to the current repository's intended scale so transfer retention is measurable. The hybrid backbone is a separate heterogeneous schedule rather than a stack of identical Transformer blocks.

```yaml
hidden_size: 4096
dense_control_layers: 32
num_attention_heads: 32
num_kv_heads: 8
head_dim: 128
norm: rmsnorm
dense_attention_position: qk_rope
dense_attention_qk_norm: per_head_rmsnorm
compressed_attention_position: trailing_partial_rope_with_inverse_output
expert_activation: swiglu
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
mamba3_chunk_size_bf16: 16
mamba3_state_rotation: native_complex_data_dependent_rope
hybrid_backbone: explicit_heterogeneous_schedule
executive_geometry: euclidean
mtp_shared_heads: 2
```

The schedule itself is versioned model configuration. No code path infers an attention cadence from a single integer such as `global_attention_period`.

These are implementation defaults, not immutable research constants. Changes require a config version bump and a reproducible ablation.

## 3. Package layout

```text
iq_model/
  architecture.py
  config.py
  model.py
  outputs.py
  embeddings.py
  position.py
  norm.py
  attention/
    context_dense.py
    masking.py
    sliding_window.py
    compressed_sparse.py
    compressed_memory.py
    sparse_indexer.py
    differential.py
    global_attention.py
    global_memory.py
    kernels.py
  state/
    mamba3.py
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
  residual/
    mhc.py
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

IQ has three distinct positional/state-rotation mechanisms plus an independent reasoning-depth coordinate. They must not be conflated.

### Dense Transformer anchors: Q/K RoPE

Dense/global Transformer context anchors use per-head Q/K RMS normalization followed by ordinary RoPE:

```text
q = RMSNorm_head(W_q x)
k = RMSNorm_head(W_k x)
q = RoPE(q, position)
k = RoPE(k, position)
```

RoPE applies to Q/K only. It is not applied to the residual stream before Mamba.

### CSA/HCA: normalized compressed KV + trailing partial RoPE

Compressed context attention follows the DeepSeek V4-family positional rule:

- normalize each query head and the shared compressed KV head before core attention;
- apply RoPE only to the configured trailing rotary dimensions;
- use the same relative-position convention for compressed entries;
- apply the inverse position rotation to the rotary slice of the attention output before grouped output projection.

The reference primitives live in `iq_model/position.py`; optimized CSA/HCA kernels must match them.

### Mamba-3: native complex/data-dependent state rotation

Mamba-3 owns its state-space rotation internally. Its complex-valued recurrence and data-dependent rotary state update remain inside the pinned upstream Mamba-3 implementation. IQ never applies Transformer Q/K RoPE to the Mamba input as a substitute for, or in addition to, Mamba-3's native state rotation.

### Reasoning depth

Outer and inner recurrence receive a continuous spectral depth encoding independent of token position and Mamba token-time state:

```text
s(r) = [sin(w_1 r), cos(w_1 r), ..., sin(w_k r), cos(w_k r)]
```

Optional state conditioning is bounded:

```text
w_i(z) = w_i0 * (1 + alpha * tanh(a_i^T z)), alpha <= configured maximum
```

Token position, Mamba token-time state, and reasoning-depth coordinates are separate state variables.

## 6. Heterogeneous hybrid backbone

IQ does not use one universal `Transformer + Mamba + FFN` block. Sequence mixing, expert computation, context attention, and executive control are distinct physical layer types arranged by an explicit `HybridSchedule`.

Canonical layer symbols:

```text
M = Mamba-3 MIMO recurrent sequence/state layer
E = routed/shared SwiGLU MoE layer
C = CSA compressed sparse context layer
H = HCA heavily compressed context layer
A = dense/global QK-normalized RoPE attention anchor
X = Hamiltonian executive update
```

A schedule is written explicitly, for example:

```text
M E
M E
M C E
M E
M H E
M E
M A E
...
X
```

This is illustrative ordering, not a hardcoded ratio. Production configs store the complete layer sequence; `iq_model.architecture.HybridSchedule` validates that Mamba-3 remains the dominant sequence mixer and that executive control terminates the backbone when present.

The division of labor is structural:

- **Mamba-3 MIMO** performs the dominant recurrent sequence processing/state evolution and later reasoning recurrence.
- **MoE/SwiGLU layers** provide nonlinear/expert capacity as separate layers; there is no automatic SwiGLU appended after every Mamba layer.
- **CSA/HCA** provide efficient explicit long-range context access.
- **Dense attention anchors** provide exact pairwise comparison, induction, strict few-shot matching, and diagnostic fallback.
- **Hamiltonian executive control** manages global reasoning state/halting; it is not another token mixer.

During the first Phi transport stage, `IQForCausalLM` remains the dense Transformer retention/control model. The hybrid model is built separately so transfer evaluation is not confounded by changing the control architecture.

### Mamba-3 recurrent state path

Use the published Mamba-3 block semantics rather than a hand-written "linear projection" approximation:

- expressive SSM discretization
- complex-valued state update
- **MIMO is mandatory in the production IQ architecture**; default rank is 4
- recurrent state carried across decode tokens
- BF16 reference implementation first
- state parameters kept at the precision required for stable recurrence
- exact state reset/continuation semantics covered by tests
- chunked training must produce the same state transition as the reference unchunked path within tolerance

The production implementation wraps the pinned upstream Mamba-3 implementation initially. IQ always instantiates it with `is_mimo=True`; there is no SISO fallback or feature flag that changes the trained architecture. If the MIMO/TileLang kernel is unavailable, startup fails explicitly. For BF16 rank-4 training, use the upstream recommended chunk size `64 / mimo_rank = 16`.

Pin the upstream source revision in the production environment. Mamba-3 incremental decode is promoted only after full-sequence vs step/state parity succeeds on the target H200 runtime; failure blocks decode deployment rather than switching to SISO. Custom Triton/Mojo kernels are permitted only after forward/backward/state parity and throughput tests beat or match the pinned upstream MIMO path.

### Transformer context/comparison service

IQ's routed Transformer service follows the newer DeepSeek V4-family **Compressed Sparse Attention (CSA) / Heavily Compressed Attention (HCA)** direction, using the V4.1 kernel family where compatible and is used only where explicit context addressing or pairwise comparison improves the Mamba-3 MIMO reasoning path.

The service exposes three execution modes:

1. `OFF`: no Transformer context call; Mamba-3 MIMO continues recurrent reasoning from its state.
2. `COMPRESSED`: local sliding-window attention plus CSA compressed long-range KV memory. A learned DSA-lineage indexer selects the most relevant compressed entries for each query; HCA-style heavily compressed global memory supplies a lower-frequency long-range anchor.
3. `DENSE`: exact dense/global attention for operations that materially benefit from explicit pairwise token comparison, strict few-shot induction, ambiguous symbol resolution, or diagnostic fallback.

The compressed mode follows the DeepSeek V4 family decomposition:

- sliding-window branch for recent fine-grained dependencies;
- compressed KV entries for distant context;
- learned sparse indexer / top-k selection over compressed entries;
- shared compressed memory representation suitable for efficient decode;
- attention sink support where it improves stability;
- optional hierarchical index search and cross-layer selection reuse only after measured quality/throughput parity.

Required behavior:

- causal masking is exact;
- compressed entries become visible only after their source span is causally complete;
- selection indices are deterministic under fixed seeds;
- no dense `N x N` computation hidden inside `COMPRESSED`;
- dense reference comparisons on small sequences;
- retrieval statistics expose window/compressed/dense utilization, indexer recall, selected-entry overlap, and compression ratios;
- optimized kernels must match the reference within dtype-specific tolerances.

Do not naively copy donor K/V into a shared compressed-KV space. Transport compatible query/output operators, then learn/fit the compressed KV constructor and indexer from calibration activations and dense-teacher behavior. The dense donor/recipient attention path remains the transfer teacher and exact-comparison path.

### Outer/global execution path

Use Differential Attention over compressed token memory + executive state, not every raw token at every inner step.

Two attention maps are computed and combined using the differential parameterization. Stream 1 receives transported donor initialization. Stream 2 starts from the same transported basis plus a small reproducible trainable symmetry-breaking delta; differential-specific lambda parameters are initialized from the published formulation.

This module runs only in the outer refinement loop.

## 7. Expert computation and routing

SwiGLU remains the expert activation, but expert computation is a **scheduled layer type**, not an automatic sublayer attached to every Mamba or attention layer.

### Dense transfer control

The Phi proof keeps its dense SwiGLU:

```text
SwiGLU(x) = W_down( SiLU(W_gate x) * (W_up x) )
```

This preserves donor-compatible gate/up/down transport.

### Hybrid production path

Hybrid `E` layers use routed SwiGLU experts plus a shared expert. The implementation must support:

- top-k expert routing;
- at least one shared expert path;
- expert capacity accounting;
- token drop disabled unless an experiment explicitly changes overflow semantics;
- load-balancing auxiliary loss;
- router z-loss;
- expert parallelism through Megatron-Core;
- grouped GEMM/fused dispatch where supported;
- explicit separation between Mamba/attention layer schedule and expert-layer schedule.

Mellum/code-MoE donors can initialize compatible experts/router through exact/operator/functional transport. Mamba-3 layers do not contain an IQ-added SwiGLU unless the explicit schedule places an `E` layer after them.

### Residual topology: standard control and mHC target

The dense Phi control keeps ordinary residual connections so donor retention remains interpretable.

The hybrid architecture includes **Manifold-Constrained Hyper-Connections (mHC)** as the residual-topology target once its reference implementation is numerically validated. mHC expands the residual stream into multiple streams and constrains the residual mixing matrix to the Birkhoff polytope (doubly stochastic mixing). IQ does not approximate this with an unconstrained learned residual mixer.

Implementation requirements before enabling mHC in training:

- exact reference forward/backward implementation;
- verified doubly-stochastic constraint tolerance;
- identity/near-identity initialization;
- memory-bandwidth profiling;
- fused or recomputed projection only after reference parity;
- standard-residual vs mHC ablation under equal active compute.

mHC changes residual topology, not the role of Mamba, attention, or MoE in the heterogeneous schedule.

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
relative_reasoning_state_delta <= eps_reason
relative_executive_state_delta <= eps_exec
abs(work_adjusted_energy_residual) <= eps_energy
predicted_value_of_more_context <= context_cost
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

Continuous port-Hamiltonian dynamics:

```text
g = grad_z E_theta(z, c)
dz/dt = (J_theta - R_theta) g + B_theta u
```

Structural constraints are enforced by construction:

```text
J = J_ring + U V^T - V U^T
J^T = -J

R = B_ring^T diag(softplus(r_edge)) B_ring
    + diag(softplus(r_self))
R >= 0
```

`B_ring` is the incidence matrix of the executive-state ring. This preserves both ring locality and positive-semidefinite dissipation; post-hoc masking of `L L^T` is not used because masking can destroy PSD structure.

The continuous identity is:

```text
dE/dt = -g^T R g + g^T B u
```

so external context injection can legitimately increase energy. Halting therefore uses an input-work-adjusted energy residual rather than raw `abs(E_next - E)` alone.

The production discrete integrator must preserve the intended stability property. Explicit Euler is only a diagnostic baseline because continuous-time dissipation does not imply finite-step Euler dissipation. Prefer a discrete-gradient/energy-controlled update or a bounded-step integrator with an explicit descent/stability check.

Ring structure:

- split `z` into 8 groups;
- primary learned interactions are self + nearest-neighbor + wraparound edges;
- optional bounded low-rank global skew correction;
- all structure is parameterized directly rather than imposed by masking a dense matrix.

Because backpropagation through `grad_z E` creates higher-order derivatives, keep the energy network small and isolated. Benchmark memory/throughput with `torch.func.grad`/autograd and activation checkpoint this controller if needed.

### Energy must score generated reasoning

The energy model receives the actual recipient trajectory/state/output context. It must never score only the unchanged problem prompt.

Contrastive/ranking training uses successful vs failed/corrupted trajectories from the same task. Example objective:

```text
L_rank = softplus((E_positive - E_negative + margin) / temperature)
L_stationary = ||grad_z E_positive||^2
L_gauge = mean_batch(E)^2
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
- compressed-memory, sliding-window, and learned-indexer parameters
- executive latent injection
- Differential Attention second stream/lambda
- Hamiltonian controller
- concept mapper
- halting controller

Then progressively unfreeze donor-derived blocks based on validation gates.

## 14. Training objectives

Losses are phase-specific; do not enable every loss from step 1.

### Base language/code

- next-token cross entropy;
- FIM formatting/data objective;
- **multi-token prediction (MTP) as a first-class pretraining objective**, with its own shared-weight prediction layers/head configuration rather than an after-the-fact auxiliary linear probe.

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

- same-task positive/negative trajectory ranking;
- successful-state stationarity penalty;
- energy gauge/centering regularization;
- conservative/dissipative diagnostics;
- work-adjusted energy residual used for halting calibration.

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
- context-service mode/window/compressed-memory utilization
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
- compressed-attention indexer recall/top-k overlap/compression metrics
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

1. integrate the pinned Mamba-3 **MIMO** state path with exact recurrent-state semantics; rank-4 MIMO is the production baseline, not an ablation behind a Boolean flag
2. verify MIMO forward/backward, chunk/state continuation, and H200 decode-step parity with no SISO fallback
3. implement the Mamba-dominant context-service hybrid so attention retrieves/compares context and Mamba performs state evolution/reasoning
4. implement bounded context injection/fusion without allowing an always-on Transformer branch to bypass Mamba reasoning
5. implement a DeepSeek V4/V4.1-style compressed context service: sliding window + compressed KV memory + learned top-k indexer
6. add heavily compressed/global memory anchors and periodic mandatory dense/global attention as a safety floor
7. optimized compressed-attention/state kernels only after reference parity
8. outer Differential Attention for noisy/competing retrieved context
9. evaluate hierarchical index search and cross-layer index reuse only after baseline parity
10. evaluate anchor frequency plus event-driven OFF / COMPRESSED / DENSE routing on long-context coding, few-shot induction, pairwise comparison, and needle retrieval

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
