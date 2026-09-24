# IQ Weight Transfer — Production Implementation Plan

Status: canonical implementation plan for transferring pretrained donor capability into IQ v2.

This document complements `SHADOW_TRANSFER.md` and `IQ_V2_IMPLEMENTATION_PLAN.md`. It defines exact donor roles, target parameter slots, calibration/alignment, operator transport, Mamba-3 bootstrap, code/MoE transfer, correction training, provenance, verification, and promotion gates.

## 0. Objectives

The transfer system must:

1. preserve as much pretrained donor capability as possible before novel IQ modules are trained;
2. avoid raw weight copying across incompatible bases/shapes;
3. support multiple donor families without creating model-specific graft architectures;
4. initialize the Transformer and code/MoE portions of IQ from proven pretrained systems;
5. bootstrap the Mamba-3 path from Transformer projections only where there is a published structural correspondence;
6. keep Hamiltonian/EBM, halting, spectral reasoning depth, hyperbolic executive geometry, and concept mapping recipient-native;
7. record the provenance of every initialized parameter;
8. make every transformation reproducible from immutable donor/config/calibration manifests;
9. fail closed on incompatible checkpoints, ambiguous tensor layouts, non-finite maps, or incomplete artifacts.

No production path may contain fake checkpoints, random string/tensor placeholders, silent shape coercion, or undocumented fallback initialization.

## 1. Donor roles

### 1.1 Phi-4 — proof donor and dense Transformer base

Role:

- token embeddings and lexical space for the first proof
- residual-stream calibration
- RMSNorm behavior
- Q/K/V/O projections
- gated MLP gate/up/down projections
- LM head
- baseline language/reasoning behavior

The first proof keeps the Phi tokenizer fixed so token alignment, logits, embeddings, and calibration spans are unambiguous.

### 1.2 Mellum2 Base — code/MoE donor

Use the released base checkpoint for transferable pretrained structure, not an instruction/RL checkpoint unless a later post-training transfer explicitly requires it.

Role:

- software-engineering representations
- sliding/global attention behavior
- GQA structure
- MoE experts/router behavior
- long-context code behavior
- MTP behavior/module where layout permits

FIM is treated primarily as a training/data behavior, not as a standalone tensor.

### 1.3 Mamba-3 — recurrent architecture source

The production Mamba-3 block implementation comes from the official architecture/reference implementation.

Until a verified pretrained Mamba-3 language-model checkpoint with compatible, redistributable weights is selected, there is **no assumed Mamba-3 weight donor**.

Bootstrap the Mamba-3 branch from the Transformer donor only through the documented linear-attention/RNN correspondence:

```text
Transformer V -> Mamba state input x
Transformer K -> Mamba B
Transformer Q -> Mamba C
Transformer O -> Mamba output projection
```

All Mamba-3-specific recurrence parameters that do not have a justified Transformer correspondence use the official Mamba-3 initialization and are learned by distillation/adaptation.

### 1.4 Future larger donors

After the Phi proof passes the scale gate:

```text
larger dense donor
 -> 30–70B donor
 -> 100B+ donor
 -> frontier MoE donor
```

A new donor adds a `DonorInspector`; it never creates a parallel graft architecture.

## 2. Transfer methods

Every target parameter slot declares exactly one initialization method.

### EXACT

Only for identical semantics and compatible shape/basis.

Examples:

- identical tokenizer row and hidden basis
- scalar/vector parameter with identical semantics
- identical expert topology after explicit validation

### OPERATOR_TRANSPORT

For linear operators across different hidden bases/shapes.

Given source activations `X_s`, target activations `X_t`, fit:

```text
X_t ~= X_s P
P = X_s^T (X_s X_s^T + lambda I)^-1 X_t
```

For a source operator `W_s`:

```text
W_t = P_out^T W_s pinv(P_in)^T
```

Use module-specific input/output maps; do not assume the residual-stream map is correct for Q/K/V/expert-intermediate spaces.

### FUNCTIONAL_TRANSFER

For behavior that is not represented by a directly compatible tensor:

- different tokenizer spaces
- attention pattern behavior
- router behavior with different expert topology
- FIM behavior
- recurrent state behavior
- MTP behavior when head layouts differ

Use paired activations/logits/routing/state trajectories on immutable calibration data.

### RECIPE_TRANSFER

No donor weight exists:

- FIM formatting/mixing
- Muon
- GaLore configuration
- RLVR/verifier training
- halting regularization
- attention-anchor schedule

### RECIPIENT_NATIVE

Never transplanted:

- Hamiltonian `J/R/B`
- scalar energy network
- halting head
- reasoning-depth spectral parameters
- hyperbolic executive geometry parameters
- concept mapper
- compressed-context memory/indexer parameters
- Differential Attention lambda/second-stream-only parameters
- Mamba-3 recurrence-specific parameters without a justified source mapping

## 3. Target-slot registry

Add an explicit target registry to `iq_transfer`.

```text
RESIDUAL
EMBEDDING
LM_HEAD
NORM_SCALE

ATTN_Q
ATTN_K
ATTN_V
ATTN_O

MLP_GATE
MLP_UP
MLP_DOWN

MAMBA3_X
MAMBA3_B
MAMBA3_C
MAMBA3_OUT

MOE_EXPERT_GATE
MOE_EXPERT_UP
MOE_EXPERT_DOWN
MOE_ROUTER

MTP_PROJECTION
```

Each slot record contains:

```text
target_module_path
target_slot
target_shape
source_donor_id
source_layer
source_operator
transfer_method
input_map_id
output_map_id
initialization_version
artifact_hashes
verification_metrics
```

No target parameter may be initialized by two donors unless an explicit merge/correction object is configured.

## 4. Immutable donor manifest

Create `iq_transfer/manifest.py`.

A donor manifest must include:

```yaml
schema_version:
donor_id:
architecture_family:
checkpoint_revision:
checkpoint_hash:
config_hash:
tokenizer_hash:
license:
dtype:
num_layers:
hidden_size:
vocab_size:
operator_layout_version:
source_uri:
local_snapshot_id:
```

The manifest is generated from the actual checkpoint and configuration. User-authored values cannot override discovered tensor shapes.

For sharded safetensors, persist:

- index hash
- shard names/hashes
- tensor -> shard mapping
- tensor shape/dtype inventory

Transport refuses to run if any persisted hash changes.

## 5. DonorInspector contract

Extend `DonorInspector` to expose:

```python
manifest() -> DonorManifest
layers() -> Sequence[LayerRef]
operators(layer) -> Sequence[OperatorRef]
activation_taps(layer) -> Sequence[ActivationTap]
tokenizer_spec() -> TokenizerSpec
validate_checkpoint() -> ValidationReport
```

Architecture-specific inspectors are limited to tensor discovery/slicing/semantic labeling.

They do **not** implement transport algorithms.

Initial inspectors:

```text
Phi4Inspector
Mellum2Inspector
future Mamba3Inspector
```

## 6. Calibration corpus

Create a versioned calibration manifest separate from training data.

It must contain:

- natural language
- code completion
- FIM/code editing
- multi-file repository snippets
- long-context retrieval
- reasoning/planning
- repeated identifiers and variable binding
- multiple context depths

Calibration data is split into:

```text
map_fit
map_validation
transfer_validation
```

Never evaluate map quality on the exact samples used to solve the coordinate map.

Persist:

- source document hash
- byte span
- task category
- tokenizer version per donor
- IQ tokenizer version
- packed sequence metadata

## 7. Cross-tokenizer alignment

### Same tokenizer

Use exact token positions and causal prefixes.

### Different tokenizers

Do not align token IDs by index.

Use canonical UTF-8 byte spans:

1. preserve original document bytes;
2. record each tokenizer token's covered byte range;
3. create semantic calibration spans at document/function/statement/identifier boundaries;
4. pool donor activations over tokens whose byte ranges overlap each span;
5. compare/fit maps on the shared span sequence.

For code, record repository path, symbol identity, AST/function span when available, and byte offsets. Byte offsets remain the canonical alignment key.

Embedding rows are copied only for exact shared token strings/IDs under a verified tokenizer mapping. Unmatched vocabulary rows are initialized through lexical/functional alignment, not row-number coincidence.

## 8. Activation capture

Add `iq_transfer/capture.py`.

Capture, in inference/eval mode:

### Transformer donor

- residual pre-attention
- Q/K/V projected activations
- attention output
- residual pre-MLP
- gate/up hidden activations
- MLP output
- final residual/logits

### Mellum MoE donor

Additionally:

- router logits
- top-k expert ids
- routing weights
- per-expert input/output activations
- expert utilization histogram
- MTP hidden/output state if present

### IQ recipient

Capture corresponding target spaces plus:

- Mamba-3 x/B/C projections
- Mamba recurrent state summaries
- compressed-context service outputs
- global attention output
- executive state/energy once those modules exist

Capture files are sharded, dtype-tagged, sample-indexed, and hash-addressed. Do not keep full donor models resident when offline cached activations are sufficient.

## 9. Functional shadows and correspondence

Keep the existing randomized quadratic shadow baseline and add stratified signatures.

For centered, normalized activation matrix `X`:

```text
shadow_j = || U_j X_hat ||^2
```

Use the same measurement seed for paired spaces.

Generate signatures by:

- task category
- context-depth bucket
- token/span type
- code vs natural language
- attention/MLP/expert module

Layer correspondence remains monotonic for sequential Transformer layers.

Persist:

- layer cost matrix
- selected mapping
- shadow error
- held-out activation reconstruction error
- condition numbers
- solver regularization

## 10. Coordinate-map solver

Extend `iq_transfer/transport.py` with typed maps:

```text
ResidualMap
QMap
KMap
VMap
AttentionOutMap
MLPIntermediateMap
ExpertIntermediateMap
MambaBMap
MambaCMap
```

Requirements:

- FP64 solve for small/medium map systems where practical
- configurable ridge regularization
- SVD/pseudoinverse tolerance explicitly recorded
- condition-number diagnostics
- held-out reconstruction metrics
- finite-value checks
- deterministic solver seed where randomized SVD is used
- map artifact saved as safetensors + JSON manifest

A map is rejected if it is rank-deficient beyond configured policy, produces non-finite output, or performs worse than the configured baseline on held-out calibration data.

## 11. Phi -> dense IQ transport

### Embeddings

For the same tokenizer:

```text
E_iq ~= E_phi P_res
```

Use the fitted residual coordinate map.

Added IQ-only special tokens receive deterministic initialized rows and are trained; they are not fabricated from unrelated donor rows.

### Norms

RMSNorm scale vectors are not naively rotated through a dense basis map.

- exact copy only when hidden basis/shape is identical;
- otherwise initialize IQ RMSNorm to the standard identity scale and match normalized activations during correction training.

### Attention

Transport full linear operators, not head numbers by index.

Phi-4 Q/K/V/O are split from fused checkpoint tensors by the inspector.

Fit independent output coordinates for Q and KV spaces:

```text
Wq_iq = transport(Wq_phi, P_res_in, P_q_out)
Wk_iq = transport(Wk_phi, P_res_in, P_k_out)
Wv_iq = transport(Wv_phi, P_res_in, P_v_out)
Wo_iq = transport(Wo_phi, P_attn_in, P_res_out)
```

This naturally handles Phi's 40Q/10KV layout versus IQ's 32Q/8KV layout without arbitrary head truncation.

### SwiGLU

Split the donor fused gate/up tensor before transport.

```text
W_gate_iq
W_up_iq
W_down_iq
```

Each receives its own input/output coordinate map.

### LM head

When vocabulary is identical, map the hidden basis while preserving vocabulary rows.

When token spaces differ, use the cross-tokenizer policy and do not compare logits by raw vocabulary index.

## 12. Transformer -> Mamba-3 bootstrap

Mamba-3 MIMO becomes the primary recurrent reasoning/state path after bootstrap. The dense Transformer remains the retention teacher and exact context/comparison path during transition.

The official Mamba-3 projection packs:

```text
[z, x, B, C, dd_dt, dd_A, trap, angle]
```

The justified Transformer initialization is:

```text
V -> x
K -> B
Q -> C
O -> out_proj
```

Because Mamba-3 state and MIMO dimensions differ from Transformer head spaces, fit dedicated target maps:

```text
P_v_to_x
P_k_to_B
P_q_to_C
P_o_to_mamba_out
```

Populate only those slices of `Mamba3.in_proj`.

Use official Mamba-3 initialization for:

- z/gating rows
- dd_dt
- dd_A
- trap
- angle
- dt_bias
- B/C biases after semantic validation
- D skip parameter
- B/C norms
- MIMO x/z/out factors
- any recurrence state parameter without a published source correspondence

Do not clone Q/K/V weights into these fields.

### Mamba bootstrap training

The dense Phi recipient remains a **separate frozen teacher/control model**. IQ does not preserve transfer capability by keeping an always-on parallel attention branch inside every Mamba layer.

Construct the explicit heterogeneous hybrid schedule and align each new Mamba-3 layer against the teacher states corresponding to its schedule position.

Train Mamba-3 to match:

1. teacher intermediate residual states at schedule-aligned anchor points;
2. downstream residual state;
3. final donor logits;
4. long-context chunk/state-continuation behavior;
5. recurrence-state consistency across packed/chunked execution.

Loss:

```text
L_mamba =
  lambda_anchor * representation_loss(student_anchor, teacher_anchor)
+ lambda_kl * token_KL
+ lambda_state * chunk_continuation_loss
+ lambda_step * recurrent_step_consistency
```

Mamba-3's native B/C normalization, complex/data-dependent state rotation, exponential-trapezoidal recurrence, and MIMO factors remain native. The Transformer-derived Q/K/V/O correspondence seeds only the justified x/B/C/out projections; it does not redefine Mamba-3 semantics.

## 13. DeepSeek V4/V4.1-style compressed context initialization

The routed context service follows the newer DeepSeek compressed-attention direction: sliding-window local attention plus compressed long-range KV memory and learned sparse indexing.

Keep the dense transported Transformer path as the teacher/exact-comparison path during transition.

Compatible initialization:

- transport/retain query projections where the target query space is compatible;
- transport/retain grouped output projection structure only where dimensions/semantics match;
- initialize per-head Q/K or Q/compressed-KV RMSNorm scales to identity unless an exactly compatible donor exists;
- fit a dedicated donor-residual -> shared compressed-KV map from calibration activations;
- initialize the sliding-window branch from the dense attention teacher where shapes permit.

Positional behavior is recipient-native:

- dense anchors use Q/K RMSNorm followed by ordinary RoPE;
- CSA/HCA use normalized query/compressed-KV states, trailing partial RoPE, and inverse rotary transformation on the output rotary slice;
- Mamba-3 native state rotation is never replaced by Transformer RoPE.

Recipient-native parameters:

- Q/K or Q/compressed-KV normalization parameters when no exact donor equivalent exists;
- partial-RoPE/inverse-output positional machinery;
- compressed-KV constructor/compressor;
- learned sparse indexer and top-k selection projections;
- heavily compressed/global memory constructor;
- attention-sink parameters;
- hierarchical indexer parameters, if enabled;
- cross-layer index-cache/reuse policy, if enabled.

Do **not** average donor K and V or copy them row-wise into a shared K=V compressed memory. Fit the compressed memory representation functionally from paired activations.

Train against the dense donor/recipient teacher using:

- context-service output distillation;
- selected-context recall against dense attention mass;
- long-range retrieval loss;
- causal visibility invariants for compressed entries;
- indexer top-k recall/overlap;
- compression reconstruction loss;
- exact pairwise/few-shot control tasks that must route to `DENSE`.

Hierarchical indexing and cross-layer selection reuse are performance optimizations, not excuses to change behavior. Promote them only after they preserve retrieval quality and improve measured throughput/latency.

## 14. Periodic global attention

Global anchor blocks retain a direct transported Transformer attention path.

Default cadence:

```text
compressed-context blocks with periodic dense/global anchors
```

The cadence remains a config, not a hardcoded architectural constant.

Transfer evaluation compares at least:

```text
8:1
4:1
3:1
2:1
```

on retrieval/coding/throughput. This experiment decides attention frequency, not whether Transformer attention should exist.

## 15. Differential outer attention

Initialize stream 1 from a transported global-attention operator.

Initialize stream 2 from the same transported basis plus a deterministic small symmetry-breaking perturbation.

Initialize differential-specific lambda parameters from the published formulation.

Train it on compressed/global memory only. It does not replace the token-level Transformer path.

## 16. Mellum2 -> code/MoE IQ transfer

### Stage A — code representation transfer

Capture the same code calibration spans from Phi/IQ and Mellum.

Use Mellum as a secondary code specialist, not a competing owner of every base parameter.

Code transfer should enter through:

- code-specialized DoRA residuals;
- MoE experts/router;
- MTP path;
- long-context attention behavior.

Do not elementwise average Phi and Mellum base weights.

### Stage B — expert matching

For every source expert, build a functional signature from:

- input activation shadow
- output activation shadow
- gate/up hidden statistics
- router utilization
- code/task-category utilization
- output reconstruction on routed calibration samples

Match source experts to IQ expert slots using minimum-cost assignment / optimal transport over these signatures.

Never assume expert index 17 in the donor means expert 17 in IQ.

### Stage C — expert operator transport

Transport each matched expert's:

```text
gate
up
down
```

using residual and expert-intermediate maps.

The first donor-compatible MoE stage should preserve Mellum's 64-expert / top-8 routing topology where feasible so code knowledge transfer is measurable without simultaneously changing expert cardinality.

### Stage D — heterogeneous capacity transition

After the transferred MoE is stable, heterogeneous expert capacities become an IQ architecture experiment.

To expand a selected expert:

- retain the transported core subspace;
- add capacity through new orthogonal/residual channels;
- initialize new output contribution near zero;
- train without changing the original core abruptly.

To shrink an expert, use activation-informed low-rank projection; never truncate rows by index.

### Router

If expert count/top-k topology matches, operator-transport the router linear map and validate routing agreement.

If topology differs, use functional router distillation:

```text
donor router logits/assignments
 -> matched IQ expert identities
 -> cross-entropy/KL routing loss
```

## 17. MTP and FIM

### MTP

MTP is a first-class pretraining component rather than a single auxiliary linear probe.

If donor and IQ MTP topology/tokenizer match, transport compatible normalization/projection/prediction-layer operators and shared lexical output space.

If topology differs, align future-token hidden states/logits functionally and train the IQ MTP prediction layers in the IQ lexical space. Record each MTP prediction layer separately in provenance; do not collapse a multi-layer donor MTP module into one tensor.

### FIM

FIM has no standalone transferable tensor.

Transfer through:

- tokenizer/control-token mapping;
- pretrained backbone behavior already present in the code donor;
- continued deterministic PSM/SPM training.

## 18. Multi-donor conflict policy

Every base target operator has a single owner.

Initial policy:

```text
Phi -> dense base
Mamba bootstrap -> Mamba-3 MIMO primary recurrent path
Mellum -> code DoRA + MoE + MTP
IQ-native -> Mamba recurrence-only parameters + QK/KV norms + CSA/HCA compression/indexing/partial-RoPE + mHC residual topology + executive/energy/halting/concept modules
```

No naive parameter averaging.

When two donors contribute to the same function, use one of:

1. base + low-rank donor-specific residual;
2. separate experts;
3. learned bounded residual branch;
4. explicit function-space merge evaluated on held-out calibration data.

All merge decisions are stored in the `TransportPlan`.

## 19. DoRA correction

Every transported dense operator may be wrapped by TP-aware DoRA.

Initial schedule:

1. freeze transported base;
2. train DoRA + new IQ modules;
3. evaluate retention;
4. unfreeze selected transported blocks with a lower learning rate;
5. optionally merge DoRA for export after exact merge-parity validation.

Code-specific Mellum contribution should initially use separate DoRA namespaces from general Phi correction so its effect can be ablated and removed independently.

## 20. Training phases

### T0 — artifact verification

- inspect donor checkpoint
- hash all shards/config/tokenizer
- validate tensor inventory
- generate donor manifest
- run inspector layout tests

Exit: zero ambiguous or missing required operators.

### T1 — Phi dense transport

- capture calibration activations
- solve layer correspondence
- solve residual/Q/K/V/MLP maps
- transport embeddings/QKV/O/SwiGLU/LM head
- initialize norms safely
- load into dense IQ recipient

Exit: forward/backward works and transfer metrics are recorded.

### T2 — dense correction

Train:

- DoRA
- new positional/config adapters
- CE/FIM/MTP
- shadow/representation alignment
- optional donor KL

Keep base mostly frozen.

Exit: donor-retention and adaptation-compute gates pass.

### T3 — Mamba-3 MIMO primary-path bootstrap

- instantiate the explicit heterogeneous schedule
- initialize x/B/C/out from justified attention transport
- preserve official initialization for recurrence-only parameters
- align Mamba schedule anchors against the frozen dense teacher
- run chunk/state continuation and recurrent-step consistency training
- keep attention/MoE as separate scheduled layer types rather than parallel branches inside Mamba

Exit: hybrid retains exact retrieval while reducing long-horizon state/compute cost relative to the dense baseline.

### T4 — compressed context service + dense/global anchors

- add per-head Q/K normalization to dense anchors
- keep compatible transported dense query/output projections
- train compressed-KV construction and learned sparse indexing
- implement CSA/HCA trailing partial RoPE and inverse output rotation
- retain explicit dense/global attention anchors in the schedule
- benchmark full heterogeneous schedules rather than one global-attention-period scalar

Exit: long-context retrieval/code dependency suite remains within retention tolerance while memory/throughput improves.

### T5 — residual topology + outer Differential Attention + recurrence

- validate standard-residual hybrid first
- implement mHC with an exact doubly-stochastic reference projection and measure memory/throughput
- initialize Differential stream 1 from transferred global attention
- add second stream/lambda
- add adaptive outer/inner recurrence

Exit: reasoning/compute tradeoff improves over non-recurrent hybrid control.

### T6 — Hamiltonian/EBM executive

Train only from generated recipient trajectories; never static prompts.

Exit: energy separates successful/failed trajectories and improves reasoning or halting efficiency.

### T7 — Mellum code/MoE transfer

- Mellum inspector
- cross-tokenizer byte-span alignment
- expert functional signatures
- expert/router transport
- code DoRA
- MTP transfer/distillation
- FIM/code mid-training

Exit: repository/code suite improves without unacceptable general/long-context regression.

### T8 — continuous concepts

- Coconut path
- Soft Thinking
- concept mapper

Each is ablated independently.

### T9 — larger donors

Replace/upgrade the base donor only after the scale gate passes. Recompute maps and provenance; do not indefinitely accumulate donor residuals.

## 21. Loss scheduling

Do not optimize every objective from step 1.

Dense correction:

```text
L = L_CE
  + lambda_shadow * L_shadow
  + lambda_repr * L_repr
  + lambda_KL * L_KL
  + lambda_MTP * L_MTP
```

Mamba bootstrap adds:

```text
+ lambda_block * L_block
+ lambda_state * L_state_continuation
```

MoE stage adds:

```text
+ lambda_router * L_router
+ lambda_balance * L_balance
+ lambda_expert * L_expert
```

Executive stage adds:

```text
+ lambda_energy * L_energy
+ lambda_ponder * L_ponder
```

Every coefficient and schedule is versioned in the experiment config and checkpoint manifest.

## 22. Numerical and functional verification

### Unit tests

- fused donor tensor slicing
- target-slot registry completeness
- shape rejection
- exact small-matrix transport
- pseudoinverse tolerance
- map serialization round-trip
- provenance serialization
- embedding/LM-head orientation
- Mamba in_proj slice placement
- Mamba chunk vs unchunked state equivalence
- router expert remapping
- DoRA merge/unmerge parity

### FP32 reference tolerances

Controlled synthetic transport tests should reproduce exact linear operators within numerical tolerance.

Production BF16/FP8 tolerances are dtype-specific and measured against the BF16 reference; they are not silently widened when tests fail.

### Model-level tests

- next-token logit comparison
- hidden-state cosine/reconstruction
- attention block-output reconstruction
- long-context needle retrieval
- multi-needle retrieval
- repeated-identifier binding
- code symbol lookup
- cross-file dependency recall
- FIM
- repository repair
- chunked Mamba state continuation
- restart/checkpoint continuation

## 23. Promotion metrics

Record at every transfer stage:

```text
donor score
IQ pre-adaptation score
IQ post-adaptation score
scratch/control score
capability retention
shadow error
held-out map error
block output error
logit KL
adaptation tokens
adaptation FLOPs
wall-clock
peak HBM
tokens/sec
needle retrieval by depth
code/repository pass@1
Mamba state memory
KV-cache memory
executed recurrent steps
```

The existing initial scale gate remains the starting policy, not a claimed universal constant:

```text
retention >= 0.80
IQ >= scratch/control
adaptation_compute / scratch_compute <= 0.50
```

Tune promotion thresholds only after the Phi proof establishes empirical baselines.

## 24. Provenance in checkpoints

Every final IQ checkpoint records parameter-level initialization lineage.

At minimum:

```text
target parameter
source donor/revision
source tensor/slice
transfer method
map ids/hashes
DoRA namespace
training phase introduced
last phase unfrozen
```

Also persist:

- donor manifests
- calibration manifest
- transfer plan
- map artifacts
- tokenizer mapping
- code expert matching
- training config
- git SHA
- container digest

An exported checkpoint without its transfer provenance is incomplete.

## 25. CLI

Implement a deterministic CLI around the library:

```bash
python -m iq_transfer.cli inspect --donor <snapshot>
python -m iq_transfer.cli capture --plan <config>
python -m iq_transfer.cli solve-maps --plan <config>
python -m iq_transfer.cli build-plan --config <config>
python -m iq_transfer.cli apply --plan <artifact>
python -m iq_transfer.cli verify --plan <artifact>
python -m iq_transfer.cli report --plan <artifact>
```

CLI commands produce immutable artifacts and machine-readable reports. Interactive prompts are not allowed in production jobs.

## 26. Repository implementation order

1. `iq_transfer/manifest.py`
2. `iq_transfer/slots.py`
3. extend `iq_transfer/donor.py`
4. `iq_transfer/capture.py`
5. `iq_transfer/alignment.py` for byte-span/cross-tokenizer alignment
6. extend `iq_transfer/shadows.py`
7. extend `iq_transfer/transport.py` with typed maps
8. `iq_transfer/plan.py`
9. `iq_transfer/provenance.py`
10. `iq_transfer/mamba3_init.py`
11. `iq_transfer/mellum2.py`
12. `iq_transfer/moe.py`
13. `iq_transfer/cli.py`
14. integration with `iq_model`
15. integration with `iq_training`
16. transfer/e2e test suite

No file is introduced as an empty scaffold: each commit must add executable behavior plus its tests.

## 27. First implementation milestone

The first milestone is deliberately narrower than the final multi-donor model:

```text
official Phi checkpoint
 -> immutable manifest
 -> real calibration capture
 -> layer/shadow alignment
 -> typed coordinate maps
 -> Q/K/V/O + SwiGLU + embedding/LM transport
 -> dense IQ recipient
 -> DoRA correction
 -> train/eval/checkpoint
 -> Mamba-3 x/B/C/out bootstrap
 -> hybrid block distillation
```

Do not start Mellum/MoE transfer until this path is green end-to-end.

A milestone is complete only when:

- no active mocks/placeholders exist;
- all checkpoint hashes are verified;
- transfer artifacts are reproducible;
- every transported parameter has provenance;
- the recipient trains forward/backward;
- interruption/resume works;
- the evaluator loads the checkpoint;
- dense donor retention is measured;
- Mamba state continuation is tested;
- hybrid needle/code retrieval is measured against the dense Transformer baseline.
