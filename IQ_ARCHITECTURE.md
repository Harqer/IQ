# IQ Mamba Hybrid-v2 Architecture

This document is the authoritative architecture target for `architecture/iq-mamba-v2`.

Controls remain available:

- `architecture/iq-research-v0` — Phi-compatible recurrent-depth control.
- `architecture/iq-hybrid-v1` — Gated DeltaNet + NSA hybrid control.
- `architecture/iq-mamba-v2` — Mamba-3 MIMO + NSA + recurrent-depth research target.

The older `nif_sovereign/` physics-inspired implementation remains legacy reference code.

## Design principle

Use each mechanism only when it solves a distinct problem. Do not combine mechanisms merely because they are novel.

1. **Mamba-3 MIMO** — compressed recurrent sequence state.
2. **Native Sparse Attention** — exact/content-addressable long-range retrieval anchors.
3. **Depth recurrence** — repeated reasoning compute with shared physical parameters.
4. **JEPA-style latent prediction** — representation-space future-state supervision.
5. **Dense SwiGLU initially** — stable FFN control while the new sequence architecture is measured.

Hamiltonian/conservative-dissipative dynamics are removed from the active v2 target. Mamba-3 already supplies learned state-space dynamics; stacking a second speculative dynamical system would duplicate responsibilities without comparable LLM evidence.

## Default topology

```text
input tokens
    |
    v
embedding
    |
    v
4 unique Phi-compatible transfer-boundary blocks
    |
    v
+---------------- recurrent reasoning core ----------------+
|                                                          |
|  0  Mamba-3 MIMO                                        |
|  1  Mamba-3 MIMO                                        |
|  2  Native Sparse Attention anchor                      |
|  3  Mamba-3 MIMO                                        |
|  4  Mamba-3 MIMO                                        |
|  5  Native Sparse Attention anchor                      |
|  6  Mamba-3 MIMO                                        |
|  7  Native Sparse Attention anchor                      |
|                                                          |
|                 repeat physical stack R=3               |
+--------------------------+-------------------------------+
                           |
                           +--> JEPA-style latent predictor
                           |
                           v
4 unique Phi-compatible transfer-boundary blocks
                           |
                           v
final RMSNorm
                           |
                    LM / auxiliary heads
```

Physical depth remains 16 blocks. Effective depth remains 32:

```text
4 + (8 x 3) + 4 = 32
```

Keeping the same macro-depth as v0/v1 makes the sequence-operator transition attributable.

## Why Mamba-3 is primary

Mamba-3 is the primary v2 sequence mixer because it is a current state-space architecture designed to improve recurrent sequence modeling while retaining linear-time/constant-state execution. IQ uses the official MIMO implementation rather than recreating the operator.

Default research geometry:

```text
state size       = 128
head dim         = 64
expansion        = 2
MIMO rank        = 4
chunk size       = 16
RoPE fraction    = 0.5
```

The chunk/rank relationship follows the official bf16 MIMO kernel recommendation.

Mamba-3 is not assumed to be universally superior. `architecture/iq-hybrid-v1` remains the matched Gated DeltaNet control, and Gated DeltaNet-2 is a future matched comparator because current NVIDIA results show it can outperform Mamba-3 variants on several 1.3B matched evaluations.

## Exact-retrieval anchors

A fixed-size recurrent state is efficient but can lose exact details. IQ therefore retains sparse attention anchors.

Current executable anchor: **Native Sparse Attention (NSA)**.

NSA contributes:

- local/sliding retrieval;
- compressed/global context;
- selected sparse blocks.

Default v2 schedule uses five Mamba-3 blocks and three NSA anchors. This is an experimental ratio, not a universal constant.

### Later anchor candidates

Evaluate independently against NSA:

- log-linear attention — logarithmically growing state to relax the fixed-state bottleneck;
- PaTH attention — data-dependent positional transformations;
- latent NSA — MLA/GLA-style latent cache inside sparse branches.

Do not enable these simultaneously in the default architecture.

## Two recurrence axes

IQ intentionally contains two different kinds of recurrence.

### Sequence recurrence

Mamba-3 carries state across token positions.

### Reasoning/depth recurrence

The eight physical middle blocks are reused across reasoning passes.

```text
physical core
    -> pass 0
    -> pass 1
    -> pass 2
```

This remains motivated separately from sequence recurrence by looped/recurrent-depth work, especially code-oriented evidence such as LoopCoder.

The two mechanisms must remain independently ablatable.

## JEPA-style latent prediction

V2 adds `FutureLatentPredictor` after the recurrent reasoning core.

The predictor outputs a representation, not tokens:

```text
reasoning representation h_r
          |
          v
FutureLatentPredictor
          |
          v
predicted target representation z_hat
```

The training pipeline owns target construction. Candidate targets include:

- future token-span representation;
- future summary representation;
- later recurrent-pass representation;
- completed-function representation;
- code-structure/state representation.

This is intentionally broader than directly copying VL-JEPA. VL-JEPA provides evidence for prediction in continuous representation space; IQ must validate which language/code target is useful.

The latent predictor is auxiliary. Autoregressive LM generation remains available and should not be removed before representation-space objectives prove their value.

## Position handling

Mamba-3 owns its internal state-space positional dynamics and MIMO rotary mechanism.

NSA currently retains its validated RoPE implementation.

PaTH is an attention-family experiment only. It must not be inserted into Mamba-3 recurrence as though it were a generic positional embedding.

`PaTH + NSA` or `PaTH + latent NSA` require their own derivation and validation.

## Feed-forward network

V2 keeps dense fused SwiGLU:

```text
gate, up = gate_up_proj(x).chunk(2)
y = down_proj(SiLU(gate) * up)
```

This is not a belief that dense FFNs are final. It keeps the Mamba transition measurable.

Next FFN experiment:

```text
shared always-on SwiGLU
        +
fine-grained routed experts
```

No semantic expert labels are allowed. Experts specialize through training.

Keystone-neuron instrumentation remains observational only until cross-task identification and causal ablation reproduce the reported effect.

## Adaptive depth

V2 uses fixed `R=3` depth recurrence first.

Later experiments may compare Mixture-of-Recursions/token-level depth routing and learned halting, but adaptive compute must not be introduced before the fixed recurrent Mamba core is numerically stable.

## Code-specific structure

A code-graph/AST/symbol stream remains a planned input experiment, not part of the v2 default. It is orthogonal to the Mamba sequence operator and should be introduced only after token-only baselines exist.

Candidate structural nodes/edges include:

- file/function/class/symbol/type/import nodes;
- call, ownership, type, dependency and test relationships.

## Transfer strategy

V2 is deliberately farther from Phi than v0/v1.

Compatible boundary components may still be copied directly where semantics match, but Mamba-3 and NSA must be learned through functional transfer/distillation rather than assumed tensor equivalence.

Canonical sequence:

```text
Phi donor
  |
  +--> direct copy: genuinely compatible embeddings/norms/FFN/head
  |
  +--> teacher operator/activation traces
  |
  +--> Mamba-3 / NSA block-level functional alignment
  |
  +--> recurrent-depth supervision across teacher depths
  |
  +--> end-to-end logit / LM distillation
  |
  `--> continued training
```

The transfer framework must become donor-neutral before architecture-independence is claimed. A later experiment must repeat transfer from a second non-Phi donor.

## Explicitly excluded from v2 default

- Hamiltonian latent/state dynamics;
- Mamba-3 + Gated DeltaNet in the same default stack;
- MoE enabled simultaneously with first Mamba transfer;
- PaTH injected into Mamba state;
- ordinary NSA renamed as latent NSA;
- adaptive halting before fixed-depth stability;
- manually named semantic experts;
- quantum/neutrino/Ising layers from legacy NIF.

## Implementation-language boundary

The research reference remains Python where upstream PyTorch/Triton implementations are required. Stable components should migrate according to:

```text
Mojo / MAX  -> accelerator hot paths and custom tensor kernels
Rust        -> runtime, checkpoint/transfer, tokenizer/data, state ownership, evaluation
Go          -> distributed orchestration, workers, telemetry, services
Python      -> upstream research compatibility, training experiments, notebooks
```

Do not rewrite a validated CUDA/Triton kernel solely to eliminate Python. Port only after the architecture wins its ablation and numerical equivalence can be tested.

## Source map

- `iq_model/config.py` — topology, Mamba/NSA settings, controls.
- `iq_model/mixers.py` — Phi GQA, Mamba-3 MIMO, GDN control, NSA, PaTH adapters.
- `iq_model/components.py` — pre-norm block + dense SwiGLU.
- `iq_model/reasoning.py` — shared depth-recurrent core.
- `iq_model/latent.py` — JEPA-style future latent predictor.
- `iq_model/keystone.py` — neuron instrumentation.
- `iq_model/model.py` — complete v2 backbone.
- `iq_model/transfer_layout.py` — current Phi depth correspondence control.

## Validation gate before donor weights

1. Phi-only control remains numerically green.
2. GDN+NSA hybrid-v1 control remains available.
3. Official Mamba-3 MIMO adapter passes CUDA forward/backward tests.
4. Mamba-3 + NSA recurrent composition has finite activations and gradients over repeated passes.
5. Packed-sequence semantics are implemented/tested for Mamba training.
6. JEPA predictor target construction is defined in the training experiment, not guessed inside the model.
7. Only then implement Phi -> Mamba-3/NSA functional distillation.
