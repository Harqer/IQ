# IQ Hybrid Research Architecture

This document is the authoritative architecture target for `architecture/iq-hybrid-v1`.
The previous `architecture/iq-research-v0` branch remains the Phi-compatible recurrent-depth
control. The older `nif_sovereign/` implementation is legacy reference code.

## Design rule

Keep three concerns separate:

1. **Architecture** — the function IQ computes.
2. **Training/optimization** — how IQ learns it.
3. **Transfer/deployment** — how pretrained knowledge and runtime compute are supplied.

MOHAWK, DoRA, Muon, GaLore, quantization, Colab, Hugging Face hosting, and QPU access are not
model layers.

## Hybrid-v1 topology

```text
input tokens
    |
    v
embedding
    |
    v
4 unique Phi-compatible prelude blocks
    |
    v
+---------------- recurrent reasoning core ----------------+
|                                                          |
|  0  Gated DeltaNet                                      |
|  1  Gated DeltaNet                                      |
|  2  Native Sparse Attention anchor                      |
|  3  Gated DeltaNet                                      |
|  4  Gated DeltaNet                                      |
|  5  Native Sparse Attention anchor                      |
|  6  Gated DeltaNet                                      |
|  7  Native Sparse Attention anchor                      |
|                                                          |
|                 repeat physical stack R=3               |
+--------------------------+-------------------------------+
                           |
                           v
4 unique Phi-compatible coda blocks
                           |
                           v
final RMSNorm
                           |
             +-------------+-------------+
             |             |             |
            LM            MTP         verifier
```

Default core ratio: **5 recurrent linear mixers : 3 sparse exact-attention anchors**.
A stricter 6:2 / 3:1 schedule remains an ablation, not an undocumented change.

## Two recurrence axes

IQ intentionally has two different kinds of recurrence:

1. **Sequence recurrence** — Gated DeltaNet carries compressed recurrent state across tokens.
2. **Depth recurrence** — the eight physical reasoning blocks are revisited across reasoning passes.

These mechanisms solve different problems and must not be conflated.

## Effective depth and donor alignment

The physical topology stays `4 + 8 + 4 = 16` blocks while the default effective depth is:

```text
4 + (8 x 3) + 4 = 32
```

This keeps the exact Phi-4-mini teacher-depth correspondence:

```text
prelude[0..3]              -> teacher layers 0..3
core pass 0, blocks 0..7  -> teacher layers 4..11
core pass 1, blocks 0..7  -> teacher layers 12..19
core pass 2, blocks 0..7  -> teacher layers 20..27
coda[0..3]                 -> teacher layers 28..31
```

`iq_model/transfer_layout.py` is the source of truth for this mapping.

## Mixer contract

`iq_model/mixers.py` owns heterogeneous sequence mixing. Every mixer receives a `MixerContext`
that carries both forms of masking/position state required by the different operator families:

- 2-D padding mask for recurrent/sparse kernels;
- additive causal mask for dense GQA;
- position ids;
- Phi-compatible RoPE values for the transfer-control path.

This avoids incorrectly forcing Gated DeltaNet or NSA through dense-attention mask semantics.

### Implemented mixer families

#### Phi-compatible GQA

Used in the unique prelude/coda and by the v0 control. Parameter names/shapes remain compatible
with Phi-4-mini (`qkv_proj`, `o_proj`).

#### Gated DeltaNet

Implemented through the pinned Flash Linear Attention (FLA) implementation. For Phi-4-mini:

```text
hidden size       = 3072
key width ratio   = 0.75
key width         = 2304
head dim          = 128
GDN heads         = 18
value expansion   = 2.0
```

Short convolution remains enabled because it is part of the validated GDN design.

#### Native Sparse Attention

Implemented through FLA NSA with three learned branches:

- compressed/global context;
- selected sparse blocks;
- sliding/local context.

Default research settings:

```text
block size   = 64
block count  = 16
local window = 512
```

The exact values are experiment parameters, not universal constants.

#### PaTH attention

FLA PaTH is available as a **standalone attention candidate**. It applies data-dependent
Householder-style position transformations as part of its attention operator.

It is intentionally *not* silently injected into NSA. `attention_position_strategy="path"`
with ordinary NSA fails explicitly until the combined operator is derived and validated.

## MLA + NSA target

The final sparse-attention direction remains latent NSA:

```text
local branch        -> Multi-head Latent Attention (MLA)
global compression  -> latent/group-head attention
global selection    -> latent/group-head attention
```

This follows the published latent-NSA direction that reduces KV-cache cost while retaining
NSA's local/compression/selection structure.

`latent_nsa` exists as an architecture identifier but currently raises
`UnsupportedMixerComposition`. Ordinary NSA must never be relabeled as latent NSA.

Transition gate:

1. validate GDN + NSA recurrent core;
2. implement latent branch projections/cache layout;
3. reproduce dense-vs-latent NSA equivalence/quality controls;
4. only then make `latent_nsa` executable.

## Position strategy

Current executable sparse path: **NSA + its validated RoPE implementation**.

Research target: **PaTH/data-dependent position on global attention paths**, but only after its
interaction with latent/sparse cache structure is mathematically and empirically validated.

PaTH can already be tested independently by placing `path_attention` in the mixer schedule.

## Feed-forward network

Hybrid-v1 still uses dense, transfer-compatible fused SwiGLU:

```text
gate, up = gate_up_proj(x).chunk(2)
y = down_proj(SiLU(gate) * up)
```

This is intentional. Changing recurrent mixers and FFN routing simultaneously would make a
failed transfer difficult to diagnose.

Next FFN stage:

```text
shared SwiGLU path (always active)
        +
top-k routed fine-grained experts
```

Experts must specialize through training. Do not restore manually named linguistic/physics/
diffusion experts from the legacy NIF architecture.

## Keystone neurons

Keystone neurons are treated as an **empirical property discovered after pretraining/transfer**,
not a predeclared neuron class.

`PhiCompatibleSwiGLU` exposes an activation-observer hook, and
`iq_model/keystone.py` collects cross-task mean absolute activation statistics.

The monitor does not yet:

- label neurons as keystone;
- freeze or protect weights;
- alter learning rates;
- prune neurons.

Those actions require reproducing the published identification and ablation protocol first.

## Recurrent reasoning core

`iq_model/reasoning.py` owns depth recurrence. Physical blocks are shared across passes and keep
fixed mixer identities.

```text
candidate = block(h)
h_next = h + alpha * (candidate - h)
```

`alpha=1` remains the transfer-safe baseline. Learned/per-pass/gated residual policies are later
controlled experiments.

Zero-initialized pass embeddings let the shared core learn visit identity without changing the
initial transplanted function.

## Not yet implemented in hybrid-v1

These remain explicit architecture stages rather than hidden placeholders:

- persistent latent reasoning workspace;
- Mixture-of-Recursions/token-dependent depth;
- learned halting;
- shared+routed MoE;
- executable latent NSA;
- PaTH-inside-latent-NSA;
- code-graph structural input stream;
- fast-weight/repository memory;
- FSP/HLP/FIM objective heads and losses;
- Hamiltonian/conservative-dissipative recurrence ablation.

## Dependencies

Base research environment:

```text
torch==2.14.0
transformers==5.17.0
```

Hybrid kernels are supplied by a pinned FLA commit in `requirements-hybrid.txt`. Do not float to
FLA `main` during an architecture experiment.

## Current source map

- `iq_model/config.py` — topology, mixer schedule, kernel parameters, donor-depth mapping.
- `iq_model/mixers.py` — Phi GQA, Gated DeltaNet, NSA, PaTH adapters, unsupported latent-NSA gate.
- `iq_model/components.py` — pre-norm block, dense SwiGLU, keystone activation hook.
- `iq_model/reasoning.py` — heterogeneous depth-recurrent reasoning core.
- `iq_model/model.py` — embedding -> prelude -> hybrid recurrent core -> coda -> heads.
- `iq_model/keystone.py` — cross-task activation instrumentation.
- `iq_model/transfer_layout.py` — dense teacher -> recurrent depth correspondence.
- `tests/test_iq_model.py` — architecture invariants without FLA/CUDA.
- `tests/test_hybrid_mixers.py` — conditional CUDA/FLA kernel smoke tests.

## Transition order before weight insertion

1. Run Phi-control tests and numerical GQA equivalence.
2. Install `requirements-hybrid.txt` on Colab GPU.
3. Run Gated DeltaNet + NSA CUDA smoke tests.
4. Run a tiny full hybrid recurrent forward/backward stability test.
5. Establish hybrid-v1 random-init numerical health (no NaNs, bounded activation growth).
6. Only then define the Phi -> hybrid operator-distillation/MOHAWK mapping.
7. After hybrid transfer is stable, implement latent NSA and PaTH integration as separate gates.
