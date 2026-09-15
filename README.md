# IQ

IQ is an experimental language-model architecture for **complex coding and multi-step reasoning**.

The active architecture work is `architecture/iq-mamba-v2`.

Controls:

- `architecture/iq-research-v0` — Phi-compatible recurrent-depth control.
- `architecture/iq-hybrid-v1` — Gated DeltaNet + Native Sparse Attention control.
- `architecture/iq-mamba-v2` — Mamba-3 MIMO + Native Sparse Attention + depth recurrence.

The old `nif_sovereign/` physics-inspired implementation is legacy reference code.

## Current v2 architecture

```text
embedding
   |
4 unique Phi-compatible transfer-boundary blocks
   |
   v
recurrent core, repeated 3 passes
   |
   |-- 0 Mamba-3 MIMO
   |-- 1 Mamba-3 MIMO
   |-- 2 Mamba-3 MIMO
   |-- 3 Native Sparse Attention
   |-- 4 Mamba-3 MIMO
   |-- 5 Mamba-3 MIMO
   |-- 6 Mamba-3 MIMO
   `-- 7 Native Sparse Attention
   |
   +--> JEPA-style future latent predictor
   |
4 unique Phi-compatible transfer-boundary blocks
   |
final RMSNorm
   |
LM / optional auxiliary heads
```

Physical depth is 16 blocks and effective depth remains 32:

```text
4 + (8 x 3) + 4 = 32
```

The default intentionally separates responsibilities:

- **Mamba-3 MIMO** — recurrent sequence state;
- **NSA** — precise local/global retrieval anchors;
- **depth recurrence** — shared repeated reasoning compute;
- **latent predictor** — JEPA-style representation-space supervision;
- **dense SwiGLU** — FFN control for the first Mamba transfer.

The default core is **6 Mamba-3 : 2 NSA**. The Mamba-3 paper's hybrid experiments use roughly 5:1 linear:self-attention; IQ starts slightly more retrieval-heavy for code and requires 7:1 / 6:2 / 5:3 schedule ablations.

Hamiltonian state dynamics are not part of v2. They overlap with the state-dynamics role already owned by Mamba-3 and currently have weaker LLM evidence.

## Research ablations

The following are experiments, not simultaneously enabled defaults:

- Gated DeltaNet / Gated DeltaNet-2 versus Mamba-3;
- log-linear attention versus NSA anchors;
- PaTH attention;
- latent NSA with MLA/GLA cache compression;
- shared + routed MoE;
- Mixture-of-Recursions / learned halting;
- structural code/graph inputs;
- keystone-neuron protection after causal identification.

## JEPA-style latent prediction

`iq_model/latent.py` contains a representation predictor attached to the recurrent reasoning output. It predicts a continuous target representation rather than tokens.

The training experiment, not the model class, defines the target. Candidate targets include future spans, future summaries, later recurrent-pass states, completed-function representations, or code-structure states.

Autoregressive generation remains active until latent prediction demonstrates a measurable benefit.

## Code

```text
iq_model/
├── config.py          # topology + Mamba/NSA settings + controls
├── mixers.py          # Phi GQA / Mamba-3 / GDN / NSA / PaTH adapters
├── components.py      # pre-norm block + dense SwiGLU
├── reasoning.py       # heterogeneous depth-recurrent core
├── latent.py          # JEPA-style future latent predictor
├── keystone.py        # neuron activation instrumentation
├── transfer_layout.py # current dense-teacher depth mapping
└── model.py           # complete language-model backbone
```

## Implementation languages

```text
Mojo / MAX  -> accelerator hot paths and custom tensor kernels
Rust        -> runtime, checkpoints/transfer, tokenizer/data, state ownership, evaluation
Go          -> distributed orchestration, workers, telemetry, services
Python      -> upstream research implementations, training, distillation, notebooks
```

Python is a research compatibility layer, not the intended final systems/runtime language. Proven CUDA/Triton kernels should not be rewritten merely for language purity.

See `IMPLEMENTATION_LANGUAGES.md` for the detailed boundary.

## Transfer strategy

The goal is cross-architecture functional transfer, not tensor transplantation into a Phi-shaped model.

```text
Phi donor
  -> direct copy only where operators are genuinely compatible
  -> operator/activation supervision for Mamba-3 + NSA
  -> recurrent-depth hidden-state alignment
  -> end-to-end LM distillation
  -> continued training
```

A later transfer from a second non-Phi donor is required before claiming donor-independent architecture transfer.

## Validation order

1. keep Phi and GDN controls green;
2. install the pinned Mamba-3 environment;
3. run Mamba-3 MIMO CUDA forward/backward smoke tests;
4. run the recurrent Mamba-3 + NSA composition test;
5. compare 7:1 / 6:2 / 5:3 retrieval-anchor schedules;
6. implement packed-sequence semantics for training;
7. define JEPA target construction;
8. only then begin donor-weight/function transfer.

See `IQ_ARCHITECTURE.md` for the authoritative architecture contract.
