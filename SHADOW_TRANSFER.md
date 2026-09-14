# IQ Functional-Shadow Transfer Ablation

This document describes an **optional transfer/probing experiment**. It is not the canonical
Phi -> IQ transfer path for `architecture/iq-hybrid-v1`.

## Canonical transfer direction

The current target is cross-architecture functional transfer:

```text
Phi-4-mini donor
    |
    +-- directly copy genuinely compatible modules
    |      embeddings / norms / dense SwiGLU / LM head where shapes and semantics match
    |
    +-- operator alignment for incompatible sequence mixers
    |      GQA teacher -> Gated DeltaNet / NSA student
    |
    +-- block hidden-state alignment
    |
    +-- recurrent-depth supervision
    |      shared IQ block j is supervised by teacher depths j across all three passes
    |
    +-- end-to-end logit / language-model distillation
    |
    `-- continued training / evaluation
```

MOHAWK-style matrix/operator alignment and staged block/end-to-end distillation are the primary
mechanisms. There is no assumed universal algebraic map from arbitrary donor weights to the new
hybrid architecture.

## What remains useful from `iq_transfer/`

The existing code is retained because several pieces are still valuable research infrastructure:

- `donor.py` — donor/config/tensor-source contracts;
- `checkpoint.py` — lazy safetensors access;
- `phi4.py` — Phi checkpoint layout inspection;
- `shadows.py` — randomized activation sketches;
- `transport.py` — controlled coordinate-map experiments;
- `scaling.py` — experimental progression metrics.

None of these files should be interpreted as proof that arbitrary architecture weights can be
transported directly.

## Functional shadows

The randomized-shadow code can still test whether compressed teacher measurements are sufficient
supervision compared with storing full activations/operators.

For centered activation matrix `X` and shared random projection `U`:

```text
observable_j = || U_j X_hat ||^2
```

This is a **classical randomized functional sketch**. It is not quantum shadow tomography and no
quantum sample-complexity claim transfers automatically to this setting.

Useful experiment:

```text
ordinary KD
vs
full hidden/operator alignment
vs
compressed functional-shadow alignment
```

Measure final capability retention, teacher-storage cost, transfer compute, and convergence speed.

## Coordinate transport

The existing ridge coordinate-map machinery is valid only as an experimental initialization or
controlled linear-equivalence test. It must not be applied to semantically different nonlinear or
recurrent operators as though their parameters correspond directly.

Direct weight transformation remains appropriate only when the source and target operators are
mathematically equivalent or connected by a known function-preserving reparameterization.

## DoRA and energy alignment

DoRA is optional PEFT and is **not** part of the canonical architecture-transfer pipeline.

Hamiltonian/energy alignment is also not a required transfer stage. Energy/verifier research belongs
to the reasoning/evaluation track and must demonstrate correlation with actual solution quality
before influencing hidden-state dynamics or halting.

## Hybrid-v1 transfer prerequisite

Do not resume donor-weight insertion until:

1. the Phi-control recurrent model is green;
2. Gated DeltaNet + NSA CUDA smoke tests are green;
3. the tiny hybrid recurrent model has stable forward/backward dynamics;
4. the exact operator targets for GQA -> Gated DeltaNet and GQA -> NSA are defined;
5. latent NSA and PaTH are kept out of the first transfer unless their own implementation gates pass.
