# IQ Functional-Shadow Transfer Ablation

This document describes an **optional transfer/probing experiment**. It is not the canonical donor -> IQ transfer path for `architecture/iq-mamba-v2`.

## Canonical v2 transfer direction

IQ v2 deliberately crosses operator families:

```text
dense Transformer donor
    |
    +-- direct copy only where semantics genuinely match
    |      embeddings / compatible norms / dense SwiGLU / LM head
    |
    +-- teacher operator and activation traces
    |
    +-- functional alignment
    |      attention teacher -> Mamba-3 MIMO student
    |      attention teacher -> NSA retrieval anchor
    |
    +-- block hidden-state alignment
    |
    +-- recurrent-depth supervision
    |      shared IQ block j learns from multiple teacher depths
    |
    +-- end-to-end logit / language-model distillation
    |
    `-- continued training / evaluation
```

MOHAWK-style staged operator/block/model alignment is the main precedent. There is no assumed universal algebraic map from Transformer weights into Mamba-3 state-space parameters.

## Transfer success criterion

The research claim is not "the student initializes without crashing." Measure:

```text
transfer advantage = transferred IQ score - scratch IQ score
```

at matched architecture, data, optimization budget and evaluation suite.

Architecture-transfer evidence becomes stronger only if that advantage survives larger architectural changes and later transfers from a second donor family.

## Existing `iq_transfer/` infrastructure

Several utilities remain useful:

- `donor.py` — donor/config/tensor-source contracts;
- `checkpoint.py` — lazy safetensors access;
- `phi4.py` — Phi checkpoint inspection;
- `shadows.py` — randomized activation sketches;
- `transport.py` — controlled coordinate-map experiments;
- `scaling.py` — progression metrics.

The donor interface must eventually be generalized beyond Phi before donor-independent transfer is claimed.

## Functional shadows

For centered activation matrix `X` and shared random projection `U`:

```text
observable_j = || U_j X_hat ||^2
```

This remains useful as a compressed supervision ablation:

```text
ordinary KD
vs
full hidden/operator alignment
vs
compressed functional-shadow alignment
```

It is a classical randomized functional sketch, not quantum shadow tomography.

## Coordinate transport

Ridge coordinate maps may be useful for controlled initialization experiments when source and target representations are approximately linearly related. They must not be interpreted as a direct parameter mapping from Transformer attention into Mamba-3 recurrence.

Direct tensor transformation is valid only for mathematically equivalent/reparameterized operators.

## JEPA latent predictor and transfer

The v2 latent predictor is trained against representation targets defined by the training experiment. It is not initialized by pretending a donor token head is the same operator.

Possible teacher targets include:

- future donor hidden representations;
- future-summary representations;
- later-depth teacher states;
- code-state/structure targets.

Evaluate each target independently.

## Not part of the canonical transfer path

- DoRA/LoRA as mandatory stages;
- Hamiltonian/energy alignment;
- quantum/neutrino transformations;
- renaming ordinary NSA as latent NSA;
- direct tensor mapping into Mamba-3 state parameters without a function-preserving derivation.

## V2 prerequisites before donor insertion

1. Phi recurrent control is green.
2. Hybrid-v1 GDN + NSA control remains reproducible.
3. Mamba-3 MIMO CUDA forward/backward is green.
4. Mamba-3 + NSA repeated-depth composition has finite activations/gradients.
5. packed/variable-length Mamba training semantics are validated.
6. exact teacher targets for attention -> Mamba-3 and attention -> NSA are specified.
7. JEPA target construction is specified separately from LM distillation.
8. only then begin donor transfer.
