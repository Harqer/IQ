# IQ Shadow Transport

Canonical architecture for transferring learned operators from pretrained donor LLMs into IQ.

## Goal

Phi-4 is the proof-of-mechanism donor, not a permanent dependency. The transport engine must remain donor-independent so later experiments can move to larger Qwen/Llama/MoE checkpoints without rewriting IQ.

```text
Donor checkpoint
    -> DonorInspector
    -> lazy OperatorRef catalog
    -> paired calibration activations
    -> FunctionalShadow observables
    -> monotonic layer correspondence
    -> source->IQ coordinate maps
    -> transported Q/K/V/O + MLP operators
    -> DoRA correction
    -> optional reasoning-EBM calibration
    -> IQ adaptation/evaluation
```

## Current implementation

`iq_transfer/` is the source of truth.

- `donor.py` - donor/config/tensor-source contracts.
- `checkpoint.py` - lazy local safetensors access; large checkpoints are not loaded wholesale.
- `phi4.py` - Phi-4/Phi-4-mini inspector for the `Phi3ForCausalLM` fused QKV and gated-MLP layout.
- `shadows.py` - architecture-independent randomized quadratic observables over paired activations and monotonic layer matching.
- `transport.py` - sample-space ridge coordinate fitting and linear-operator transport.
- `scaling.py` - objective gate for deciding whether to progress to a larger donor.

The old Gemma-specific mock graft is intentionally removed. New donors implement `DonorInspector`; they do not get separate graft architectures.

## Shadow definition

For centered calibration activations `X` with Frobenius-normalized `X_hat`, use the same random measurement matrix `U` across donor and IQ:

```text
observable_j = || U_j X_hat ||^2
```

This measures quadratic forms of the sample Gram operator and is comparable even when donor and IQ hidden widths differ. It is the classical randomized-shadow baseline. A quantum-shadow backend can later replace the measurement engine without changing the transport interface.

## Coordinate transport

Fit paired coordinates with the sample-space ridge solution:

```text
X_iq ~= X_donor P
P = X_donor^T (X_donor X_donor^T + lambda I)^-1 X_iq
```

For source linear operator `W_s`, source->target input map `P_in`, and output map `P_out`:

```text
W_iq = P_out^T W_s pinv(P_in)^T
```

Q, K, V, O, gate, up, and down projections are transported independently. Fused donor tensors are represented as lazy row slices and are only materialized when needed.

## Phi proof

Use a local official Phi checkpoint directory containing `config.json` and safetensors files. The inspector reads checkpoint metadata rather than assuming one Phi size.

The first experiment should keep the tokenizer/calibration prompts fixed and compare:

1. randomly initialized IQ,
2. IQ + ordinary distillation,
3. IQ + full-activation transport,
4. IQ + functional-shadow transport.

Do not claim success from training loss alone. Record held-out language/reasoning scores, shadow error, coordinate-map error, adaptation compute, and scratch baseline.

## Scale gate

Only advance donor size when the configured gate passes. Initial defaults are experimental, not scientific constants:

```text
retention = IQ_score / donor_score >= 0.80
IQ_score >= scratch_score
adaptation_compute / scratch_compute <= 0.50
```

After Phi establishes a real baseline, tune these thresholds from evidence.

## IQ v2 multi-donor integration

Detailed production sequencing, parameter provenance, Transformer -> Mamba-3 bootstrap, cross-tokenizer alignment, and Mellum2 MoE transfer are specified in `IQ_WEIGHT_TRANSFER_IMPLEMENTATION_PLAN.md`.

The current ownership policy is:

```text
Phi-4 -> dense Transformer base
Transformer projections -> Mamba-3 x/B/C/out bootstrap
Mellum2 -> code-specialized DoRA + MoE + MTP
IQ-native -> reasoning EBM critic, halting, spectral depth, concept mapper
```

Mamba-3-specific recurrence parameters without a justified donor correspondence use the official initialization and are learned. Do not invent mappings for recurrence-only fields.

## Scaling path

Keep the IQ recipient architecture fixed first:

```text
Phi-4 family -> mid-size dense donor -> 30-70B donor -> 100B+ donor -> frontier MoE
```

Each new checkpoint needs only a donor inspector plus checkpoint-layout tests. The shadow, correspondence, transport, DoRA, energy, and evaluation pipeline stays shared.

## Tests

```bash
pip install -r requirements-transfer.txt
python -m unittest discover -s tests -p 'test_transfer.py' -v
```

The unit tests verify fused-operator parsing, checkpoint shape validation, cross-width shadows, order-preserving layer matching, coordinate-map recovery, exact controlled linear transport, and scale-gate behavior.
