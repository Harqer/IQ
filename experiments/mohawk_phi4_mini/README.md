# MOHAWK: Phi-4-mini -> IQ

This experiment tests one claim only:

> Can an IQ linear-time sequence mixer inherit the token-mixing behavior of a pretrained Phi-4-mini attention layer through MOHAWK matrix orientation?

It is intentionally **not** a full IQ conversion yet.  Changing attention, FFN, recurrence, depth, tokenizer, and hidden width simultaneously would make a failed transfer uninterpretable.

## Donor

Default donor: `microsoft/Phi-4-mini-instruct`.

Expected architecture:

- 32 decoder layers
- hidden size 3072
- intermediate size 8192
- 24 query heads
- 8 KV heads
- head dimension 128
- fused Phi3-style QKV projection

The script reads these values from the checkpoint config instead of hard-coding them.

## Target for experiment 1

The student keeps the donor dimensionality and directly copies Q/K/V/O where those parameters are mathematically compatible.  The only changed mechanism is the mixer:

```text
Phi-4-mini:  softmax(QK^T / sqrt(d))
                       |
                       v
MOHAWK Stage 1
                       |
                       v
IQ: normalized causal phi(Q) phi(K)^T, phi(x)=ELU(x)+1
```

The training implementation materializes the full `T x T` student matrix only for Stage 1.  A later production kernel should execute the same linear-attention operator with a recurrent/prefix-scan formulation.

## Why this comes before the full IQ architecture

MOHAWK's published result succeeds by changing the sequence mixer while retaining compatible high-information components, then increasing the distillation scope in three stages:

1. mixer-matrix orientation;
2. block hidden-state alignment;
3. end-to-end knowledge distillation.

For IQ we will follow the same discipline.  Differential/sparse attention, Hamiltonian recurrence, routed FFNs, learned halting, and other IQ experiments are introduced only after the previous transfer gate passes.

## Corpus policy

The smoke test does **not** download a generic web or synthetic dataset.  By default it consumes local `.py` and `.mojo` files under `--corpus-root` and deterministically holds out 20% by file path.

For a serious transfer run, replace the smoke corpus with a reviewed manifest of permissively licensed, high-quality source repositories and reasoning material.  Do not train on HumanEval, LiveCodeBench, SWE-bench, or other benchmarks reserved for evaluation.

## Run Stage 1

From the repository root on a CUDA machine:

```bash
python -m experiments.mohawk_phi4_mini.stage1 \
  --teacher microsoft/Phi-4-mini-instruct \
  --corpus-root . \
  --layer 15 \
  --seq-len 256 \
  --train-steps 64
```

The experiment records:

- initial held-out normalized Frobenius error;
- final held-out normalized Frobenius error;
- relative held-out improvement;
- the exact files used for train/evaluation;
- the student mixer checkpoint.

Artifacts are written under `artifacts/` and are ignored as experimental output rather than source.

## Gate before Stage 2

Do not proceed because training loss merely decreased.  Stage 1 should demonstrate all of the following on held-out files:

- stable finite loss;
- a meaningful reduction in normalized Frobenius error;
- improvement across multiple middle and late layers rather than one cherry-picked layer;
- no dependence on a benchmark evaluation set;
- reproducibility over at least three seeds.

Only after that result should Stage 2 wrap the mixer in a Phi-compatible block and optimize block-output alignment.  Stage 3 then copies the compatible embeddings/MLPs/norm/head and performs end-to-end logit distillation.

## Controls

The serious experiment should compare:

1. random IQ mixer initialization;
2. copied Phi Q/K initialization without Stage 1;
3. MOHAWK Stage 1 from copied Q/K;
4. an official Mamba-2/SSD mixer as a positive cross-architecture control.

This separates gains from simple parameter copying from gains caused by matrix orientation.

## Primary references

- Bick et al., *Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models*, arXiv:2408.10189.
- Official MOHAWK/Phi-Mamba implementation: `goombalab/phi-mamba`.
- Microsoft Phi-4-mini technical report/model card.
