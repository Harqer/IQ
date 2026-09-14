# IQ

IQ is an experimental language-model architecture focused on **complex coding and multi-step reasoning**.

The active architecture work is on `architecture/iq-hybrid-v1`. The older `nif_sovereign/` physics-inspired prototype is legacy reference code; `architecture/iq-research-v0` remains the Phi-compatible recurrent-depth control.

## Current architecture

The authoritative design is in [`IQ_ARCHITECTURE.md`](IQ_ARCHITECTURE.md).

Hybrid-v1 keeps the `4 + 8x3 + 4` recurrent-depth topology but replaces the homogeneous middle stack with a heterogeneous sequence-mixing schedule:

```text
embedding
   |
4 unique Phi-compatible prelude blocks
   |
   v
recurrent core, repeated 3 passes
   |
   |-- 0 Gated DeltaNet
   |-- 1 Gated DeltaNet
   |-- 2 Native Sparse Attention
   |-- 3 Gated DeltaNet
   |-- 4 Gated DeltaNet
   |-- 5 Native Sparse Attention
   |-- 6 Gated DeltaNet
   `-- 7 Native Sparse Attention
   |
4 unique Phi-compatible coda blocks
   |
final RMSNorm
   |
LM / optional auxiliary heads
```

Physical depth is 16 blocks. Effective depth remains 32:

```text
4 + (8 x 3) + 4 = 32
```

This preserves the exact Phi-4-mini depth correspondence while giving the recurrent core two complementary sequence mechanisms:

- **Gated DeltaNet** for linear-time recurrent state propagation;
- **Native Sparse Attention** anchors for precise local/global retrieval.

## Position and latent-attention transition

Current executable sparse anchors use NSA's validated RoPE path.

Two research targets are deliberately exposed but not silently approximated:

- **PaTH attention** can be tested as a standalone data-dependent positional-attention operator;
- **latent NSA** (MLA/GLA inside NSA branches) is blocked until its published latent-cache structure is implemented and validated.

The code must fail explicitly rather than relabel ordinary NSA as MLA+NSA or pretend PaTH is a drop-in RoPE replacement inside NSA.

## Feed-forward and keystone neurons

The first hybrid transition retains dense fused SwiGLU so attention/recurrent changes remain attributable.

FFN activations expose a zero-impact observer hook. `iq_model/keystone.py` records cross-task activation strength for later keystone-neuron identification. It does not predeclare, freeze, or protect neurons before the published identification/ablation protocol is reproduced.

Shared + routed MoE is the next FFN transition after the hybrid mixer core is stable.

## Code

```text
iq_model/
├── config.py          # topology + hybrid schedule + kernel settings
├── mixers.py          # Phi GQA / Gated DeltaNet / NSA / PaTH adapters
├── components.py      # pre-norm block + dense SwiGLU + activation hook
├── reasoning.py       # heterogeneous depth-recurrent core
├── keystone.py        # cross-task neuron activation instrumentation
├── transfer_layout.py # exact dense-teacher -> recurrent-depth mapping
└── model.py           # full language-model backbone
```

Tests:

```text
tests/test_iq_model.py       # CPU/control architecture invariants
tests/test_hybrid_mixers.py  # conditional CUDA/FLA smoke tests
```

## Reproducible environment

Base research versions are pinned in `requirements-research.txt`.
Hybrid kernels are pinned to a reviewed Flash Linear Attention commit in `requirements-hybrid.txt`.

Open the GPU validation notebook:

```text
https://colab.research.google.com/github/Harqer/IQ/blob/architecture/iq-hybrid-v1/experiments/architecture_v1/colab.ipynb
```

The notebook does **not** download Phi weights. It validates the architecture, reads the real Phi-4-mini configuration, and runs small CUDA smoke tests for Gated DeltaNet and NSA.

## Transfer order

Do not insert Phi weights into the legacy NIF architecture.

The current order is:

1. validate the Phi-control recurrent backbone;
2. validate the hybrid Gated DeltaNet + NSA physical stack;
3. establish stable tiny forward/backward behavior;
4. define Phi -> hybrid MOHAWK/operator-distillation targets;
5. transfer/distill the hybrid core;
6. implement latent NSA as its own gated transition;
7. evaluate PaTH integration separately;
8. then add latent reasoning, adaptive depth, and shared+routed MoE.

MOHAWK is a transfer procedure, not a decoder component. DoRA/LoRA are optional PEFT methods; Muon/AdamW are optimizers; GaLore is optimizer-memory reduction; Colab/Hugging Face/QPU services are infrastructure.
