# IQ

IQ is an experimental language-model architecture focused on **complex coding and multi-step reasoning**.

The repository is being migrated away from the older `nif_sovereign/` physics-inspired prototype toward a research-grounded recurrent architecture that can inherit knowledge from a pretrained Phi-4-mini donor and then evolve through controlled ablations.

## Current architecture

The authoritative design is in [`IQ_ARCHITECTURE.md`](IQ_ARCHITECTURE.md).

The first research-valid backbone is:

```text
embedding + Phi-compatible RoPE
        |
4 unique prelude blocks
        |
8 physical recurrent blocks
executed for 3 passes
        |
4 unique coda blocks
        |
final RMSNorm
        |
LM / optional auxiliary heads
```

Default effective depth:

```text
4 + (8 x 3) + 4 = 32
```

This intentionally matches Phi-4-mini's 32-layer depth while using only 16 physical decoder blocks.

### v0 components

- Phi-compatible tokenizer/vocabulary during transfer
- Phi-compatible GQA initially
- RMSNorm
- fused SwiGLU / gated SiLU FFN
- fixed recurrent depth (`R=3`)
- zero-initialized pass embeddings
- optional MTP/verifier extension points
- PyTorch reference implementation for Colab training and transfer

The recurrent topology, mixer, and FFN are separate modules so later experiments can independently test Differential Attention, native sparse attention, linear/recurrent mixers, MoE routing, latent reasoning, learned halting, and Hamiltonian-style update rules.

## Code

```text
iq_model/
├── config.py       # topology + exact donor-depth mapping
├── components.py   # pluggable block + Phi-compatible SwiGLU
├── reasoning.py    # recurrent reasoning core
└── model.py        # complete language-model backbone
```

Tests live in `tests/test_iq_model.py`.

## Legacy code

`nif_sovereign/`, CUDA-Q configuration, neutrino oscillation blocks, Ising gates, manually named expert types, and the older physics feed-forward path are **legacy experimental code**. They are retained for reference until the new architecture is validated, but they are not the target for new weight transfer.

## Transfer strategy

Do not insert Phi weights into the legacy architecture.

The planned order is:

1. validate the new recurrent backbone;
2. instantiate it with the real Phi-4-mini config;
3. verify all compatible parameter shapes;
4. map 32 teacher depths onto `4 + 8x3 + 4` effective student depths;
5. transfer unique compatible modules directly;
6. distill the shared recurrent core against its three corresponding donor depths;
7. only after the recurrent model is stable, replace individual mixers/FFNs in controlled experiments.

MOHAWK is a **transfer procedure**, not a model layer. DoRA/LoRA are optional PEFT methods, Muon/AdamW are optimizers, GaLore is an optimizer-memory technique, and Colab/Hugging Face/QPU services are infrastructure rather than architecture.

## Compute

The trainable reference model is PyTorch-first because the available remote path is Google Colab / hosted GPU inference. Mojo can be used later for optimized kernels after the architecture has been validated.

No local GPU is required for the intended workflow.
