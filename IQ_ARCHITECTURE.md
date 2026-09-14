# IQ Research Architecture

This document is the authoritative architecture target for the current IQ research branch.
The older `nif_sovereign/` implementation predates the 2026 architecture research and is
**legacy reference code**, not the target for new weight transfer.

## Design rule

Keep three concerns separate:

1. **Architecture** — the function IQ computes.
2. **Training/optimization** — how IQ learns that function.
3. **Transfer/deployment** — how pretrained knowledge and runtime infrastructure are supplied.

MOHAWK, DoRA, Muon, GaLore, quantization, Colab, Hugging Face inference, and QPU access are
not core model layers.

## v0 architecture

The first model changes one major structural property relative to Phi-4-mini: **depth is
recurrent**. Tokenization, embedding width, GQA, RoPE, RMSNorm, and the gated FFN remain
Phi-compatible so transfer can be measured cleanly.

```text
input tokens
    |
    v
embedding + Phi-compatible RoPE
    |
    v
4 unique prelude blocks
    |
    v
8 physical recurrent blocks  <-----------------+
    |                                           |
    +-- pass 0 ---------------------------------+
    +-- pass 1 ---------------------------------+
    +-- pass 2 ---------------------------------+
    |
    v
4 unique coda blocks
    |
    v
final RMSNorm
    |
    +--> LM head
    +--> optional MTP heads
    +--> optional verifier head
```

Default topology:

| Property | v0 value |
|---|---:|
| Physical decoder blocks | 16 |
| Prelude blocks | 4 |
| Recurrent physical blocks | 8 |
| Recurrent passes | 3 |
| Coda blocks | 4 |
| Effective depth | 32 |
| Hidden width | donor config (Phi-4-mini: 3072) |
| FFN width | donor config (Phi-4-mini: 8192) |
| Mixer | Phi-compatible GQA initially |
| FFN | fused SwiGLU / gated SiLU |
| Norm | pre-RMSNorm |
| Position | donor Phi RoPE/LongRoPE initially |
| Halting | disabled in v0 |
| Latent workspace | disabled in v0 |
| MoE | disabled in v0 |
| Energy descent | disabled in v0 |

## Exact donor-depth correspondence

For the default `4 + 8 x 3 + 4` topology:

```text
prelude[0..3]              -> teacher layers 0..3
core pass 0, blocks 0..7  -> teacher layers 4..11
core pass 1, blocks 0..7  -> teacher layers 12..19
core pass 2, blocks 0..7  -> teacher layers 20..27
coda[0..3]                 -> teacher layers 28..31
```

This mapping is defined in `iq_model/config.py` and is the later dense-to-recurrent
transfer target. It allows each shared core block to be supervised against three donor
depths instead of inventing a layer-folding map after training starts.

## Block definition

Every physical v0 block is explicit and pluggable:

```text
x
|
RMSNorm
|
mixer
|
+ residual
|
RMSNorm
|
feed-forward
|
+ residual
```

`iq_model/components.py` currently provides:

- Phi-compatible GQA mixer from the canonical Transformers implementation;
- `PhiCompatibleSwiGLU`, with fused `gate_up_proj` and `down_proj` parameters matching
  the donor layout;
- `IQBlock`, which allows mixer and FFN families to be replaced independently.

## Recurrent reasoning core

`iq_model/reasoning.py` owns reasoning depth. The same eight block objects are executed
for multiple passes. They are **not cloned per pass**.

The recurrent state update is currently:

```text
candidate = block(h_r)
h_{r+1} = h_r + alpha * (candidate - h_r)
```

with `alpha = 1` by default. This preserves the transplanted block function for the first
experiment. Recurrence-aware residual scaling is an explicit ablation rather than an
untracked architecture change.

A learned pass embedding is available, but initialized to exactly zero. It therefore adds
no behavior before training and later allows the shared core to distinguish recurrence
visits.

## Feed-forward network

The old NIF feed-forward path (`GELU -> neutrino oscillation -> Ising gate`) is not part of
the research backbone.

v0 uses a fused gated SiLU/SwiGLU-style FFN:

```text
gate, up = gate_up_proj(x).chunk(2)
y = down_proj(SiLU(gate) * up)
```

This is deliberately transfer-compatible with Phi-4-mini.

### Later FFN experiments

Only after recurrent transfer is stable:

1. fine-grained MoE / shared + routed experts;
2. differentiable routing (ReMoE-style);
3. recurrent-depth routing jointly with expert routing;
4. tensor-factorized FFN/operator experiments.

Do not add manually named "linguistic", "physics", or "diffusion" experts to the v0
backbone.

## Mixer roadmap

The v0 mixer is intentionally conservative. Later controlled replacements:

1. Differential Attention;
2. Native/DeepSeek-style sparse attention;
3. linear recurrent mixer (KDA/DeltaNet-family experiment);
4. hybrid linear-time layers plus sparse exact-attention correction layers;
5. Differential + sparse only if each component wins independently.

Changing recurrence and the sequence mixer in the same first experiment is prohibited
because a failed transfer would be uninterpretable.

## Reasoning roadmap

The research sequence is:

1. fixed recurrent depth (`R=3`);
2. recurrence stability/residual-scaling ablations;
3. latent workspace / looped latent reasoning;
4. token-dependent recursion (Mixture-of-Recursions-style);
5. learned halting/adaptive computation;
6. auxiliary energy/verifier calibration;
7. Hamiltonian/conservative-dissipative recurrence only as a controlled alternative
   update rule, not as an assumption of quantum computation.

## Auxiliary heads

The code exposes optional MTP and verifier heads, but both are disabled by default.

The verifier/energy scalar is **auxiliary**. It does not alter hidden states and does not
control halting until calibration demonstrates that its score tracks actual solution
quality.

Planned objective experiments after the backbone is stable:

- next-token prediction;
- multi-token prediction;
- FIM / structural FIM;
- horizon-length prediction for infilling;
- future-summary prediction;
- latent-reasoning supervision.

## Not architecture

The following should not appear as mandatory decoder components:

| Item | Correct role |
|---|---|
| MOHAWK | dense-to-new-architecture transfer procedure |
| DoRA / LoRA | optional parameter-efficient fine-tuning |
| Muon / AdamW | optimizer |
| GaLore | optimizer-memory reduction |
| quantization | training/deployment optimization |
| classical shadows | optional transfer/probing research |
| Colab / remote GPU | compute infrastructure |
| QPU | optional isolated quantum experiment, not LLM tensor runtime |

## Current implementation

- `iq_model/config.py` — topology and exact teacher-depth mapping.
- `iq_model/components.py` — pluggable decoder block and explicit SwiGLU FFN.
- `iq_model/reasoning.py` — fixed-depth recurrent reasoning subsystem.
- `iq_model/model.py` — complete embedding -> prelude -> recurrent core -> coda -> heads model.
- `tests/test_iq_model.py` — shape, sharing, mapping, and transfer-safety invariants.

## Next implementation gate

Before any Phi weights are inserted:

1. install the current pinned Transformers/PyTorch environment in Colab;
2. run the tiny architecture tests;
3. instantiate the real Phi-4-mini config without loading weights;
4. verify parameter names/shapes for embeddings, norms, GQA, SwiGLU, and LM head;
5. verify the recurrent model forward pass at small sequence length;
6. only then implement dense-to-recurrent weight initialization/distillation.
