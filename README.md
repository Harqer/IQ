# IQ — Hybrid Reasoning LLM

IQ is an experimental language-model architecture for complex coding, long-context retrieval, and adaptive multi-step reasoning.

The production target is **not** the original Neutrino/Ising mock architecture. IQ v2 combines a transferred Transformer backbone with a Mamba-3 recurrent state path, sparse/global attention, adaptive recurrence, and a differentiable Hamiltonian executive controller.

## Architecture

```text
tokens
  ↓
embeddings + token position + reasoning-depth encoding
  ↓
┌──────────────── hybrid inner block ────────────────┐
│ Mamba-3 recurrent state  ||  Transformer attention │
│                              ├─ NSA inner blocks    │
│                              └─ global anchors      │
│                 ↓ bounded residual fusion           │
│                    dense SwiGLU                     │
│              executive-latent injection             │
└─────────────────────────────────────────────────────┘
  ↓
compressed/global memory
  ↓
Differential Attention
  ↓
Hamiltonian/energy executive state
  ↓
adaptive halt / refine
  ↓
continuous concept path
  ↓
Concept Mapper → LM head
```

### Why the hybrid

- **Mamba-3** carries efficient long-horizon recurrent state.
- **Transformer attention** remains the exact content-addressable path for identifiers, code dependencies, and needle-in-context retrieval.
- **Native Sparse Attention** handles most token-level context efficiently.
- **Periodic global attention** preserves high-recall access to the complete context.
- **Differential Attention** is reserved for outer-loop executive reasoning.
- **SwiGLU** is the dense inner FFN; routed heterogeneous MoE is introduced later in the outer path.
- **Hamiltonian/EBM control** replaces the old fake Ising gate with a real scalar-energy recurrent controller.

## Retained NIF ideas, redefined

The useful ideas from the original NIF design are kept only where they have a defensible mathematical role:

| Legacy idea | IQ v2 |
| --- | --- |
| Neutrino oscillation | Mamba-3 complex recurrent state dynamics |
| Ising gate | differentiable port-Hamiltonian / EBM executive controller |
| Riemannian manifold | optional hyperbolic executive/concept geometry |
| Heterogeneous MoE | learned heterogeneous routed SwiGLU experts |
| Muon | real matrix-gradient/update orthogonalization |
| GaLore | optional memory-constrained optimizer mode |
| Muon adapters | DoRA correction around transported weights |
| CUDA-Q | isolated research backend; never required for production training/inference |

IQ does **not** claim that ordinary language-model reasoning requires quantum hardware.

## Weight transfer

`iq_transfer/` is the canonical donor-independent transfer package.

Current flow:

```text
donor checkpoint
  → DonorInspector
  → lazy operator catalog
  → calibration activations
  → functional shadows
  → layer correspondence
  → coordinate maps
  → operator transport
  → DoRA correction
  → IQ adaptation
```

Phi-4 is the proof donor. The transfer engine is intentionally model-independent so larger dense, code-specialized, MoE, and future Mamba-3 donors can be added through inspectors rather than separate graft architectures.

See:

- `SHADOW_TRANSFER.md`
- `IQ_WEIGHT_TRANSFER_IMPLEMENTATION_PLAN.md`
- `IQ_V2_IMPLEMENTATION_PLAN.md`

## Training runtime

Production training target:

- PyTorch autograd
- NVIDIA Megatron-Core for distributed parallelism
- Transformer Engine where numerically validated
- upstream Mamba-3 reference implementation initially
- Mojo custom kernels only after forward/backward/state parity
- MAX for later production inference
- BF16 reference training before FP8/MXFP8 promotion

The legacy Mojo NIF implementation remains only as migration source until each active path has a tested replacement.

## Agent runtime

`iq_harness/` provides the model-independent orchestration layer:

- Agent Skills-compatible loading
- tools and provider boundaries
- manager/subagent delegation
- handoffs
- sessions
- approvals
- guardrails
- tracing

The trainable IQ model will connect through `ModelBackend.generate()`.

## Engineering rules

- no mock tensor paths on the active model
- no constant energies or fake checkpoint loaders
- no placeholder production credentials/resources
- reference implementation before optimized kernels
- deterministic checkpoint/resume
- immutable tokenizer/data/donor manifests
- every trainable parameter belongs to exactly one optimizer group
- completed green work is committed and merged into `main`

## Repository status

Implemented today:

- donor-independent Phi checkpoint inspection
- lazy safetensors operator access
- functional-shadow measurements
- monotonic layer matching
- coordinate-map/operator transport
- scale-gate metrics
- model-independent agent harness

In migration:

- real IQ v2 model/training packages
- Mamba-3 + Transformer hybrid blocks
- real Muon optimizer
- DoRA correction
- Hamiltonian/EBM controller
- adaptive recurrent reasoning
- code-focused FIM/MTP/MoE training

## License

See `LICENSE` and `PROPRIETARY_LICENSE.md` for repository licensing terms.
