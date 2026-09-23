# IQ — Hybrid Reasoning LLM

IQ is an experimental language-model architecture for complex coding, long-context retrieval, and adaptive multi-step reasoning.

The production target is **not** the original Neutrino/Ising mock architecture. IQ v2 uses Mamba-3 MIMO as the recurrent reasoning/state-evolution path, with Transformer attention retained as a routed context/comparison service, plus adaptive recurrence and a differentiable Hamiltonian executive controller.

## Architecture

```text
tokens
  ↓
embeddings + token position + reasoning-depth encoding
  ↓
context state ───────────────► Context Router
  │                            OFF / NSA / DENSE
  │                                  │
  │                          retrieved/comparison
  │                               context
  └──────────────────────┬───────────┘
                         ↓
                  Mamba-3 MIMO
                recurrent reasoning
                 / state evolution
                         ↓
                      SwiGLU
                         ↓
                executive-latent input
                         ↓
              Hamiltonian/energy control
                         ↓
                 adaptive halt/refine
                         ↓
                  Concept Mapper
                         ↓
                      LM head
```

### Why the hybrid

- **Mamba-3 MIMO (rank 4)** is the primary recurrent state/reasoning path; MIMO is not optional and IQ has no silent SISO fallback.
- **Transformer attention** remains the exact content-addressable context/comparison path for identifiers, code dependencies, few-shot induction, and needle-in-context retrieval.
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

Install the model runtime with:

```bash
bash scripts/install_model_runtime.sh
```

This installs Torch first, then builds the exact pinned Mamba-3 source revision with `--no-build-isolation`, and verifies that the CUDA MIMO kernel is available. The verifier fails rather than switching IQ to SISO.

Before promoting Mamba-3 inference on the H200 training/decode target, run:

```bash
python scripts/verify_mamba3_mimo_h200.py
```

That gate exercises the full 4096-wide rank-4 BF16 MIMO block, backward gradients, pure recurrent decode, and mixed prefill+decode parity. A failure blocks deployment; it does not change the architecture to SISO.

- PyTorch autograd
- NVIDIA Megatron-Core for distributed parallelism
- Transformer Engine where numerically validated
- pinned upstream Mamba-3 MIMO implementation; missing MIMO/TileLang capability is a hard startup error
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
- Mamba-3 MIMO reasoning blocks + routed Transformer context service
- real Muon optimizer
- DoRA correction
- Hamiltonian/EBM controller
- adaptive recurrent reasoning
- code-focused FIM/MTP/MoE training

## License

See `LICENSE` and `PROPRIETARY_LICENSE.md` for repository licensing terms.
