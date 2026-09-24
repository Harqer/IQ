# IQ — Hybrid Reasoning LLM

IQ is an experimental language-model architecture for complex coding, long-context retrieval, and adaptive multi-step reasoning.

The production target is **not** the original Neutrino/Ising mock architecture. IQ v2 uses Mamba-3 MIMO as the recurrent reasoning/state-evolution path, with Transformer attention retained as a routed context/comparison service, plus adaptive recurrence and a differentiable Hamiltonian executive controller.

## Architecture

```text
tokens
  ↓
embeddings
  ↓
explicit heterogeneous schedule
  │
  ├─ M = Mamba-3 MIMO
  │      exponential-trapezoidal recurrence
  │      complex/data-dependent state rotation
  │      rank-4 MIMO state evolution
  │
  ├─ E = routed/shared SwiGLU MoE
  │
  ├─ C = CSA compressed context attention
  │      Q/KV RMSNorm + trailing partial RoPE
  │
  ├─ H = HCA heavily compressed context attention
  │
  └─ A = dense context anchor
         QK RMSNorm + RoPE
         exact pairwise/few-shot comparison
  ↓
reasoning recurrence + executive state
  ↓
Hamiltonian/energy control
  ↓
adaptive halt/refine
  ↓
Concept Mapper
  ↓
MTP + LM head
```

### Why the hybrid

- **Mamba-3 MIMO (rank 4)** is the primary recurrent state/reasoning path; MIMO is not optional and IQ has no silent SISO fallback.
- **Transformer attention** remains the exact content-addressable context/comparison path for identifiers, code dependencies, few-shot induction, and needle-in-context retrieval.
- **DeepSeek V4-family CSA/HCA compressed context attention** handles long-range context: a local sliding window plus compressed KV memory with learned sparse indexing; dense attention is reserved for strict pairwise/few-shot comparison.
- **Dense attention anchors** use per-head Q/K RMSNorm followed by RoPE for exact comparison/induction tasks.
- **CSA/HCA** use normalized compressed KV states, trailing partial RoPE, and inverse output rotation; Mamba-3 keeps its own native state rotation internally.
- **SwiGLU MoE** is a separate scheduled expert-compute layer, not an automatic FFN attached to every Mamba layer.
- **mHC** is the target residual-topology experiment after exact reference parity; the Phi control keeps ordinary residuals.
- **MTP** is a first-class pretraining objective/head stack rather than an after-the-fact probe.
- **Differential Attention** is reserved for outer-loop executive reasoning.
- **Hamiltonian/EBM control** replaces the old fake Ising gate with a real scalar-energy executive controller using PSD-preserving dissipation and energy-aware discrete integration.

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

- dense Phi-compatible Transformer control/teacher with packed/padded GQA
- donor-independent Phi checkpoint inspection and executable transfer job
- lazy safetensors operator access
- functional-shadow measurements and monotonic layer matching
- coordinate-map/operator transport + DoRA correction
- mandatory rank-4 Mamba-3 MIMO runtime wrapper and H200 parity gate
- explicit heterogeneous `HybridSchedule` validation
- executable heterogeneous Mamba-3 MIMO / routed-MoE / dense-attention backbone for implemented layer types
- packed/padded repository batches converted to Mamba-3 varlen `cu_seqlens` with document-state isolation
- routed/shared SwiGLU MoE `E` layer with top-k routing, load-balance/z losses, and no silent token drop
- sequential shared-embedding/shared-head MTP prediction stack with packed-document-safe causal chaining
- pretraining wrapper combining NTP/MTP/MoE auxiliary objectives with checkpointed coefficients
- real-batch H200 hybrid forward/backward and cross-document-isolation gate
- QK-normalized RoPE dense-context anchor
- DeepSeek V4-style CSA/HCA full-sequence PyTorch reference with sliding-window branch, learned Lightning indexer, shared K=V MQA, sinks, grouped low-rank output, and packed-document isolation
- leading RoPE plus interleaved trailing partial/inverse RoPE reference primitives
- scale-gate metrics
- model-independent agent harness

Specified / next runtime integration:

- complete heterogeneous schedule once CSA/HCA and executive layer runtimes are implemented
- CSA/HCA compression, learned indexer, shared compressed KV, sinks, grouped output projection
- exact mHC residual-topology reference + optimized kernel
- Hamiltonian/EBM controller with PSD ring dissipation and energy-controlled discrete integration
- adaptive recurrent reasoning and concept collapse
- real Muon optimizer
- code-focused FIM/MTP/MoE training

## License

See `LICENSE` and `PROPRIETARY_LICENSE.md` for repository licensing terms.
