# IQ — Hybrid Reasoning LLM

IQ is an experimental language-model architecture for complex coding, long-context retrieval, and adaptive multi-step reasoning.

The production target is **not** the original Neutrino/Ising mock architecture. IQ v2 uses Mamba-3 MIMO as the dominant token-time sequence/state mixer, DeepSeek V4-family CSA/HCA for long-context addressing, Stable LatentMoE for routed expert computation, Block AttnRes for cross-depth retrieval, and mHC as an orthogonal residual-topology experiment. Reasoning-time recurrence, the EBM critic, and adaptive halting are separate from the physical backbone schedule.

## Architecture

```text
tokens
  ↓
embeddings
  ↓
heterogeneous backbone
  │
  ├─ M = Mamba-3 MIMO
  │      token-time recurrent sequence/state mixing
  │
  ├─ E = Stable LatentMoE + SiTU-GLU
  │
  ├─ C/H = CSA/HCA + sliding-window context
  │
  ├─ Block AttnRes = cross-depth residual retrieval
  │
  └─ mHC = within-depth residual-stream topology experiment
  ↓
reasoning-time latent recurrence
  ↕
optional EBM critic
  │   candidate scoring / branch ranking / verification
  │   optional evidence for halting
  ↓
adaptive halting
  ↓
Concept Mapper (ablation)
  ↓
MTP + LM head

Optional dense attention remains a teacher/control/fallback path for exact
pairwise comparisons; it is not a mandatory backbone anchor.
```

### Why the hybrid

- **Mamba-3 MIMO (rank 4)** is the primary token-time recurrent sequence/state mixer; MIMO is not optional and IQ has no silent SISO fallback.
- **DeepSeek V4-family CSA/HCA** is the production long-context system: sliding-window local attention plus compressed long-range memory and learned sparse selection. Dense attention remains an optional teacher/control/fallback for exact pairwise comparison rather than a required anchor.
- **Stable LatentMoE** is the production expert path: full-width sigmoid routing, latent routed experts, post-aggregate RMSNorm, full-width shared experts, SiTU-GLU, and one-step-delayed Quantile Balancing. The older full-width SwiGLU MoE remains an ablation/control.
- **Block AttnRes** provides content-dependent retrieval across completed depth blocks; **mHC** is an orthogonal within-depth residual-stream topology experiment. Neither is a reasoning controller.
- **Reasoning-time recurrence** is separate from Mamba's token-time recurrence and owns iterative latent refinement. The reference path uses a gated residual transition with a continuous spectral depth coordinate. Teacher-forced training requires explicit `reasoning_context_lengths` so the recurrent state only observes the causal prompt prefix; inference uses the full visible prefix and injects reasoning only at the final prediction source position.
- **ReasoningEnergyCritic** is an optional EBM scorer over the generated reasoning trajectory. It can provide ranking/verification signals and halting evidence, while the recurrence transition remains identical with the critic enabled or disabled.
- **Adaptive halting** is a separate learned head. Training uses differentiable stop/survival weights across the bounded reasoning trajectory; inference can exit early only when halt probability and state-convergence criteria pass. Optional energy stabilization adds evidence rather than replacing those criteria.
- **MTP** is a first-class pretraining objective/head stack rather than an after-the-fact probe.
- **Differential Attention**, Coconut-style recurrence, and the Concept Mapper remain ablation-controlled reasoning experiments rather than mandatory backbone stages.

## Legacy NIF disposition

The useful ideas from the original NIF design are kept only where they have a defensible mathematical role:

| Legacy idea | IQ v2 disposition |
| --- | --- |
| Neutrino oscillation | superseded by Mamba-3 recurrent state dynamics |
| Ising / Hamiltonian gate | retired from the canonical architecture |
| Scalar energy | retained only as an optional learned EBM critic over generated reasoning states |
| Riemannian manifold | isolated research experiment; not part of the canonical backbone or controller |
| Heterogeneous MoE | Stable LatentMoE + SiTU-GLU, with full-width SwiGLU retained as a control |
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
- Stable LatentMoE/SiTU-GLU reference with full-width sigmoid routing, latent routed experts, full-width shared experts, delayed Quantile Balancing, plus the original routed SwiGLU control
- sequential shared-embedding/shared-head MTP prediction stack with packed-document-safe causal chaining
- pretraining wrapper combining NTP/MTP/MoE auxiliary objectives with checkpointed coefficients
- real-batch H200 hybrid forward/backward and cross-document-isolation gate
- QK-normalized RoPE dense-context anchor
- DeepSeek V4-style CSA/HCA full-sequence PyTorch reference with sliding-window branch, learned Lightning indexer, shared K=V MQA, sinks, grouped low-rank output, and packed-document isolation
- leading RoPE plus interleaved trailing partial/inverse RoPE reference primitives
- scale-gate metrics
- adaptive reasoning-time recurrence with spectral depth encoding, differentiable training-time halting, hard inference early exit, and packed-document isolation
- optional EBM energy/energy-delta features in the halting path with a tested invariant that the critic does not alter the generated state trajectory
- near-zero gated injection of final reasoning state back into token representations
- model-independent agent harness

Specified / next runtime integration:

- wire Block AttnRes and validated mHC behavior into the complete heterogeneous runtime without changing their distinct responsibilities
- finish CSA/HCA runtime optimization and parity work; keep dense attention as an optional teacher/control rather than a mandatory anchor
- train/evaluate recurrence + halting on reasoning/code tasks and calibrate halt thresholds against quality/compute
- construct successful/failed/corrupted reasoning-state pairs and train the EBM ranking objective before enabling energy-assisted halting by default
- concept-mapper and continuous-thought ablations
- real Muon optimizer
- code-focused FIM/MTP/MoE training

## License

See `LICENSE` and `PROPRIETARY_LICENSE.md` for repository licensing terms.
