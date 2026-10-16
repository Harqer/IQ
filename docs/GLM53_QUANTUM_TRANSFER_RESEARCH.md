# Quantum–classical GLM-5.3 → IQ alignment: evidence and executable experiment

Status (2026-10-16): **research candidate; no model-transfer quantum advantage demonstrated.** The module is an opt-in experiment, never a silent replacement of IQ's current weight-transfer pathway. `quantum_router_alignment.py` plus `benchmark_quantum_router_alignment.py` are executable using existing full-width weight/calibration tensors. They do not create, train, or modify a model checkpoint.

## Actual integration surface

Use `feat/iq-glm53-78-full-experts`, **not** the older `main`/24-stage, 32-expert transfer configuration, as the current experimental baseline. `iq_transfer/glm53_parallel_contract.py` specifies 78 GLM decoder blocks, 256 routed experts per MoE stage, top-8, donor residual width 6144; `nvidia_parallel_78_config()` preserves 78 parallel IQ blocks and 256 IQ experts. `iq_transfer/warm.py` performs classical orthogonal remapping via a full feature-space Gram matrix eigendecomposition, and `iq_transfer/glm53_calibration.py` fits stage coordinates and activation correspondences. `iq_transfer/glm53_moe.py` already handles donor MoE transformations. Do not replace these proven classical operations without a matched benchmark.

The actual Hugging Face donor **weight shards were not accessible** to this research environment. The source topology is grounded in the publisher's pinned config and IQ repository; **numeric results below are from explicitly synthetic full-256 fixture tensors, not real GLM or IQ checkpoints**. Prior Colab/Kaggle notebooks record `hf://buckets/Visionboxx/macha`, but that does not establish that the bucket currently contains complete verified data.

## Research comparison (method → equation → input/access → disposition)

| Algorithm (original source) | Mathematically relevant operation | Demonstrated result / available implementation | Applicability and binding limitation |
| --- | --- | --- | --- |
| QSVT — Gilyén et al. (STOC 2019), https://arxiv.org/abs/1806.01838 | Given efficient block encoding of matrix `A`, implement `P(Σ)` in its singular-vector frame; potential ridge pseudoinverse and WARM covariance spectral filtering. | Theoretical block-encoding algorithm and public reference implementations; no GLM-to-IQ measurements. | Need efficient block encoding of actual 6144×6144 Gram, coherent state input, and typically a *classical 6144×4096 basis output*. State loading and tomography/readout remove plausible present-day speedup; **not implemented as a production transfer**. |
| Quantum PCA — Lloyd, Mohseni, Rebentrost (Nature Physics 2014), https://arxiv.org/abs/1307.0401 | Approximate principal eigenspaces of density matrix ρ ∝ WᵀW. | Mathematical qPCA, requires many quantum state copies / density matrix access. | Quantum eigenstates ≠ the classical dense WARM coordinate map. Not presently interchangeable with `torch.linalg.eigh`. |
| Quantum linear solver / HHL — Harrow et al. (PRL 2009), https://doi.org/10.1103/PhysRevLett.103.150502 | Ridge / least squares `x=(AᵀA+λI)⁻¹ Aᵀb`. | Quantum-state solution for restricted data/condition-number assumptions. | IQ needs full dense transformation matrices; state preparation + readout dominate without structured oracle. Not implemented. |
| QAOA — Farhi et al. (2014), https://arxiv.org/abs/1411.4028; warm-start — Egger et al. (2021), https://arxiv.org/abs/2009.10095 | Optimize binary choices via `|ψ_p>=∏ exp(-iβ∑X)exp(-iγH_C)|+…+>`, with cost derived from real router geometry. | IBM Qiskit tutorial: https://qiskit.qotlabs.org/docs/tutorials/warm-start-qaoa . Full 256×256 quadratic assignment needs vastly more than 256 bit variables in a naive global encoding. | **Executable here** as a constrained p=1 *local permutation-update* solver: every global assignment still includes all 256 experts. No automatic application to IQ. The swap neighborhood is an approximation to the global optimization, not an architecture reduction. |
| CVaR-VQE — Barkoutsos et al. (Quantum 2020), https://quantum-journal.org/papers/q-2020-04-20-256/ | Replace expectation by lower-tail conditional expectation `CVaR_α(E)` when optimizing diagonal Ising energy. | Reported faster convergence in their tested optimization instances, including hardware, not LLM transfer. | **Executable here** as a different objective over exactly the same p=1 QAOA distribution. Does not prove better model weights. |
| VQE — Peruzzo et al. (Nat. Commun. 2014), https://doi.org/10.1038/ncomms5213 | Variational expectation minimization of Hamiltonians. | Photonic experimental eigensolver. | For diagonal classical quadratic objectives, a generic VQE adds no demonstrated advantage over QAOA or classical solvers. Not separately implemented. |
| Quantum walks — Childs et al. (STOC 2003), https://arxiv.org/abs/quant-ph/0209131 | Continuous-time unitary `e^{-itH}` on a graph / mixer over candidate permutations. | Quantum walk speedup for a constructed black-box traversal problem, **not** expert alignment. | Constraint-preserving permutation mixers require reversible circuit and cost oracle; not currently a validated GLM mapping acceleration. |
| Grover / amplitude amplification — Brassard et al. (2002), https://arxiv.org/abs/quant-ph/0005055 | Increase probability of a marked feasible/better mapping from `p` to order `sqrt(p)` state-preparation calls. | Proven query-complexity under ideal coherent oracle assumptions. | Full objective scoring/oracle construction and classical readout can be expensive; no demonstrated GLM transfer advantage. |
| Photonic quantum transfer learning — Quandela Perceval + MerLin, https://github.com/Quandela/HybridAIQuantum-Challenge and https://merlinquantum.ai/reproduced_papers/reproductions/quantum_transfer_learning.html | Freeze classical feature encoder, fit trainable photonic quantum layer. | Reproductions report Ants/Bees 90.8%, CIFAR binary 83.0% / 96.6% accuracy (task accuracy, not speed). | **Relevant to learned activation calibration**, not moving GLM expert or MLA tensor weights. No experiment on IQ checkpoints; do not present classification results as transfer speedup. |
| Classical low-rank dequantization — Tang (2018/2019), https://arxiv.org/abs/1807.04271; Chia et al. (2019), https://arxiv.org/abs/1910.06151 | ℓ² sampling / randomized low-rank SVT. | Algorithms with sublinear access under defined sampling models. | A strong *classical* counterbaseline for quantum-PCA/QSVT; prerequisite to claiming quantum advantage. |
| MOHAWK (architecture transfer) — Bick et al. (2024), https://arxiv.org/abs/2408.10189 | Matrix-mixer orientation, hidden-state matching, and weight/prediction distillation. | Official code https://github.com/goombalab/phi-mamba . | Necessary model-level validation and distillation separate from quantum solver; no quantum method in the original paper. |

## Exact quantum benchmark objective and algorithm

Let `C ∈ ℝ^(256×256)` be per-expert angular mismatch between `donor_router @ WARM` and IQ router rows. Let `A` and `B` be full 256×256 symmetric cosine affinity matrices of donor and IQ router directions. They are **not router coactivation counts**. Optimize permutations of all 256 labels:

`E(π) = Σ_i C[i, π(i)] + λ Σ_(i<j)(A[i,j] - B[π(i),π(j)])²`

`λ` is an explicit experimental hyperparameter (no hidden renormalization). Linear assignment (`λ=0`) is polynomial-time and solved optimally by Hungarian; **do not claim quantum advantage over it**. The quadratic affiliation term gives a quadratic assignment problem; its relevance depends on evidence that preserving router geometry improves *downstream* expert behavior.

Compute the classical Hungarian permutation `π₀` using all 256 router rows, then choose disjoint source-expert pairs as *move variables*. Every move swaps two destination positions of the **full 256-expert** permutation. For `q=8`, this subproblem has `2^8` feasible local configurations, but **does not reduce, drop, or merge any original experts or IQ layers**. Because each moved row belongs to exactly one pair and `E` is quadratic in pairwise similarities, the local energy is an exact QUBO:

`E(x) = E₀ + Σ_i l_i x_i + Σ_(i<j) q_ij x_i x_j`.

The implementation verifies this equality on every local bitstring. QAOA p=1 prepares `H^⊗q`, applies the diagonal `exp(-iγH_C)` phase, and the `exp(-iβΣX)` mixer; parameters are fit with SciPy differential evolution. CVaR-QAOA uses the lower energy tail instead of mean. The exact solver enumerates **the same local neighborhood**, providing a fair correctness ceiling. Opt-in IBM-compatible OpenQASM 3 generation uses `H`, `RZ`, `CX`, `RX` and measurements; only external simulator cross-checking was run. No QPU was used.

**Important:** `q` is the number of simultaneously optimized **local swap moves**, not the number of experts, layers, or checkpoints. Increasing `q` increases exact statevector memory as `2^q`; this is a quantum subroutine resource control, not downsizing IQ.

## Execute with real published checkpoints and WARM alignment

```
python scripts/benchmark_quantum_router_alignment.py \
  --glm-checkpoint /path/to/verified/GLM-5.3-BF16 \
  --glm-router-key model.layers.3.mlp.gate.weight \
  --iq-checkpoint /path/to/verified/IQ-checkpoint \
  --iq-router-key layers.3.moe.router.weight \
  --warm-map /path/to/saved/coordinate-map \
  --interaction-weight 0.01 --swap-pairs 8 --shots 1024 \
  --out /path/to/results.json --qasm-out /path/to/circuit.qasm
```

The IQ router key **must be checked against the actual output checkpoint**, not assumed to match this example path. The source is the BF16 model; raw E4M3 tensors require verified inverse-scale dequantization. The experiment fails closed on wrong shapes (donor 256×6144, IQ 256×4096, WARM 6144×4096), non-finite inputs, invalid permutations, or zero-norm router rows. For repeatable runs, use `--export-bundle` once, then `--bundle <npz>` with the same 256×256 matrices. No synthetic donor data is generated by the production CLI.

It reports **baseline energy, exact local optimum, QAOA/CVaR-QAOA results, probabilities, integrity, runtime, input provenance, and QASM**. QASM can be verified with qBraid built-in simulator or run through IBM's documented Qiskit primitives after checking backend calibration and runtime eligibility. Free QPU access does not imply zero data-loading or compilation cost.

## Empirical boundary

A full-256-row synthetic-matrix *unit/integration fixture* was executed locally to validate the pipeline. Its baseline and QAOA produced the same best objective; no speedup was observed or claimed. This is not a measurement on pretrained GLM weights. The exact local enumeration beat the variable-parameter noiseless QAOA solver on guarantee, as expected. QASM gate probabilities were independently checked against qBraid's exact simulator on a 2-qubit known-answer gate identity test. **No full-checkpoint GLM→IQ experiment, hardware run, or model-level forward/distillation comparison has occurred.** An improvement to this artificial pairwise matching loss would not imply better language-model quality.

## Production gate

Before applying a remapping to the IQ assembler, require: pinned donor/recipient checkpoint SHA-256 manifests, real paired activation captures, monotonic 78-layer correspondence, expert-router permutation provenance for all 256 experts at each MoE stage, preservation of top-8 routing, teacher/student activation error, held-out loss/perplexity, numerical map error and parameter hash checks, full-end-to-end elapsed cost (including encoding/shots/queue/readout), and improvement relative to Hungarian + strong classical quadratic assignment solvers. **Never alter IQ's original geometry or silently auto-select the quantum method.**