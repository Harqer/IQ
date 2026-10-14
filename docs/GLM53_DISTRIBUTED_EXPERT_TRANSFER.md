# IQ full GLM-5.3 expert-parallel execution and shard conversion

**Implemented:** an actual multi-process `StableLatentMoE` expert-sharded
operator with differentiable all-to-all token dispatch, and a BF16 expert
checkpoint compiler that preserves all 256 GLM expert IDs per MoE stage.
These do **not** amount to a fully GLM-5.3-converted or trained IQ model.

## Runtime and memory

`nvidia_parallel_78_config(expert_parallel_size=8)` creates the existing
78 fused blocks (0–2 dense SwiGLU; 3–77 Stable LatentMoE).
`IQHybridForCausalLM` resolves the **already-initialized Megatron Core expert
model parallel group** via `megatron.core.parallel_state`. You may instead
inject an explicit `expert_parallel_group` initialized by
`torch.distributed`. A mismatch fails before allocating the model.

`ExpertParallelStableLatentMoE` constructs **256/EP local routed experts**
and retains the original global sigmoid router, top-8 scores, delayed
routing bias, latent codec, shared experts and SiTU activations. It exchanges
token activations through autograd-enabled all-to-all, while integer global
expert IDs travel separately. Each global expert retains its original index;
no expert pruning, low-rank merging, or token dropping occurs.

Per-rank routed-weight BF16 lower bounds for 75 MoE stages:

| EP size | Experts per stage per rank | Approx routed matrix GiB / rank |
| --- | ---: | ---: |
| 1 | 256 | 450 |
| 8 | 32 | 56.25 |
| 16 | 16 | 28.13 |
| 32 | 8 | 14.06 |

These estimates **exclude** attention, Mamba, embeddings, shared experts,
router, gradients, optimizer states, activations and checkpoint buffers.
Full model memory will be materially larger. No claim that an eight-GPU
node alone can host this 78-block full model is made. Pipeline/model
partitioning will also be needed.

The group comes from NVIDIA Megatron Core. The execution operator remains
IQ's real SiTU/latent expert graph using PyTorch differentiable A2A;
it is **not** NVIDIA's TE grouped GEMM or Megatron Core `MoELayer`, and
it does not yet provide a `torch_dist` sharded state dict. A two-rank Gloo
test verifies outputs, input gradients and per-global-expert gradients
against the unsharded reference. GPU NCCL tests remain mandatory.

## Actual donor expert tensor conversion

Convert from the **real mounted GLM-5.3 BF16 HF safetensors** checkpoint and
**real activation-refined** IQ 78-stage coordinate calibration. A checkpoint
with FP8 block scales must first be dequantized using NVIDIA's documented
`GLM5Bridge`; do not interpret raw FP8 weights as BF16.

On a conversion worker, run a separate process for each EP rank (outputs in
distinct empty directories):

```bash
python -m scripts.compile_glm53_expert_parallel \
  --glm-checkpoint /mnt/mocha/your-actual-glm53-bf16-directory \
  --calibration /scratch/iq/activation-refined-78 \
  --output /scratch/iq/glm53-expert-ep0 \
  --donor-revision YOUR_VERIFIED_GLM53_BF16_COMMIT_SHA \
  --ep-rank 0 --ep-size 8 --device cuda
```

Then repeat with `--ep-rank 1` through `7`, using **different output
directories**. The path and revision above must refer to the actual source
you staged and verified, not an invented Mocha subdirectory.

The compiler reads each donor expert lazily via `SafetensorsSource` and
writes only that rank's owned expert tensors. Rank 0 additionally outputs
the three dense SwiGLU stage weights and replicated MoE router, bias,
latent codec, shared expert and norms. Each file is SHA-256 indexed.
`manifest.json` is written last, after every expected shard is present.
Every emitted parameter name matches IQ's nested fused block/module layout.

Coordinate transport is computed from real learned calibration matrices.
The transport is **not** function-preserving by construction: GLM operates in
6144-dimensional residual space, IQ uses 4096 residual/2048 latent channels,
and SiTU is not identical to the GLM feed-forward activation. Donor
teacher/student validation is required before the expert gate is opened.

To load the expert portion only:

```python
from iq_transfer.glm53_expert_shards import load_glm53_expert_shards

loaded = load_glm53_expert_shards(
    model,
    local_dir="/scratch/iq/glm53-expert-ep0",
    shared_dir="/scratch/iq/glm53-expert-ep0",
    ep_rank=0,
)
```

Each other EP rank uses its own `local_dir` and rank-zero `shared_dir`.
The loader verifies every shard's checksum, tensor shape, rank assignment,
target config fingerprint and expert-ID coverage **before updating weights**.
It deliberately does not claim to populate Mamba, GLM attention, embeddings,
fusion calibration, or Block AttnRes weights.

## Outstanding gates

- Actual GLM checkpoint and learned calibration are not supplied to this
  environment; no donor bytes were loaded, converted or uploaded.
- Transfer all GLM MLA/DSA attention tensors and verify IndexShare parity.
- Map NVIDIA Nemotron lexical weights and independent Mamba-3 24→78
  recurrent weights into their owned branches.
- Implement compatible Megatron Core distributed `torch_dist` checkpoint
  partitioning (including PP/TP/EP) for IQ-specific modules, and NCCL/H200
  performance, forward/backward and restart tests.
- Calibrate `ExpertResidualGate`, validate SiTU vs GLM behavior, and run
  held-out teacher/student language/reasoning evaluation.

Official NVIDIA sources:
- https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py
- https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html
