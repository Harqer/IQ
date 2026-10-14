# IQ 78-block GLM-5.3 full-expert architecture and transfer contract

## Verified source geometry

Authoritative publisher checkpoint:
`zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42`.
NVIDIA's shared `GLM5Bridge` represents the 78 HF transformer blocks as
156 Megatron HybridModel *physical* stages (attention + FFN/MoE). It does not
reduce GLM to fewer than 78 logical decoder blocks. Matching the count
simplifies positional correspondence, **not** 6144-to-4096 functional
equivalence.

GLM-5.3 uses 3 dense SwiGLU blocks (0-2), 75 MoE blocks (3-77),
256 routed experts/MoE stage, top-8 routing, 1 shared expert, DSA IndexShare
(full indexers at 0,1,2,6,10,...,74; shared sources elsewhere). Appended MTP
is not one of the 78 ordinary decoder blocks.

## Implemented in IQ

- `nvidia_parallel_78_config()`: 78 composite blocks, **all with
  PARALLEL_CSA** rather than alternate unrelated HCA operators. Mamba-3 MIMO
  and DSA attend to a shared normalized residual input in each block.
- `parallel_dense_layers=(0,1,2)`: each of these blocks executes
  `RMSNorm → SwiGLU(gate,up,down)` instead of creating fake routed experts.
- Blocks 3–77 instantiate Stable LatentMoE with **256 routed experts,
  top-8, one shared expert**. Original smaller serial IQ configs remain
  backwards compatible; old checkpoints retain the old config fingerprint.
- `parallel_index_sources`: validated exact GLM IndexShare source map.
  The full-indexer attention branch computes its actual top-k indices and
  forwards them to the next 3 designated shared-indexer branches.
  CPU attention parity tests cover full-sequence vs chunked prefill/step.
- `iq_transfer/glm53_parallel_contract.py`: validates an actual publisher
  config.json including stage count, expert geometry, dense and indexer types,
  and tracks HF stage i ↔ NVIDIA physical stages 2i/2i+1 ↔ IQ block i.
- `scripts/preflight_glm53_parallel.py` compares a real local donor config
  with IQ and writes a deterministic, SHA-256-fingerprinted mapping manifest.
- `iq_transfer/parallel_parameter_plan.py`: gathers only parameter/buffer
  metadata across ranks, sorts by global name and PP/TP/EP rank, and fails
  closed on missing dense/expert/attention/mixer/fusion parameter families.
  It labels all actual conversions `verified=False` until a later transfer
  and numerical validation. No checkpoint tensor values are copied by this
  preflight.

## Readiness and resource constraints

**Critical:** "256 experts" here preserves the **number and routing
geometry**, not GLM's full parameter capacity. GLM routed experts act in
the source 6144-dimensional residual representation; IQ's Stable LatentMoE
projects from 4096 to a 2048-dimensional latent expert. The weights require
real coordinate transport, functional alignment, and routing calibration.

The 75×256 latent expert stacks are still enormous: three 2048×2048 matrices
per expert implies roughly 483 GB of BF16 **routed-expert matrices alone**,
before embeddings, Mamba-3, dense attention, shared experts, optimizer,
activations, communications or checkpoint copies. The existing
`StableLatentMoE` is a PyTorch reference with all experts resident in
each process; merely increasing `num_experts` does **not** distribute it.
Production requires real Megatron Core expert parallelism, supported
parameter mappings, and measured memory allocation. CUDA single-process
model construction is not an accepted completion test.

The inter-layer IndexShare implementation currently **rejects packed
documents and padded sequences** instead of silently violating source
index-sharing semantics. A packed/CP/PP-aware DSA sharing state and backward
recomputation correctness gate remain before production.
The current CSA attention is an IQ reference approximation; NVIDIA's full
MLA/DSA operator is not automatically identical after weight transfer.
Numerical attention/teacher correlation must still pass.

## Run metadata-only validation on mounted Mocha donor

```bash
python -m scripts.preflight_glm53_parallel \
  --donor-config /mnt/mocha/path-to-actual-glm53/config.json \
  --output /scratch/iq/glm53-78-topology.json
python -m unittest tests.test_glm53_parallel_contract \
  tests.test_parallel_parameter_plan tests.test_parallel_hybrid -v
```

Use the **actual** mounted GLM-5.3 model directory; path above is an
illustrative location, not a presumed bucket subfolder. Nothing is uploaded
or modified at `hf://buckets/kofecheckmate/mocha` by these commands.

Next runtime/weight milestones, not yet accomplished:
1. Bridge-import actual GLM and Nemotron checkpoints with pinned revisions.
2. Correct TP/PP/EP mapping with real tensor metadata, allocate 256
   experts through sharded Megatron Core expert parallel execution.
3. Transform attention/MLA, dense SwiGLU and routed/shared expert tensors
   into IQ's 4096-wide residual/2048 latent spaces; transfer Mamba-3
   separately without overwriting its rank-4 recurrent operator.
4. Execute distributed and CUDA acceptance tests, compare IndexShare
   and packed-context behavior, verify hidden-state/logit/quality
   preservation and checkpoint round-trips.

References:
- https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html
- https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py
- https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json
