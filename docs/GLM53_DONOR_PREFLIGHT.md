# GLM-5.3 donor preflight and IQ parameter mapping

**Goal:** before compute allocation, prove that the staged GLM-5.3 checkpoint
contains the exact Hugging Face parameter inventory required by NVIDIA's
`GLM5Bridge` and IQ's 78 composite blocks. This is a **CPU/file-metadata
gate**, not a GPU benchmark and not a substitute for trained coordinate maps.

Authoritative references:
- [NVIDIA GLM5Bridge source/registry](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py)
- [NVIDIA GLM-5.3 conversion documentation](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html)
- [Publisher GLM-5.3-BF16 safetensors index](https://huggingface.co/zai-org/GLM-5.3-BF16/blob/main/model.safetensors.index.json)
- [GLM-5.3 FP8/BF16 distinctions](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html#glm-5-2-glm-5-3-compatibility)

## What PR #50 now enforces

`iq_transfer/glm53_donor_preflight.py` checks:
- GLM publisher config geometry: 78 blocks, first 3 dense, next 75 MoE,
  256 routed experts, top-8, GLM DSA full/shared IndexShare.
- Every one of **57,600 routed expert projection names** (75 × 256 ×
  gate/up/down), plus all dense FFNs, 75 routers and biases, shared expert
  projections, MLA and full-indexer attention tensor names.
- All indexed safetensors shards exist, remain inside the donor directory
  and match their actual safetensors header key inventories.
- Tensor dtypes and key expert geometries match the publisher's configuration.
  For native BF16 expert conversion, FP8 weight tensors fail closed:
  NVIDIA GLM5Bridge must first dequantize them, or the actual official
  GLM-5.3-BF16 checkpoint must be selected.
- Config and index SHA-256, selected source ID and proof that **no donor
  weight bytes were materialized** by the audit.

`iq_transfer/glm53_expert_mapping.py` enumerates the exact GLM source →
IQ destination names with the owner rank of every routed expert tensor,
rather than inferring ownership from arbitrary file ordering. The existing
expert compiler now refuses unregistered outputs and verifies all source-
backed routed/shared/dense/router destinations are emitted once.

The publisher has an appended MTP block outside the default 78-block
decoder graph. We **do not** silently map it into the 78 IQ blocks.

## Local inspection from the mounted Mocha bucket

Use a local worker/CPU container with the pinned IQ requirements and actual
donor checkpoint mounted. The BF16 donor root must contain `config.json`,
`model.safetensors.index.json` and every referenced shard. No staged donor
paths are assumed.

```bash
python -m scripts.preflight_glm53_donor \\
  --checkpoint "$GLM_BF16_DIR" \\
  --identity zai-org/GLM-5.3-BF16 \\
  --ep-size 8 \\
  --require-bf16 \\
  --output /scratch/iq/glm53-donor-audit.json
```

**Important:** This is a **read-only** file/header scan, not a no-I/O
operation. On remote mounts, reading file headers may incur bucket requests
and latency. It does not copy or modify checkpoint data.

The full expert compiler (`scripts/compile_glm53_expert_parallel.py`)
now executes the same preflight before creating output shards. It still
requires real activation-refined calibration for every IQ stage and creates
a distinct manifest per EP rank.

## Remaining nontrivial architectural gaps

These cannot be hidden by matching 78 layers:
1. IQ CSA/HCA's output low-rank heads, compressor and indexer implementation
   are **not NVIDIA's GLM MLA/DSA operators**. Index-sharing top-k does not
   imply MLA/DSA weight or output equivalence. Exact projection mapping,
   kernel behavior, causal packed sequences and teacher correlation remain.
2. GLM residual width 6144 → IQ 4096 and GLM expert behavior (SwiGLU)
   → IQ SiTU/latent MoE require learned coordinate/operator alignment.
   Preserving all 256 expert *IDs* does not preserve trained semantics.
3. The Mamba-3 24→78 operator overlay is independent of GLM-5.3's weights;
   no Mamba-2 parameter should be misidentified as Mamba-3.
4. Full optimizer+checkpoint topology still only supports the IQ EP-only
   same-rank setup; Megatron Bridge does not yet register an IQ 78-block
   model/provider/mapping_registry. Model-wide TP/PP/DP sharding is absent.
5. Source revisions supplied by the caller are **not automatically verified
   against Hugging Face**. Validate the actual publisher commit, shard
   hashes, tokenizer/lexical compatibility and license before release.

**Acceptance path:** pass the donor metadata audit → produce actual
calibrated source-to-IQ tensors → implement and verify the GLM MLA/DSA
operator map → partition complete IQ model and training state through
documented Megatron APIs → validate forward/backward/restart on actual
hardware. No GPU procurement is necessary for the first metadata gate.
