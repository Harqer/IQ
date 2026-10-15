# IQ GLM-5.3 attention and router compatibility

**Authority:** published GLM-5.3 checkpoint, Hugging Face's
`GlmMoeDsaAttention` / `GlmMoeDsaIndexer` / `GlmMoeDsaTopkRouter`, and
NVIDIA's shared `GLM5Bridge` (not IQ's preexisting CSA implementation).

- [GLM-5.3 pinned config](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json)
- [Hugging Face publisher architecture implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
- [NVIDIA GLM5Bridge mappings](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py)
- [NVIDIA verified conversions](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html)

## Fixed now: MoE routing

The pinned GLM-5.3 config uses **FP32 sigmoid router scores,
`routed_scaling_factor=2.5`, `n_group=1`, `topk_group=1`, top-8
with corrected score choice, and normalized selected scores**.

The previous IQ model used a generic sigmoid top-k router with scale 1.0,
and did not implement group-aware selection for future donor variants.
The GLM 78-block variant now explicitly selects FP32 routing,
group-corrected top-k and source scaling. The existing IQ SiTU reference
continues to be the default of `StableLatentMoEConfig`, whereas the GLM
target uses source-compatible SiLU SwiGLU experts. CPU tests compare the
router scores, expert IDs and normalized weights against an independent
functional transcription of `GlmMoeDsaTopkRouter.forward`.

**Remaining expert limitation:** GLM acts in 6144-dimensional residual
space with 6144→2048→6144 expert matrices. IQ uses a learned
6144→4096 residual mapping and 4096→2048 latent expert path, so
preserving expert IDs, routing and activation does **not** establish
equal expert function or decoder quality.

## Pinned source MLA and DSA geometry

`iq_transfer/glm53_attention_contract.py` enumerates exact **HF-side**
MLA weights for all 78 GLM decoder blocks: 7 source tensors per block.
For the **21 full indexers**, it additionally enumerates
`wq_b`, `wk`, `weights_proj`, and both LayerNorm weights and biases.

Donor geometry from the GLM-5.3 publisher:
- width=6144; 64 attention heads;
- query low rank=2048; KV low rank=512;
- query/key non-rotary=192, rotary=64; value head=256;
- indexer: 32 heads × 128; top-k=2048; RoPE theta=8,000,000;
- full indexers at 0,1,2,6,...,74; shared indexers reuse a designated
  earlier full-indexer top-k selection.

Before conversion `iq_transfer/glm53_donor_preflight.py` verifies the
actual safetensors source header against all these shapes and verifies
`indexer.k_norm.bias` (missing from the older audit). It does not read
full tensor data.

## Explicit incompatibility: not a direct CSA weight copy

**GLM MLA:** independent `q_a_proj`, `q_a_layernorm`, `q_b_proj`,
`kv_a_proj_with_mqa` split into normalized compressed KV and shared
RoPE key, `kv_a_layernorm`, `kv_b_proj` per-head K/V expansion, and
dense `o_proj`. Indexer scores use FP32 ReLU and per-head weights,
with interleaved RoPE. GLM DSA masks non-top-k keys *after* computing
the MLA projected Q/K/V. It keeps per-layer KV caches and an independent
indexer key cache on full-indexer layers.

**Current IQ CSA:** a single `kv_proj`, compressed windows and optional
gated compressor, optional token indexer, `GroupedLowRankOutput`, and
its own query/candidate norms, sinks and RoPE handling. These operators
cannot load the GLM Q/K/V tensors 1:1, nor is equal top-k scheduling
sufficient for behavioral equivalence. In particular, IQ's grouped
output factorization, latent compression and missing GLM K/V projections
must be addressed by an explicit candidate subspace/coordinate transport
and validated teacher/student causal outputs.

The attention source manifest deliberately sets
`destination=null`, `weight_equivalent=false` and `converted=false`,
and `parallel_parameter_plan` labels attention
`BLOCKED_until_MLA_DSA_operator_mapping_and_indexshare_parity`.
Do not silently use `torch.reshape`, padding or random QKV adapters
as pretrained attention weight conversion.

## Real source-native attention evaluation, one layer at a time

`iq_transfer/glm53_official_attention.py` uses actual
`transformers==5.15.0` `GlmMoeDsaAttention` rather than duplicating
GLM's MLA/DSA kernels. It can load a **single real BF16 donor attention
layer** from indexed Hugging Face safetensors, with exact source tensor
keys, shape checks and source-native `indexer` submodules.

The executable `scripts/evaluate_glm53_attention_teacher.py` takes
**captured GLM source-layer input activations** from the corresponding
layer. It does not accept fabricated vectors or relabel IQ residual
states as GLM's 6144-dimensional source coordinates:

```bash
python -m pip install "transformers==5.15.0"
python -m scripts.evaluate_glm53_attention_teacher \\
  --donor "$GLM_BF16_DIR" \\
  --layer 6 \\
  --source-activations /scratch/glm-captures/layer-006.safetensors \\
  --output /scratch/iq/teacher-attention-layer-006
```

The activation file must contain `source_layer_input` of shape
`[batch, tokens, 6144]`, obtained from the actual GLM donor capture.
For a **shared-indexer** layer (e.g. 7), additionally pass
`--shared-indices` with the `dsa_indices` tensor from its preceding
full-indexer source layer's evaluation, using the *same* token sequence.
The evaluator writes source attention output and index selections plus
a checksum manifest. It does **not** overwrite any donor file, convert
IQ attention or assert IQ model quality.

The CI `official-glm-attention` job runs the real published
`GlmMoeDsaAttention` CPU operator on small diagnostic parameters
and validates causal masking / source IndexShare. Large pretrained
activations and checkpoints remain external inputs.

## Next gates (still open)

1. Choose **either** a genuine MLA/DSA branch in IQ with source-owned
   Q/K/V/DSA operator semantics and learned 6144↔4096 residual adapters,
   **or** a measured teacher-distilled CSA replacement; these are not
   equivalent training claims.
2. Verify causal attention scores, index selection, KV cache and gradient
   parity against actual official Hugging Face/Bridge GLM operator outputs
   before enabling IQ's GLM attention branch.
3. Integrate source operator and calibrated weight transfer into a full
   IQ Bridge provider and distributed model-wide TP/PP/EP checkpoints.
4. Verify all 78 blocks end-to-end with the real pretrained donor and
   learning/calibration data. No complete pretrained IQ checkpoint exists.
