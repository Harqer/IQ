# GLM-first + full Mamba-3 MIMO: implementation status and reference usage

**Status:** Implemented reference-compatible CPU-tested integration; **NOT a trained or inference-validated GLM-5.3 checkpoint**. Do not overwrite legacy all256 or advertise model readiness.

## Components implemented

- `iq_model/glm_first_mamba.py`: retains the unmodified `GlmMoeDsaForCausalLM` as `.glm_model`, keeps GLM's own 154,880-ID tokenizer/embedding and untied LM head, installs temporary hooks after specified decoder layers (preserving their DSA top-k return value), and inserts 24 complete upstream Mamba-3 donor blocks.
- Every donor block includes its MIMO mixer **and** pretrained gated MLP, including both normalization sublayers. Blocks are reused directly from `MambaLMHeadModel.backbone.layers`, not approximated by mixing Transformer Q/K/V into SSM tensors.
- New per-block 6,144 -> 2,048 -> 6,144 linear adapters and scalar gates start at zero. The original GLM forward is unchanged when branches are disabled. Gates/adapters are trainable; GLM and Mamba donor weights are frozen in eval mode.
- Active training with full-sequence data is available by `hybrid.enable_branches(True)` and `hybrid(..., use_cache=False)`. Incorrect cache/padding/position semantics fail closed. Active `.generate()` deliberately raises until recurrent+KV cache integration and training quality are validated.
- `iq_transfer/glm_first_tokenizer.py`: checks all token IDs against model embeddings/head/special tokens and a pinned **full mapping SHA-256**. A 154880-row table alone does not establish alignment.
- `iq_model/glm_first_validation.py`: compares baseline teacher and zero-gated hybrid logits exactly on identical GLM token IDs and refuses nonzero/unvalidated paths.
- `tests/test_glm_first_mamba_hybrid.py`, `tests/test_glm_first_transformers_reference.py`, `tests/test_glm_first_tokenizer.py`, `tests/test_glm_first_validation.py`: CPU functional, negative, tokenizer and baseline-parity cases.

## How the integration is constructed (reference code, NOT a compute-launch script)

```python
# Requires locally staged/verified original GLM-5.3 weights and pinned
# state-spaces/mamba3-mimo-1.5b weight file. Do not run without GPU capacity.
from iq_model.glm_first_mamba import wrap_preloaded_official_models
from iq_transfer.glm_first_tokenizer import mapping_fingerprint

# glm is already a fully loaded original GlmMoeDsaForCausalLM.
# donor is already a fully loaded MambaLMHeadModel (all 24 blocks).
# glm_tokenizer was loaded from the exact same pinned GLM revision.
# GLM_REFERENCE_MAPPING_SHA256 comes from a separately stored, validated
# provenance manifest for the official GLM tokenizer—not computed from
# whichever tokenizer happens to be passed to the factory.
hybrid = wrap_preloaded_official_models(
    glm, donor, tokenizer=glm_tokenizer,
    expected_glm_tokenizer_sha256=GLM_REFERENCE_MAPPING_SHA256,
)
hybrid.eval()
# Baseline only, no Mamba execution, GLM-native generation:
# outputs = hybrid.generate(...), after reference GPU/weight checks.

# With an explicit training pipeline and supported GPU placement:
hybrid.enable_branches(True)
hybrid.train()
# loss = hybrid(input_ids=..., labels=..., use_cache=False).loss
# Backpropagate through adapters/gates; pretrained GLM and Mamba frozen.
# Active cached autoregressive generation remains BLOCKED.
```

`GLM_REFERENCE_MAPPING_SHA256` must be pinned to an independently fetched, immutable official GLM source revision and recorded with tokenizer file hashes and chat-template/EOS checks. Never derive it from an arbitrary runtime tokenizer and call that proof of provenance.

## Provenance and shape contract

- GLM model source: `zai-org/GLM-5.3-BF16`, pinned in existing transfer provenance to `abf42e6c0e53bc29a875511301e7b7fcd220315b`. Original 78-layer / 154880-vocab / 6144-wide model remains intact initially.
- Mamba source: `state-spaces/mamba3-mimo-1.5b`, revision `bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec`, with 24 mixer+MLP blocks, 2048 donor width, rank-4 MIMO, state 128 and 3824 MLP intermediate rounded to 3840 actual FC storage.
- New branch placement defaults to evenly spaced GLM decoder layer indices in monotonic donor order. This is an explicit, **provisional and untrained** schedule—not a ratio justified by the paper or a claim of optimality.
- Strict production factory checks official GLM norm, activation, routing and EOS config, official Mamba donor config, full block tensor layouts, and tokenizer fingerprint.
- Adapters are initialized on the Mamba donor block's device/dtype. Enabling requires colocation with target GLM hidden states; independent distributed placement is not implemented by this factory.

## Validation completed and outstanding

1. Python unit tests verify original lexical arrays and DSA indices survive zero-gated insertion; complete block structure; zero-gate trainability; frozen pretrained weights; fail-closed cache/padding/errors; exact baseline logits; and token-ID corruption rejection.
2. Optional tiny Hugging Face `GlmMoeDsaForCausalLM` CPU test verifies that the **actual** GLM decoder API is compatible with the overlay without downloading pretrained GLM weights.
3. The all256 bucket contains 79 final `model-mamba-NNN.safetensors` files (000..078), including final `model-mamba-018.safetensors`, alongside an orphan `.partial` transfer file. Full indexed-shard/hash verification is STILL NOT complete.
4. Open blockers: actual GLM-5.3 teacher weight integrity/parity; actual full Mamba donor kernel/MLP tensor parity; model-parallel GLM teacher/runtime memory; trained adapter quality; stateful KV+Mamba streaming cache; numerical and coherent multi-task generation. All must be validated before deploying as inference-ready.

**No paid compute, GPU job, training, inference endpoint, checkpoint overwrite or remote bucket mutation was performed during implementation.**
