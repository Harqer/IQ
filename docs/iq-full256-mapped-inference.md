# IQ GLM-5.3 full256 — mounted-bucket inference preflight

**Scope:** IQ \`kofecheckmate/iq-glm53-all256\`, checkpoint \`IQ/runs/glm53-full78-local-20261025a\`. The checkpoint is immutable input. Do not install or download its 539 GB of weights on a laptop.

## What this patch changes

- Supports a \`layer_to_gpu\` map for IQ's existing physical layer sequence. It does **not** change Mamba-3 MIMO, CSA/HCA, MoE topology, the 12-layer AttnRes accumulation rule, tokenizer IDs, or weights.
- Keeps embeddings on the first CUDA device, LM head and reasoning on the final device, each AttnRes mixer on its associated physical layer device.
- Moves activations, masks and \`BlockAttnResState\` across GPU boundaries as required, including cuts *inside* a 12-layer residual group.
- Streams tensors from SafeTensors files in a read-only HF bucket mount rather than downloading an entire checkpoint or keeping every shard in CPU RAM.
- Adds metadata fingerprint and shard placement checks with a CPU-only CLI.
- Adds contextual errors: physical layer, operator, device, activation dtype/shape, allocated bytes.

## CPU preflight (does not load model weights)

Set \`IQ_CHECKPOINT\` to the **remote Hugging Face Job mount** and \`IQ_DEVICE_MAP\` to the verified metadata plan.

\`\`\`bash
export IQ_CHECKPOINT=/checkpoint
export IQ_DEVICE_MAP=/bucket/IQ/device-maps/full256-hybrid-pattern-h200x8.metadata-plan.json
python scripts/check_iq_device_map.py "$IQ_CHECKPOINT" --device-map "$IQ_DEVICE_MAP"
\`\`\`

The preflight verifies the exact SHA256 checkpoint index/config fingerprints, 235-layer schedule, pinned MIMO rank/chunk size, 20,283 shard inventory and total file size, and assignments for all indexed tensor keys. It reads metadata and the headers of the two mixed controller/global shards, **not model tensor values**.

Expected outcome for the known checkpoint: \`metadata_passed=true\`, \`inference_ready=false\`, \`tokenizer_found=false\` unless a separate authorized tokenizer mount has been provided.

For a generation-ready gate:

\`\`\`bash
python scripts/check_iq_device_map.py "$IQ_CHECKPOINT" \
  --device-map "$IQ_DEVICE_MAP" \
  --tokenizer-dir "$IQ_TOKENIZER_DIR" --require-tokenizer
\`\`\`

The original checkpoint lacks tokenizer assets. The correct tokenizer must be supplied from an authorized location; the validation script does not select an arbitrary substitute. A successful tokenizer-file check is **not** a token-vocabulary integrity proof: the generation script checks that \`len(tokenizer)==128256\` once the tokenizer runtime is present.

## Full inference (not executed or certified by this patch)

Only after explicit authorization to use suitable *paid* 8-GPU compute, with the authorized tokenizer available and the required pinned Mamba-3 CUDA/TileLang kernels verified:

\`\`\`bash
export IQ_DEVICE_MAP=/bucket/IQ/device-maps/full256-hybrid-pattern-h200x8.metadata-plan.json
export IQ_TOKENIZER_DIR=/tokenizer
bash scripts/hf_full_inference.sh /checkpoint --max-new-tokens 16
\`\`\`

The generator performs full-prefix IQ MIMO inference. It does **not** replace Mamba layers with attention, switch to SISO, train/fine-tune weights, or infer from another model.

## What to check when inference fails

| First failure | Diagnosis |
| --- | --- |
| Checkpoint validation | SHA256 mismatch, omitted shard, wrong tensor-to-device assignment, incorrect vocab |
| Load | Tensor name, shape, dtype, exact shard and target CUDA device |
| GPU memory | The device with out-of-memory, original peak residency and activation shape |
| Mamba-3 | Pinned upstream revision, MIMO rank-4, chunk size 16, CUDA/TileLang compatibility |
| AttnRes / boundary | \`block_sources\`, \`prefix_sum\`, masks and physical layer placement |
| CSA/HCA/MoE | First failing operator with physical layer and input tensor shape |
| Generation | NaN/Inf logits, token IDs outside vocabulary, first decode inconsistency |

**Do not edit \`COMPILE_COMPLETE.json\` to mark inference ready.** Keep the original evidence unchanged and create a separate runtime report after actual successful generation and evaluation. The \`trainable parameter count\` example in GeeksforGeeks is a training metric; \`requires_grad\` is not stored in SafeTensors and does not establish inference capability.

## Known limits

The source code supports multi-GPU allocation and transfers by design, but no multi-H200 IQ forward pass has been run. Its actual performance, GPU memory peaks, and kernel compatibility remain unverified. Mixed source/reconstructed Mamba layers must preserve their original dtype and numeric values; previous CPU checks covered 79 Mamba headers, 55 small identity-layer invariants and sampled MoE headers. None of those tests proves token generation.

The original foundation tokenizer is not packaged and the connected HF account currently lacks access to the gated Meta Llama 3.1 tokenizer. That is a separate external prerequisite.
