# NVIDIA-grounded parallel IQ runtime — implementation milestone

**Reference:** [NVIDIA's Falcon-H1 implementation overview](https://developer.nvidia.com/blog/implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/) and [Megatron Bridge Falcon-H1 documentation](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/falcon/falcon-h1.html).

**Weights:** NVIDIA Nemotron 3 Super Base BF16, Mamba-3 MIMO original and GLM-5.3 BF16 are assumed staged by the bucket owner at `hf://buckets/kofecheckmate/mocha/`. This milestone does **not** inspect, download, transform or upload donor files.

## Executable block

`iq_model/hybrid.py:ParallelMambaContextMoEBlock` implements:

```text
                          ┌──────────── native Mamba-3 MIMO ────────┐
 x ── RMSNorm(x) ──────────┤                                        ├─ concat → W_f → +x ── IQ MoE → output
                          └──────────── CSA / HCA attention ────────┘
```

Both mixers receive the same pre-normalized hidden representation. Mamba uses
the existing packed-document varlen path, while CSA/HCA receives the canonical
attention mask, position IDs and document boundaries. Each composite block
contributes **one Mamba recurrent slot, one compressed-attention branch, and
one expert FFN**, not three serial layers.

`ParallelBranchFusion` starts with `W_f=[I,0]`, so its attention branch
does not corrupt a previously transplanted Mamba block at initialization.
The attention slice remains trainable and is enabled through ordinary gradient
updates. This is an IQ-specific *transfer-safe initialization*, not a
claim that it reproduces Falcon-H1's original trained output projection.

- `HybridLayerType.PARALLEL_CSA` and `PARALLEL_HCA` are serializable.
- Old interleaved schedules are unchanged and continue to load.
- `nvidia_parallel_78_config()` produces an explicit 78-physical-block
  geometry with 78 Mamba slots, 78 DSA/CSA attention branches, 3 dense SwiGLU
  blocks and 75 routed MoE stages (256 experts each, top-8),
  4096 residual width, 131072 NVIDIA lexical tokens, and Block AttnRes.
- This is a **candidate architecture, not an initialized model checkpoint**.
  Specifically, source Mamba-3 MIMO provides only 24 pretrained layers and
  NVIDIA's BF16 interleaved donor only eight attention layers and 40 MoE
  operator layers.

## Test and runtime boundaries

CPU tests run against the real fusion projection, operator accounting,
configuration serialization and negative validation paths:

```bash
python -m unittest tests.test_parallel_hybrid tests.test_nvidia_foundation -v
```

The fully instantiated model uses IQ's real production Mamba-3 CUDA kernels.
This milestone does **not** certify GPU execution, batch=1 recurrent decoding,
KV cache updates or prefill/step equivalence on hardware. These must remain
release-blocking before any claimed inference-ready checkpoint.

## What the next checkpoint milestone must include

1. Exact, pinned NVIDIA Base checkpoint manifest (index+tensor shapes),
   entitlement/license and tokenizer file hashes.
2. Real NVIDIA embeddings/LM head transfer (explicit NVIDIA lexical owner),
   together with the separate Mamba-3 donor 24→78 layer placement.
   Missing 54 recurrent slots must be explicit residual identities before
   training; do not duplicate and mislabel as pretrained.
3. Independent native Mamba-3 states and CSA/HCA KV caches per parallel block,
   with packed-document resets, decode-step parity and causal masking tests.
4. NVIDIA's 8 attention and 40 MoE donor mappings followed by GLM-5.3
   attention and expert capability transfer. Preserve source/provenance mapping
   per output tensor. The target currently uses 78 MoE blocks; the GLM dense
   first-three mapping needs a separate exact dense branch, not an invented
   expert conversion.
5. Staged MOHAWK alignment followed by real CUDA integration, checkpoint
   round-trip loading, held-out capability and latency evaluation.

This code builds the **NVIDIA-pattern parallel topology** using IQ Mamba-3
and CSA/HCA operators. It does not copy NVIDIA Megatron Core's Mamba-2 CUDA
operator or imply unmodified Nemotron checkpoint compatibility.


## 78-slot pretrained Mamba-3 transplant — implemented

IQ's existing pinned source checkpoint is **Mamba-3 MIMO 1.5B** with 24 recurrent
layers, 2048 hidden channels and rank-4 MIMO. The 78-block fused recipient has
4096 channels; the compiler now applies the existing **exact replication-space
input/output transformations** at 24 monotonic ordinal positions among 78.
Remaining **54** recurrent branches are **residual-neutral when the overlay is
applied**: the applicator zeros only their native Mamba output projections.
It never copies Mamba-2 weights into the Mamba-3 operator.

Once the bucket owner has made the actual donor bytes available on the worker
at a local directory containing \`config.json\` and \`pytorch_model.bin\`,
execute from the IQ checkout:

\`\`\`bash
python -m iq_transfer.cli mamba3-direct \
  --checkpoint /kaggle/working/mamba3 \
  --output /kaggle/working/iq-parallel78-mamba \
  --target-mamba-layers 78 \
  --no-foundation-globals
\`\`\`

The output includes 24 safetensors layer files, \`mamba3_transplant.json\`,
and \`SHA256SUMS\`; the 54 residual-neutral ordinals are recorded in the
manifest, not falsely serialized as pretrained tensors. **Use a new, empty
output directory.** The donor SHA-256 is pinned to
\`139c09e9728d1f10ac7f4354d2ae29aaca41851a954aedd8f6728c05aeb6c744\`
and the Hub revision is
\`bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec\`.

Run \`sha256sum -c SHA256SUMS\` in the output directory to check all shards;
the model applicator independently rechecks the SHA-256 entries before
loading the 78-slot foundation. The bytes may be manually uploaded to the
owner's \`hf://buckets/kofecheckmate/mocha\` under a separate output
subdirectory; this step does **not** access or modify the bucket.

**Lexical provenance:** \`--no-foundation-globals\` intentionally omits
Mamba's embeddings and LM head because this 78-block architecture uses NVIDIA
vocabulary geometry. Exact Mamba operator behavior is guaranteed only on the
Mamba source-aligned residual subspace; NVIDIA lexical embeddings require
subsequent coordinate alignment and evaluation. No claim of complete-model
behavior preservation is made.

**Expert safety:** New fused blocks have an \`ExpertResidualGate\` initialized
to zero, so uncalibrated random expert residuals do not change the
Mamba-derived hidden trajectory. Attention fusion is initialized to
\`[I,0]\`. Both are trainable and should be unlocked through real
GLM-5.3/NVIDIA transfer calibration, not by inventing pretrained weights.

**Unfinished execution gates:** 78-block actual CUDA inference, recurrent/
attention cache parity, NVIDIA embedding and MoE conversion, GLM-5.3
weight transport and held-out capability retention. These are still required
before any model-complete or deployment claim.


## Fused-block incremental inference — implementation

The parallel IQ runtime now supports a single authoritative Mamba-3 MIMO
\`InferenceParams\` container, with each fused block indexed by its original
Mamba layer ordinal. Its CSA/HCA branch has a separate cache keyed by the
same physical IQ layer index, never by source GLM layer numbering.

- \`IQHybridForCausalLM.allocate_inference_cache(batch_size, max_seqlen)\`
  checks that the pinned upstream Mamba-3 CuTe step kernel exists before
  allocating the cache.
- \`model(..., inference_cache=cache)\` rejects training mode, mismatched
  batch/config fingerprints, labels, packed document IDs, padded inputs,
  unsupported multimodal/reasoning execution, invalid positions, and a
  multi-token continuation after prefill.
- \`ParallelMambaContextMoEBlock\` forwards the same normalized input to
  native Mamba-3 with the shared upstream recurrent cache and CSA/HCA with its
  own attention-cache object. Its fusion and expert residual gates are
  unchanged.
- \`DeepseekV4CSACache\` explicitly owns direct-token index keys and the full
  token-value history when GLM's \`direct_token_indexer=True\`. The
  compressed-indexer mode retains the original window buffers and overlap
  contract. HCA caches compression windows separately.
- \`cache.reset()\` clears every attention buffer and upstream Mamba states.

**Validation:** CPU tests compare real compressed-attention prefill+token
decoding with full-sequence reference outputs, including direct-token DSA and
reset. Actual Mamba-3 CuTe CUDA inference remains a **GPU-required acceptance
gate**. No CPU substitute for Mamba-3 MIMO is used in production.

This implementation follows the pinned IQ source's validated Mamba step
contract and NVIDIA's Falcon-H1 principle of independent sequence-operator
state but one fused hidden representation. The current cache uses mutable
state, so a failure during the forward pass requires resetting/rebuilding
the cache before retry; no transaction-style rollback is claimed.

## GLM-5.3 complete expert-geometry update

The separate [full-expert/IndexShare contract](GLM53_FULL_EXPERT_PARALLEL78.md)
now specifies 3 dense + 75 256-expert top-8 MoE stages, all 78 DSA/CSA
mixers, and inter-layer top-k sharing. The existing note above documents
historical work; current full-expert configuration is not executable on a
single unsharded GPU. See the new preflight and distributed EP requirements.
