# NVIDIA-grounded IQ parallel-hybrid foundation: design and execution gates

**Status:** donor preflight implemented; weights are **not yet transferred**. This
document describes the required production compiler/runtime, not a claim of
completed conversion or demonstrated capability retention.

## Canonical sources

1. NVIDIA's official guide: https://developer.nvidia.com/blog/implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/
2. TII Falcon-H1's Megatron Bridge architecture, mappings, and µP multipliers:
   https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/falcon/falcon-h1.md
3. **Actual NVIDIA-owned pretrained source**:
   https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16
4. Existing IQ-native Mamba-3 MIMO source: original official checkpoint already
   placed at `hf://buckets/kofecheckmate/mocha/` by the bucket owner.
5. Subsequent GLM capability source: `zai-org/GLM-5.3-BF16`.

**Provenance correction:** The Falcon-H1 checkpoints are developed by TII and
published under `tiiuae`. NVIDIA's Megatron implementation is not an NVIDIA
Falcon-H1 pretrained checkpoint. Actual NVIDIA Nemotron checkpoints are
interleaved Mamba-2 + attention + MoE, **not** pretrained 78-layer
Falcon-H1-style parallel hybrids. Neither nomenclature nor matching shape
permits an unmodified cross-architecture `load_state_dict`.

## Verified actual NVIDIA source geometry

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16`:
- BF16 Base checkpoint, about **247 GB** stored on the Hugging Face Hub.
- 88 physical layers: **40 Mamba-2, 8 attention, 40 LatentMoE**.
- Residual width **4096**, vocabulary **131072**.
- LatentMoE width **1024**, **512** routed experts, **22** experts/token.
- Model family `nemotron_h`. Specific revision must be obtained using
  Hugging Face's authoritative `model_info().sha`, not hardcoded to `main`.

IQ's target candidate from the user's GLM transfer intent:
- **78 fused parallel blocks**, each with Mamba-3 MIMO and CSA/HCA attention
  processing the **same** normalized input.
- Per-block dense FFN for GLM's first three stages and LatentMoE for the
  remaining 75 stages, following confirmed GLM checkpoint architecture.
- Width **4096**, subject to choosing a **single tokenizer owner**.
- Independent SSM recurrence and KV caches, one shared residual stream.

The NVIDIA source **cannot** directly populate all 78 attention or MoE
branches: it has only eight attention and 40 MoE operator layers. It cannot
populate even one **native rank-4 Mamba-3 MIMO** branch by byte-for-byte copying
its Mamba-2 parameters.

## Safe parameter ownership and initialization

| Recipient parameter | Primary source | Transfer disposition |
| --- | --- | --- |
| Tokenizer, embeddings, LM head | NVIDIA Nemotron 3 Super Base, **if** NVIDIA becomes lexical owner | adopt exact NVIDIA tokenizer and 131072 vocab; verify embedding/head shapes, pin SHA and source license |
| Mamba-3 MIMO recurrence | Official Mamba-3 MIMO 1.5B checkpoint (not NVIDIA Mamba-2) | recompile 24 pretrained MIMO layers into 78 native MIMO slots; 54 remaining must be function-neutral on initialization |
| CSA/HCA attention | NVIDIA eight source attention layers initially, later GLM-5.3 | transplant only mathematically compatible kernels; perform QKV/MLA transport, preserve RoPE semantics and measure attention behavior |
| LatentMoE | NVIDIA 40 source MoE layers initially, later GLM-5.3 | expert selection, latent projection transport, routing/activation calibration; must not claim 512 experts retained in 32 IQ expert slots |
| Mamba/attention joint fusion | Newly introduced IQ parameter | use true parallel branches + fusion projection; initialize to preserve the verified native Mamba residual operator, then learn attention contribution |
| Additional context and MoE slots | Existing donor layers through explicit weight correspondence + alignment | DO NOT label fresh identities or duplicated weights as independently pretrained |
| Reasoning control, Block AttnRes | IQ-native | initialize and train under explicit integration/evaluation tests |

IMPORTANT: Reassigning IQ's lexical owner from Mamba to NVIDIA invalidates
preexisting Mamba embedding/LM-head output claims. Pretrained recurrent blocks
must then be **re-aligned** to NVIDIA's representation. It is not mathematically
valid to preserve both lexical boundaries simultaneously without a verified
tokenizer/representation interface.

## Reference fused block

The architectural principle is NVIDIA's Falcon-H1 **parallel** Mamba-plus-
attention, not serial Mamba → attention. Both branches receive the same
block input, and their outputs are concatenated/projected into the residual
stream before the feed-forward transformation:

```text
 x ── norm ──┬── native Mamba-3 MIMO ──┐
             └── CSA/HCA attention ────┤─ concat → fusion → +x → FFN/MoE
                                      └──────────────────────────────────
```

The real implementation must correctly preserve:
- independent per-branch and per-layer inference caches;
- packed documents, causality, prefill-vs-step parity and cache reset;
- branch scaling (respect source semantics; Falcon-H1 µP multipliers belong to
  TII Falcon checkpoints and must not be blindly assigned to Nemotron weights);
- tensor-parallel sharding rules for Mamba, attention QKV, and expert parameters;
- multimodal fusion placement, Block AttnRes and model fingerprints.

## Execution (safe, immediate preflight)

```bash
python -m pip install -r requirements-model.txt
python -m scripts.preflight_nvidia_foundation \
  --output /kaggle/working/iq-nvidia-foundation-preflight.json
python -m unittest tests.test_nvidia_foundation -v
```

This loads **real NVIDIA repo metadata** and pins its full commit SHA; it does
not download the 247 GB weight set or publish a runnable IQ checkpoint.

## Required gates before declaring the full model initialized

1. Choose and fingerprint one exact tokenizer; update IQ vocab and global
   model layout together.
2. Add an executable parallel block and schedule to `iq_model/hybrid.py` and
   `iq_model/architecture.py`; remove/replace the older Mamba-dominance rule
   deliberately, not with a bypass.
3. Implement NVIDIA tensor-to-IQ param mappings against **real NVIDIA
   safetensors index and shape metadata**, with exact inclusion/exclusion
   coverage and staged processing. Keep NVIDIA Mamba-2 -> Mamba-3
   **explicitly prohibited** unless separately research-validated.
4. Recompile the official Mamba-3 rank-4 donor into the 78-slot fusion
   schedule, rather than loading an incompatible 32-slot checkpoint.
5. Add GLM-5.3-BF16 as the second capability donor, with layer-paired
   activation maps and provenance—not as a substitute for NVIDIA's lexical
   source.
6. Run strict checkpoint round-trip loading, unit/integration tests,
   numerical alignment, causal decoding, benchmark and held-out capability
   retention. Publish finalized shards under a **new** directory in
   `hf://buckets/kofecheckmate/mocha/`. Never replace the original raw donor.

Treat the official NVIDIA implementation as the **source of truth for the
parallel-block structure and checkpoint-conversion machinery**, but not as a
claim that NVIDIA's interleaved Nemotron weights are already parallel-model
weights.
