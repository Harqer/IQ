# IQ CSA/HCA + Mamba-3 incremental-cache reference audit

**Branch:** `fix/mamba3-mimo-cached-step`  
**Review PR:** https://github.com/Harqer/IQ/pull/39  
**Disposition:** Draft / not release-ready until the acceptance gates below pass.

## Pinned reference code

1. Hugging Face Transformers, revision `4cc2aa84301c9aa210b5513dab9fefae03981f6e`:
   [`modeling_deepseek_v4.py`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) — `DeepseekV4HCACompressor`, `DeepseekV4CSACompressor`, `DeepseekV4Indexer`, `DeepseekV4HCACache`, `DeepseekV4CSACache`, `DeepseekV4Attention`.
2. Official DeepSeek-V4 [model documentation](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/docs/source/en/model_doc/deepseek_v4.md) — causal compression boundaries and `DynamicCache` layer types.
3. Official state-spaces Mamba-3, IQ-pinned revision `e9594ce1c732d97440f0332fdc43170a2294dbfa`:
   [`mamba_ssm/modules/mamba3.py`](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py) — `forward`, `step`, `allocate_inference_cache`, `_get_states_from_cache`.
4. Pinned [`InferenceParams`](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/utils/generation.py) — `seqlen_offset` and layer-keyed state dictionaries.

DeepSeek's "hybrid" refers to interleaved CSA and HCA, **not** to a published combined CSA+Mamba-3 cache. IQ must compose the two upstream contracts at its physical-layer boundary. The composition is an IQ adapter, not upstream code or proof of numerical equivalence.

## One-to-one source cross-reference

| Upstream behavior | IQ source | Finding and remediation |
| --- | --- | --- |
| HCA projected weights buffered to closed windows, compressed entries, entry counts | `iq_model/attention/compressed.py:DeepseekV4HCACache` | Added cache methods `store_compression_weights`, `update_compressor_states`, `reset` |
| CSA previous-window Ca retained independently for compressor and indexer | `compressed.py:DeepseekV4CSACache` | Added `update_overlap_state` and separate indexer/overlap state |
| Shared K=V sliding cache with uncompressed local window | `compressed.py:DeepseekV4HCACache.update` | Added sliding local K state, retaining `window-1` before next decode token |
| HCA closed-window causality, CSA indexer sparse top-k | `compressed.py:_forward_cached` | Added incremental attention preserving closed-window selection and gated FP32 compression |
| Dense anchor projected K and V reuse | `iq_model/attention/context_dense.py:DenseContextCache` | Added incremental dense cache and causal per-step queries |
| Mamba-3 prefill writes angle/SSM/K/V states; decode consumes those states | `iq_model/state/mamba3.py:Mamba3MIMOState.forward` | Wired the pinned `_get_states_from_cache` and `step` path; call `step` with documented `[B,D]` inputs |
| Mamba-3 recommended chunk: `64/mimo_rank` | `iq_model/state/mamba3.py:recommended_mamba3_chunk_size` | Removed unsupported float32-specific divisor; updated rank-4 float32 regression 8 -> 16 |
| One state lifetime across heterogeneous layers | `iq_model/hybrid.py:IQHybridInferenceCache`, `IQHybridForCausalLM.forward` | Added layer-keyed attention caches and shared pinned Mamba `InferenceParams` with offset advanced once per model call |
| Local cached K shape `[B,T,D]` | `compressed.py:_forward_cached` | Corrected extra singleton dimension found in GitHub Actions run 37855595821 |

## Validation history and replay

- Existing stateless CSA/HCA unit suite: seven tests passed on an earlier remote test run (not evidence for the new cache).
- GitHub Actions run [37855595821](https://github.com/Harqer/IQ/actions/runs/37855595821) executed 141 tests; **two new cache tests failed** with `ValueError: candidate shape mismatch`, traced to local cache tensor shape `[B,1,1,D]`. Fixed in commit `722c322913f60c1a0d74e7107bc6900633f0647f`.
- `tests/test_compressed_context.py`: full-vs-cached CSA/HCA parity across aligned and unaligned chunk schedules, batch size >1, overlap, reset, closed-window boundaries.
- `tests/test_dense_context_cache.py`: full-vs-cached dense attention, reset, offset mismatch.
- `tests/test_mamba3_mimo.py`: validates pinned MIMO chunk recommendation for bf16 and float32.
- `tests/test_hybrid_incremental_cuda.py`: actual pinned CUDA Mamba-3 MIMO plus CSA/HCA/dense prefill-decode parity; **skips** if that actual GPU kernel and source revision are unavailable. A CPU-only CI success does not verify Mamba-3 GPU parity.

## Independent cross-check against DeepSeek's first-party inference

Additional upstream owner source: [DeepSeek-V4-Pro `inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py), including its `Compressor`, `Indexer` and `Attention`, and its companion `inference/kernel.py`. This source corroborates CSA previous-Ca/current-Cb overlap, FP32 compression softmax, token-aligned compressed-window writes, and local shared-KV sliding attention.

**Important differences, not covered by the current CPU parity tests:**

- DeepSeek's first-party production inference uses quantized FP8/FP4 operations and an indexer-side activation rotation before score projection; IQ's PyTorch/HF-Transformers-style eager reference uses ordinary floating point without those first-party quantized kernels. Numerical parity against DeepSeek's quantized checkpoint therefore **has not been proven**.
- The first-party source uses preallocated circular KV/cache buffers and dedicated `sparse_attn` kernel; IQ retains capped local K and appends long-range KV tensors, calculating candidates eagerly. This preserves the mathematical candidate selection tested in IQ but does not reproduce the first-party kernel throughput or memory allocations.
- First-party DeepSeek `Attention` implements CSA/HCA only. It contains no Mamba-3 recurrence; integration in IQ is the composition of two independent documented cache contracts, **not** a verbatim upstream combined reference.

These differences are not silently marked `no issue` by a passing suite. Matching the first-party quantized execution path requires adopting and validating the actual DeepSeek kernels/weight format, not inventing lookalike quantization.

## GitHub Actions verification

- Final validated commit before this audit addendum: `cc3f44eeb1acb718ad91f274796629e26feb2ed5`.
- [GitHub Actions #37856052642](https://github.com/Harqer/IQ/actions/runs/37856052642) **success**: compileall and **141 Python tests passed, 1 skipped** (the hardware-dependent real Mamba-3 CUDA integration test). This validates compressed/dense CPU cache equivalence, not real end-to-end MIMO GPU parity.

## Acceptance gates and unresolved boundaries

1. **Critical:** GitHub Actions must pass on the final head SHA, including new CSA/HCA and dense parity tests. Prior failed/cancelled builds do not satisfy this.
2. **Critical:** Run hybrid prefill + recurrent token decoding on a compatible CUDA runtime with the pinned Mamba-3 MIMO and CuTe step kernels; compare all resulting logits and cache state. CPU CI skips this gate.
3. **Critical integration boundary:** Cached generation currently rejects packed documents, padded new tokens, reasoning recurrence, and multimodal fusion to prevent unsupported state reuse or silently incorrect equivalence. Do not claim general IQ multimodal/reasoning cached generation.
4. **Performance gap:** The IQ cached compressed-attention implementation uses a per-token eager path rather than upstream vectorized prefill and optimized sparse kernels. Functional numerical parity does not prove long-context throughput or memory targets.
5. **Upstream integration difference:** IQ owns separate per-physical-layer cache objects; it does **not** instantiate Hugging Face `DynamicCache(config)`, since IQ's layer schedule also contains Mamba-3 recurrence, MoE and dense anchors. This is a deliberate adapter boundary requiring its own parity tests.
6. **No phase advancement** until critical gates are passed, or a blocked gate is expressly recorded for hardware-enabled verification.

## Re-audit invariants

For every follow-up patch: recheck upstream pinned code, ensure `store_compression_weights` and `entry_count` stay synchronized, keep CSA indexer and compressor overlap independent, verify Mamba's four states are neither recomputed nor wiped between decoding calls, compare stateless/prefill/chunked outputs, rerun every prior test, and inspect the actual GitHub Actions logs before declaring ready.
