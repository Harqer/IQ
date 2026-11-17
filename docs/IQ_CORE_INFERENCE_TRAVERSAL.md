# IQ core-architecture and inference traversal (GitDiagram audit)

## Scope and invariants

Audited `Harqer/IQ` against the GitDiagram supplied by the project owner on 2026-11-16. The actual default branch has a separate `IQForCausalLM` Transformer (`iq_model/model.py`) and a **canonical** `IQHybridForCausalLM` (`iq_model/hybrid.py`); they are not synonyms.

The canonical schedule defined in `iq_transfer/complete_transplant.py` has **80 physical operators**: 32 rank-4 Mamba-3 MIMO mixers, 24 context-attention operators (12 CSA + 12 HCA), and 24 Stable LatentMoE operators, plus optional separate reasoning and residual controls. The role of optional Transformer attention is configurable; a purely recurrent target is not identical to preserving a pretrained Transformer donor's computations.

## Traced actual runtime

```text
scripts/serve_iq_hf.py
  -> checkpoint_preflight.py (source tokenizer/index checks, before CUDA allocation)
  -> complete_transplant.load_complete_iq_checkpoint()
     -> iq_model/hybrid.IQHybridForCausalLM
        -> embed_tokens
        -> schedule: state/mamba3.Mamba3MIMOState (pinned CUDA/TileLang)
                     attention/compressed.{CSA,HCA}
                     attention/context_dense.DenseContextAttention (optional)
                     mlp/latent_moe.StableLatentMoE / mlp/situ.SiTU
        -> residual/attnres (optional), reasoning/recurrence
           reasoning/verifier and energy/critic (optional)
        -> multimodal pathway (optional), lm_head
  -> scripts/serve_iq_hf.generate() (uncached full-prefix greedy reference)
  -> /v1/chat/completions HTTP response

iq_harness/runtime.AgentRuntime
  -> iq_harness/model.ModelBackend protocol
  -> iq_harness/iq_http_model.IQHTTPModelBackend
  -> existing IQ chat-completion server (text only)
```

Other traversed files: `iq_model/objectives/mtp.py` (multi-token training objective, not a generation cache), `iq_model/residual/mhc.py` (ablation), `iq_training/pretraining.py` (training wrapper), `iq_transfer/glm53.py` / `glm53_compile.py` (inspection and transformed GLM capability source), and `iq_transfer/mamba3_direct.py` (MIMO donor tensor replication).

## Foundation-source truth

- `compile_glm53_iq_checkpoint()` keeps Mamba-3 MIMO as its **lexical/recurrent foundation** (128,256 Llama-3.1 tokenizer IDs) and remaps selected GLM-5.3 context, indexer and MoE components. It does **not** replace the Mamba tokenizer with the GLM tokenizer, even though GLM is a donor.
- `compile_complete_iq_checkpoint()` uses the GPT-OSS-20B lexical source with a **201,088-row** padded LM head, plus Mamba recurrent and GPT-OSS-derived capability modules. Its original tokenizer defines **200,019 valid token IDs**; GPT-OSS uses its own special tokens, not Llama's.
- Neither compiled checkpoint alone proves GLM-level reasoning/coding performance. Source tensor coverage and transformation shape checks do not establish behavior preservation.
- Choosing a different foundation requires an explicit, source-verified tokenizer + embedding/head contract and suitable mapping; a tokenizer replacement does not transfer neural capabilities.

## Actual defects corrected on this branch

1. The inference server required Llama token IDs for **all** checkpoint sources. It now validates the manifest's lexical foundation and chooses Mamba/Llama or GPT-OSS rules.
2. An untrusted/partial index could reach the expensive CUDA model construction without path checks. The loader now verifies relative regular safetensors shard filenames, rejects symlinks, verifies declared sizes, and verifies the manifest/config fingerprint where present.
3. GPT-OSS padded output-head rows were previously not excluded during decoding. Generation now masks IDs outside the defined tokenizer vocabulary before sampling.
4. A missing/incorrect chat template could silently revert to ad-hoc `user:` and `assistant:` strings, invalidating the pretrained chat format. The server now rejects such requests and offers explicit raw completion mode.
5. A 2,048-token prompt could request more generated tokens beyond the pretrained Mamba donor's 2,048-token context. Full-prefix generation now enforces a bounded combined prompt + output budget until long-context quality is actually verified.
6. The agent harness' `ModelBackend` edge was only an abstract Protocol, with no IQ inference adapter. The new **first-party** HTTP backend executes text generation through IQ, rejects unavailable native tool calling, validates HTTP credentials/host requirements and preserves transcript roles.
7. The cheap pre-job checkpoint CLI now rejects unknown/missing manifests and corrupted or unsafe shard layouts before expensive runtime initialization.

## Verification boundaries

- **CPU unit tests**: metadata and shard path integrity; donor-specific tokenizer mapping and special IDs; real torch tensor full-prefix generation including padded output-head IDs; HTTP round-trip into IQ harness with the local fake server; explicit tool-call refusal; existing model tests.
- **Still required for inference readiness**: loading the actual compiled pretrained checkpoint, validating the upstream pinned Mamba-3 MIMO CUDA kernel and recurrent step/cache parity, generating coherent responses, measuring coding/reasoning quality and throughput, and rechecking tokenizer assets against immutable donor revisions.
- Actual full-model HF H200 inference is billed and was **not** launched. No model-weight checkpoint or hosted bucket was mutated.

## Decisions

The primary work target is **the existing custom IQ architecture**, not GPT-OSS or GLM-5.3 weight transplantation as an end in itself. The foundation is an implementation choice. Avoid calling any donor-remapping path lossless until tested with exact pretrained weights and equivalent outputs. Keep experimental source-alignment branches separate from the primary inference runtime.
