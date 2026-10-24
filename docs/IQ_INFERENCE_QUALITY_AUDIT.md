# IQ full-checkpoint inference quality audit

**Scope:** `Harqer/IQ`, `hf://buckets/kofecheckmate/mocha`, H200 job
`6adbf1f1046eb82b1ecd880a` (now **CANCELED**).

## Observed, not inferred

- The server loaded the checkpoint, accepted an authenticated chat request, and
  returned HTTP 200. Output: `, Hadateraterateraterateraterater,vaNat …king,b`.
  This is a **failed language-quality check**.
- `mocha/transplant_manifest.json`: schema 2; method
  `mamba3_foundation_plus_glm53_capability_weight_transfer`;
  **calibration_mode=`weight_only`**.
- `mocha/iq_config.json`: vocabulary 128256; hidden width 4096;
  80 physical layers (32 Mamba3 + 24 context + 24 MoE).
- The donor `state-spaces/mamba3-mimo-1.5b` is a 24-layer pretrained
  **Mamba mixer + gated-MLP** causal language model. Its model card specifies
  the **Llama 3.1 8B tokenizer**, and its config ties embeddings.
- `iq_transfer/mamba3_direct.py` widens donor lexical tensors with exact
  duplication and 1/2 LM-head scaling; this algebra preserves donor logits
  **only if the hidden state remains on the duplicated donor subspace**.
- `iq_transfer/glm53_compile.py` adds GLM-5.3 attention/MoE remapped
  weights, while `complete_transplant.py` also initializes new controller
  modules. This does not reproduce the complete 24-layer donor function.
- Strict checkpoint coverage verifies keys, shapes, finite source tensors and
  successful copies; it **cannot prove functional equivalence or coherent text**.

## Conclusions by confidence

1. **Confirmed:** working GPU model forward and HTTP integration; incoherent output.
2. **Confirmed:** no activation-refined alignment recorded in this bucket;
   weight-only placement/geometry is the source of the compiled checkpoint.
3. **Confirmed structural gap:** donor MLP blocks and 80-layer recipient
   attention/MoE/controller operations are not guaranteed function-preserving.
4. **Not yet resolved:** actual logit distribution, layerwise RMS, model logits
   versus a true donor reference, and precise byte-level tokenizer equivalence.
   Tokenizer 128256 geometry and key Llama3.1 special IDs are now checked.
   Access to the donor model card confirms canonical tokenizer family; access
   to `meta-llama/Llama-3.1-8B/tokenizer.json` was restricted, so no full SHA
   comparison was possible. No evidence justifies claiming token ID mismatch.

## Remediation order (no new H200 for diagnostics alone)

1. **Quality gate:** one arithmetic canary must yield `4` or `four`,
   otherwise raise `MODEL_QUALITY_FAILURE`, cancel the job and report the
   actual text. HTTP 200 must never count as inference success.
2. **Provenance gate:** classify `weight_only` as experimental/unvalidated;
   the paid launcher requires explicit acknowledgment and retains the existing
   cancellation/timeout protections.
3. **Next *approved* inference:** record per-layer RMS, first-step entropy,
   top-5 token IDs/probabilities, logit min/max/std, and final-hidden RMS
   during **the same** model forward (no separate standalone GPU job).
   Compare where residual norms start to diverge and which token IDs dominate.
4. **Offline operator invariants:** test width replication, donor embedding
   + LM-head scaling, Mamba mixer parity on a tiny CPU fixture. This is a
   component check, *not* end-to-end language preservation.
5. **Data/activation alignment:** capture paired donor and student states on
   fixed token sequences at each mapped stage; fit residual-input and
   output/latent maps with held-out validation. Recompile with
   `activation_refined` provenance. Normalization scales and expert routing
   distributions need inspection, not blind adjustment.
6. **Student training:** likely necessary after architecture substitution.
   Distill donor logits (temperature/KL plus next-token CE), align selected
   intermediate states and sequence representations, then evaluate held-out
   perplexity and simple QA. Avoid claiming this is solved by just sampling,
   adjusting decoding temperature, or swapping APIs.
7. **Acceptance test:** run controlled canaries plus a held-out corpus,
   report pass rate, perplexity/CE and generation artifacts; keep a
   `generation_quality_verified` flag separate from runtime
   `checkpoint_loaded`. Only then optimize repeated JIT compilation/decode.

## Source locations

- `scripts/serve_iq_hf.py`: tokenizer checks, real generation, optional
  first-token/layer probes via `IQ_DIAGNOSTICS=1`.
- `scripts/iq_hf_inference.py`: guarded launcher + quality smoke + cancel.
- `scripts/audit_iq_checkpoint.py`: small metadata-only CPU audit.
- `iq_transfer/glm53_compile.py`: weight transfer/provenance.
- `iq_transfer/glm53_calibration.py`: weight-only versus activation-refined map.
- `iq_transfer/mamba3_direct.py`: lexical replication, mixer widening.
- Source donor: https://huggingface.co/state-spaces/mamba3-mimo-1.5b

**Cost policy:** No new H200 run was launched during this audit. Paid model
inference remains explicit only, and a single incoherent arithmetic response
halts the interactive session rather than keeping a GPU running.
