# IQ Mamba-3 MIMO + CSA/HCA — Phase 2 text-generation probe

This package follows a successful OpenShift run with:

- original checkpoint loaded with `strict=True` (24 blocks; 2048 hidden width; no remap)
- identity-initialized CSA/HCA extension exactly matched original CPU logits and greedy token IDs
- random nonzero attention contributions affected logits but were *not trained*

**Phase 2 tests actual decoded text, not just two token IDs**. It executes a fresh CPU Job (no GPU quota) that downloads the original pretrained checkpoint again, loads an explicitly selected Llama-3.1 tokenizer, produces several tokens and checks identity equivalence at every step. A publicly accessible Llama-3.1 tokenizer mirror is configured. Its vocab size and BOS ID are checked, but byte-for-byte identity against the gated `meta-llama/Llama-3.1-8B` tokenizer is **not established by these checks**. For strict provenance, configure an authorized original tokenizer and a Hugging Face token.

The CPU recurrence is still **not** validated against official CUDA kernels, and this does not train CSA/HCA or prove better language quality.

## Linux CLI (from any directory)

```bash
cd ~/Downloads
unzip -o iq_mamba3_csa_hca_phase2.zip
cd iq_mamba3_csa_hca_phase2
oc create configmap iq-mamba3-csa-hca-phase2 \
  --from-file=iq_hybrid_cpu.py --from-file=iq_text_probe.py \
  --from-file=iq_long_context_smoke.py \
  --dry-run=client -o yaml | oc apply -f -
oc delete job iq-mamba3-csa-hca-text-probe --ignore-not-found
oc apply -f openshift_text_probe.yaml
oc logs -f job/iq-mamba3-csa-hca-text-probe
```

Existing previous Job is left unchanged. This phase uses its own Job and ConfigMap names.

## What output proves

The output will contain `event: generated_step` records with readable decoded tokens and `identity_logits_exact: true`, followed by `event: completed`. These validate only that adding identity-initialized attention does not alter the CPU implementation's generation.

`active_untrained_wiring` uses random, tiny nonzero attention outputs strictly to test wiring. It is **not** a trained hybrid, and the next-token choice may remain the same.

The original weights and tokenizer IDs are never remapped or resized, no files are written over the pretrained checkpoint, and no actual Mamba pretrained parameters are trained here.

## Caveats

- If the model's baseline output is nonsense, investigate CPU recurrence parity *before* blaming attention; the original implementation is CUDA/TileLang only.
- Attention here is CSA/HCA-inspired: mean pooling + causal top-K compressed memory and local path, not exact DeepSeek-V4 CSA/HCA.
- This CPU generation recomputes full prefixes and may take significant time per token.
- OpenShift jobs use temporary scratch volumes; this job will download the 3 GB weights again.

## Phase 2 local synthetic validation (passed)

The standalone CSA and HCA compressed-memory paths were tested on 264-token synthetic sequences. Changing future tokens did not change earlier outputs; changing far-past tokens beyond the local window changed late outputs through each compressed-memory path. This confirms causal execution and memory-path activity, **not pretrained language quality**.
