# IQ Phase 5 — Retrieval selection pilot on original Mamba-3 MIMO

This phase **continues from** the accepted Phase 4d `iq_attention_pilot.pt` stored on the `iq-mamba3-attention-artifacts` PVC. No original Mamba weights or Llama token IDs are changed. It trains **only Q/K selector projections** in the two **CSA/HCA-inspired** modules, preserving the pilot's values/output projections. It uses the CPU reference Mamba recurrence, not the CUDA kernel.

## What it evaluates

- Train/heldout supervised selection of a compressed memory pool containing a fact within a 270+ token prompt, with a competing distractor fact.
- Original Mamba vs pilot vs trained hybrid answer-token negative log-likelihood for heldout facts.
- Short baseline language prompts and same-tokenizer KL safety check.
- Strict gate: both selector validation losses, actual heldout answer likelihood, and ordinary language preservation must improve or meet limits. **It may reject a candidate.** Rejection preserves Phase 4d checkpoint and saves a JSON report.

**This is not yet real DeepSeek CSA/HCA**, nor proof of multi-kilostep context or official CUDA parity. The CPU reference and attention semantics remain experimental. A small nonzero attention residual can preserve fluency without proving retrieval gains.

## Execute (after downloading zip into ~/Downloads)

```bash
cd ~/Downloads
unzip -o iq_mamba3_csa_hca_phase5.zip
cd iq_mamba3_csa_hca_phase5

# Uses existing PVC and existing pilot file; no old checkpoint overwritten.
oc create configmap iq-mamba3-csa-hca-phase5 \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --from-file=iq_retrieval_phase5.py \
  --dry-run=client -o yaml | oc apply -f -

oc delete job iq-mamba3-csa-hca-retrieval-v1 --ignore-not-found
oc apply -f openshift_retrieval_phase5.yaml
oc logs -f job/iq-mamba3-csa-hca-retrieval-v1
```

If the run fails, inspect `oc logs job/iq-mamba3-csa-hca-retrieval-v1` and `oc describe job iq-mamba3-csa-hca-retrieval-v1`.

## Read the result even after the job completes

```bash
oc apply -f artifact_reader.yaml
oc wait --for=condition=Ready pod/iq-mamba3-attention-reader --timeout=180s
oc exec iq-mamba3-attention-reader -- ls -lh /artifacts
oc cp -c reader iq-mamba3-attention-reader:/artifacts/iq_phase5_retrieval_report.json \
  "$HOME/IQ/checkpoints/mamba3-csa-hca/iq_phase5_retrieval_report.json"
# Only when the quality gate accepted:
oc cp -c reader iq-mamba3-attention-reader:/artifacts/iq_phase5_attention_candidate.pt \
  "$HOME/IQ/checkpoints/mamba3-csa-hca/iq_phase5_attention_candidate.pt"
oc delete pod iq-mamba3-attention-reader
```

> The report is always written upon reaching the quality gate; if the job errors earlier, check logs. The saved 83.9MB Phase 4d pilot remains untouched on the PVC and local machine. The hardcoded pilot SHA-256 is from the verified Phase 4d user output; it fails safely if a different checkpoint is present.
