# IQ Phase 5b — content-aware HCA and answer-bearing attention pilot

This is a **research experiment** based on the intact 1.5B Mamba-3 MIMO model and the accepted Phase 4d attention pilot.
It is not the original DeepSeek CSA/HCA architecture. The CPU Mamba-3 recurrence remains unvalidated against the official CUDA implementation.

## Why Phase 5 failed

Four training records produced apparent overfitting: CSA reached 2/2 held-out selection, but HCA fell to 0/2. Both actual answer top-1 predictions were still wrong. The very small NLL improvement did not demonstrate retrieval.

## What changed

- **HCA** now compresses 128 tokens into a content-weighted pooled K/V, with learnable weights; zero initialization makes it initially equivalent to the former arithmetic-mean compressor (to numerical precision).
- **CSA** keeps its existing 4-token mean compression and sparse selection.
- Train Q/K **and** V/output projections using a token-embedding classification auxiliary objective, so the selector is not optimized in isolation. This *does not* directly optimize full-model answer NLL, which is checked separately.
- Uses 8 training records, 4 development records (early model selection), and 4 **locked evaluation** records. Prompts use distinct project keys and randomized filler; the token-answer vocabulary is fixed independently of the held-out examples.
- **Hard gate** on both compression selectors, real answer-token top-1 accuracy and NLL, and language preservation. A small NLL change alone is not accepted.
- Training receives original frozen hidden taps, but the result is evaluated end-to-end with Mamba-3; the frozen backbone and Phase 4d pilot are never overwritten.

**Note:** Four locked examples still provide only preliminary evidence. Even an accepted Phase 5b candidate remains a research checkpoint, not proof of general retrieval or full coherence. This is not training of the actual DeepSeek compression scheme. OpenShift has no GPU allocation, so this run uses CPU kernels and may be costly in CPU work.

## Run (after downloading ZIP to ~/Downloads)

```bash
set -e
cd ~/Downloads
test -s iq_mamba3_csa_hca_phase5b.zip
unzip -o iq_mamba3_csa_hca_phase5b.zip
cd iq_mamba3_csa_hca_phase5b
oc create configmap iq-mamba3-csa-hca-phase5b \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --from-file=iq_retrieval_phase5.py \
  --from-file=iq_retrieval_phase5b.py \
  --dry-run=client -o yaml | oc apply -f -
oc apply -f openshift_retrieval_phase5b.yaml
oc logs -f job/iq-mamba3-csa-hca-retrieval-v2
```

If this run is restarted, first delete *only* `iq-mamba3-csa-hca-retrieval-v2` job; do not delete PVC.

## Recover the saved report after the job completes (accepted or rejected)

```bash
oc apply -f artifact_reader.yaml
oc wait --for=condition=Ready pod/iq-mamba3-attention-reader --timeout=180s
mkdir -p ~/IQ/checkpoints/mamba3-csa-hca
oc cp -c reader \
  iq-mamba3-attention-reader:/artifacts/iq_phase5b_retrieval_report.json \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5b_retrieval_report.json
# Only if the quality gate ACCEPTS:
oc cp -c reader \
  iq-mamba3-attention-reader:/artifacts/iq_phase5b_attention_candidate.pt \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5b_attention_candidate.pt
oc delete pod iq-mamba3-attention-reader
```

The accepted **Phase 4d checkpoint remains** `/checkpoint/iq_attention_pilot.pt` (verified SHA-256 `f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53`). Phase 5b output files use their own distinct names. A rejection causes a failed Job but still saves the JSON report on the PVC.
