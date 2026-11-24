# IQ Mamba-3 MIMO: Phase 5e paired-context retrieval pilot

Phase 5d improved answer-token NLL on eight new locked cases but **0/8 top-1**. Its source-swap sensitivity gain was small relative to the untrained foundation. Phase 5e tests an actual **paired-fact, full-vocabulary top-1** training objective rather than relaxing that gate.

## Invariants

- Restarts from the **accepted Phase 4d** `/checkpoint/iq_attention_pilot.pt` on PVC `iq-mamba3-attention-artifacts`; verifies SHA-256 `f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53`.
- Keeps the 24 original pretrained Mamba-3 MIMO blocks and embeddings frozen, preserves the same original tokenizer interface and two CSA/HCA insertions at 8/16.
- Freezes HCA Q/K, norm and the 128:1 content scorer. Trains only HCA `v_proj` and `out_proj` with the real final eight frozen Mamba blocks.
- Each training key has a **source fact** and **one-token swapped source fact**, with the same question; learning both correct answers forces evidence-conditioned prediction rather than assigning one fixed answer to a project name.
- Uses real next-token CE plus a full-vocabulary hard-negative ranking term, selects checkpoints only on development pairs.
- Locked evaluation uses **new project keys 24–31**, never used or shown in Phase 5–5d (we exposed the earlier locked keys; those are no longer clean). It evaluates **both** source variants, matched source-swap change in logit margin, and ordinary language KL.
- Rejects a candidate unless original **and swapped** locked top-1 answer counts increase, both-case NLL improves, source sensitivity increases, development loss improves, and language guards pass. Never saves a rejected candidate.
- Writes the report to `/checkpoint/iq_phase5e_retrieval_report.json` whether or not training is accepted. The Phase 4d pilot is *never* overwritten.

**Limits:** This is a tiny, deterministic synthetic retrieval test, not proof of long-context generalization. The CPU Mamba recurrence lacks validated parity against the official CUDA TileLang kernel. The CSA/HCA-inspired pooling is not an exact DeepSeek implementation.

## Run with no GPU allocation

Download `iq_mamba3_csa_hca_phase5e.zip` to `~/Downloads` and run:

```bash
set -euo pipefail
cd ~/Downloads
test -s iq_mamba3_csa_hca_phase5e.zip
unzip -o iq_mamba3_csa_hca_phase5e.zip
cd iq_mamba3_csa_hca_phase5e

oc create configmap iq-mamba3-csa-hca-phase5e \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --from-file=iq_retrieval_phase5.py \
  --from-file=iq_retrieval_phase5b.py \
  --from-file=iq_answer_phase5d.py \
  --from-file=iq_answer_phase5e.py \
  --dry-run=client -o yaml | oc apply -f -

oc apply -f openshift_answer_phase5e.yaml
oc logs -f job/iq-mamba3-csa-hca-answer-v5
```

The Job deliberately has a new name (`answer-v5`) so `oc apply` will not silently reuse v4.

## Read the outcome

Look for `phase5e_quality_gate`, `phase5e_report_saved`, then either `phase5e_checkpoint_saved` or `phase5e_candidate_rejected`.

### Retrieve the diagnostic report safely after completion

The previous artifact-reader Pod was deleted. Recreate it using the included YAML. The Pod mounts the **existing PVC**. After the Job exits:

```bash
oc apply -f artifact_reader.yaml
oc wait --for=condition=Ready pod/iq-mamba3-attention-reader --timeout=180s
mkdir -p ~/IQ/checkpoints/mamba3-csa-hca
oc cp -c reader iq-mamba3-attention-reader:/artifacts/iq_phase5e_retrieval_report.json \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5e_retrieval_report.json
# Only if accepted and the file exists:
# oc cp -c reader iq-mamba3-attention-reader:/artifacts/iq_phase5e_attention_candidate.pt \
#   ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5e_attention_candidate.pt
oc delete pod iq-mamba3-attention-reader
```

This report is *diagnostic*; no external inference serving changes are made.
