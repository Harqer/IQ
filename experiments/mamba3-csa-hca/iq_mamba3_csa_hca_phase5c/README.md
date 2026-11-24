# IQ Phase 5c: direct frozen-Mamba-tail answer training

Phase 5b correctly rejected its candidate: HCA lost held-out selection accuracy (2/4 → 1/4), real QA top-1 stayed 0/4, and held-out answer NLL worsened (3.96082 → 4.01218). Its "semantic" auxiliary objective optimized similarity to token embeddings, **not** real Mamba output probability.

Phase 5c changes the training objective, rather than relaxing acceptance:

- Original official `state-spaces/mamba3-mimo-1.5b` checkpoint stays frozen; original 24 Mamba blocks unchanged.
- Reload the **accepted Phase 4d** attention-only checkpoint with SHA256 `f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53` from existing PVC. Do not use rejected Phase 5b weights.
- Frozen CSA after block 8 remains active; cache the exact first 16 Mamba block outputs including the CSA residual. **Only the existing HCA after block 16 is trained** using actual next-token answer loss backpropagated through the frozen last eight Mamba blocks. Use activation checkpointing to reduce peak memory, and a small HCA pool-label auxiliary loss.
- Select training progress using development **answer NLL**, not attention-pool selection. Locked evaluation is used only once after training. A candidate cannot pass on a tiny NLL improvement alone: it must improve locked top-1 QA, locked answer NLL, and not worsen the HCA selector; ordinary language KL and top-1 must remain within existing limits.
- Write `/checkpoint/iq_phase5c_retrieval_report.json` to PVC regardless of pass/fail. Save `/checkpoint/iq_phase5c_attention_candidate.pt` only on acceptance. Never overwrite `/checkpoint/iq_attention_pilot.pt`.

**Limits:** CPU-only Mamba recurrence is *not numerically checked* against official Mamba CUDA. Training a 281-token sequence through eight frozen recurrent Mamba layers is computationally/memory expensive; the 22 GiB container limit can be reached. If the Job is OOMKilled, treat that as a resource limit, not a model-quality rejection. The four locked synthetic examples are insufficient to claim general-language quality or real-world retrieval. This remains a CSA/HCA-inspired research model, not DeepSeek's exact compression architecture.

## Run on the existing CPU-only OpenShift namespace

Download the ZIP **to `~/Downloads`** using the chat attachment first.

```bash
set -euo pipefail
cd ~/Downloads
test -s iq_mamba3_csa_hca_phase5c.zip
unzip -o iq_mamba3_csa_hca_phase5c.zip
cd iq_mamba3_csa_hca_phase5c

oc create configmap iq-mamba3-csa-hca-phase5c \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --from-file=iq_retrieval_phase5.py \
  --from-file=iq_retrieval_phase5b.py \
  --from-file=iq_answer_phase5c.py \
  --dry-run=client -o yaml | oc apply -f -

oc apply -f openshift_answer_phase5c.yaml
oc logs -f job/iq-mamba3-csa-hca-answer-v3
```

If `oc logs` reports ContainerCreating, obtain logs after the pod becomes ready:

```bash
oc get pods -l job-name=iq-mamba3-csa-hca-answer-v3
oc logs -f job/iq-mamba3-csa-hca-answer-v3
```

Do not rerun with `oc apply` alone: immutable Kubernetes Job specs aren't automatically replaced. Only if you *want* to restart the experiment:

```bash
oc delete job iq-mamba3-csa-hca-answer-v3 --ignore-not-found
oc apply -f openshift_answer_phase5c.yaml
```

## Copy the persistent report even after a failed job

```bash
oc apply -f artifact_reader.yaml
oc wait --for=condition=Ready pod/iq-mamba3-attention-reader --timeout=180s
mkdir -p ~/IQ/checkpoints/mamba3-csa-hca
oc cp -c reader \
  iq-mamba3-attention-reader:/artifacts/iq_phase5c_retrieval_report.json \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5c_retrieval_report.json
# ONLY if quality_gate says accepted:true:
oc cp -c reader \
  iq-mamba3-attention-reader:/artifacts/iq_phase5c_attention_candidate.pt \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5c_attention_candidate.pt
oc delete pod iq-mamba3-attention-reader
```

If the reader pod already exists, don't re-create it; `oc apply` reports unchanged. If a prior reader is already using the PVC, check the actual pod before starting the job.

### Key log events

`phase5c_dev_before`, `phase5c_train_step`, `phase5c_dev_selected`, `phase5c_quality_gate`, `phase5c_report_saved`, and either `phase5c_checkpoint_saved` or `phase5c_checkpoint_rejected`.

**Real-1.5B status:** code and synthetic end-to-end tests are locally verified, but no true Phase 5c OpenShift run has been performed by the assistant.
