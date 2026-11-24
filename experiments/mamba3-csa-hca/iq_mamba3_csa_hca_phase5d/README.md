# IQ Mamba-3 CSA/HCA Phase 5d — V/O-only answer learning + source intervention

## Why this version

The actual Phase 5c OpenShift job improved 4-example previously exposed locked QA NLL from **3.9608** to **3.2439**, with correct top-1 answers increasing **0/4 -> 1/4**. Its HCA selector held **2/4** pool hits but its selector CE worsened **0.6891 -> 0.7244**, correctly triggering the configured gate. The previous locked test indices **12–15** have been inspected repeatedly and must not be represented as an untouched independent holdout in later iterations.

Phase 5d **does not relax the Phase 5c selector check**. It freezes HCA Q/K, learned 128-token compressor, norm, CSA, and the complete Mamba-3 foundation. Only the existing HCA value (`v_proj`) and output (`out_proj`) projections receive gradients from actual Mamba-3 answer-token loss propagated through the frozen final 8 Mamba blocks. HCA selector *weights* are checked bit-for-bit and locked selector score/hits must remain unchanged.

Newly held-out generated key indices **16–23** (eight records, not the previously exposed four) are tested after selecting the best state using **development answer NLL**. An additional **counterfactual intervention test** changes exactly one evidence-answer token in each locked prompt (without changing the question), and checks whether the hybrid becomes more sensitive to that changed fact than the Phase 4 pilot. It does not leak locked answers into gradient updates or checkpoint selection. The test is still very small/synthetic, so a pass is not proof of general retrieval.

Strict checkpoint acceptance requires all: better development answer NLL, better fresh locked QA NLL plus more correct answers than pilot, EXACT HCA selector stability, better counterfactual context sensitivity on average and at least 2/8 examples, ordinary prompt top-1 unchanged with KL <=0.01, foundation hash unchanged. Rejections leave Phase 4d original pilot untouched and save only the Phase 5d report. The `--min-context-gain 0.02` is an experimental hypothesis threshold; do not lower it in response to failure without independent evidence.

## Run on existing CPU-only Red Hat OpenShift

Download `iq_mamba3_csa_hca_phase5d.zip` to `~/Downloads` **first**.

```bash
set -euo pipefail
cd ~/Downloads
test -s iq_mamba3_csa_hca_phase5d.zip
unzip -o iq_mamba3_csa_hca_phase5d.zip
cd iq_mamba3_csa_hca_phase5d

oc create configmap iq-mamba3-csa-hca-phase5d \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --from-file=iq_retrieval_phase5.py \
  --from-file=iq_retrieval_phase5b.py \
  --from-file=iq_answer_phase5d.py \
  --dry-run=client -o yaml | oc apply -f -

oc apply -f openshift_answer_phase5d.yaml
oc logs -f job/iq-mamba3-csa-hca-answer-v4
```

If `oc logs` shows `ContainerCreating`, rerun that last command after the container starts. Re-applying an unchanged Job does NOT rerun it; delete the named Phase 5d Job first only if an intentional rerun is required:

```bash
oc delete job iq-mamba3-csa-hca-answer-v4 --ignore-not-found
oc apply -f openshift_answer_phase5d.yaml
```

The job uses the original pilot on your existing PVC `/checkpoint/iq_attention_pilot.pt` (SHA256 `f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53`) and writes **new** paths:

- `/checkpoint/iq_phase5d_retrieval_report.json` — written even when quality gate fails
- `/checkpoint/iq_phase5d_attention_candidate.pt` — only written on acceptance

For a completed Job, the report remains on the PVC. Use the included `artifact_reader.yaml` pod and copy after the Job terminates:

```bash
oc apply -f artifact_reader.yaml
oc wait --for=condition=Ready pod/iq-mamba3-attention-reader --timeout=180s
mkdir -p ~/IQ/checkpoints/mamba3-csa-hca
oc cp -c reader \
  iq-mamba3-attention-reader:/artifacts/iq_phase5d_retrieval_report.json \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5d_retrieval_report.json
# Only if phase5d_quality_gate says accepted:true:
oc cp -c reader \
  iq-mamba3-attention-reader:/artifacts/iq_phase5d_attention_candidate.pt \
  ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5d_attention_candidate.pt
oc delete pod iq-mamba3-attention-reader
```

A previously present reader pod may require reuse or removal; the reader is not needed during training.

## Scope and limitations

* Phase 5d is locally tested with the small-model synthetic training path, selector integrity assertion, value/output gradients, and evidence-answer token substitution. Actual 1.5B outcome must be measured on your OpenShift cluster.
* This is a **CSA/HCA-inspired** additive module, not DeepSeek's exact CSA/HCA implementation; compression and selectivity mechanisms are still experimental.
* CPU Mamba3 recurrence is not proven numerically equivalent to official Mamba3 GPU TileLang kernels.
* Eight synthetic locked prompts cannot establish broad language behavior, robust factual retrieval, or production readiness; a future run needs larger, independently generated/balanced data and factual ablations.
* The 22GiB memory limit and CPU-only model can produce OOM/timeouts; these are not model-quality results.
