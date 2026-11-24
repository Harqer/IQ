# IQ Phase 4c: repair memory warm-start and prevent language calibration from erasing it

## The specific Phase 4b failures addressed

- CSA started with a zero output projection. AdamW's fixed-scale steps could not find an improving update on the *real* Mamba-3 hidden activations, although its gradients existed.
- HCA did improve the memory *proxy* initially, but the later language update damaged both new attention modules: CSA proxy 145.7, HCA proxy 3.18 after calibration.
- The France and Jupiter token sequences still matched, but the maximum held-out teacher KL was 0.04148, above the configured 0.01 limit.

## Phase 4c changes

1. **Output-projection analytical line search**: for a fixed attention feature representation, find a descending step along the output gradient using the quadratic proxy objective. Limit the output update to a conservative norm. Subsequent Q/K/V updates use gradient-normalized, backtracked, loss-monotone steps.
2. **Separate representation learning from contribution calibration**: after warm-start, do not change attention tensor weights during language-preservation calibration. Instead test bounded *nonzero* residual contributions, selecting the largest one passing next-token KL and top-1 checks on three short prompts.
3. **Check the exact deployed path**: evaluate memory proxy both before and after the selected residual norm limiter. A low raw memory loss is not enough if the limiter destroys the signal.
4. **Refuse unsafe checkpoints**: unchanged foundation hash, memory proxy improvement for both CSA and HCA, the two three-token held-out trajectories, nonzero effective attention, and teacher-KL gate must all pass. Rejects exit with status 2 and do not save a checkpoint.

These fixes address the demonstrated Phase 4b failure; they do **not** establish long-context language capabilities. The proxy is reconstruction of a mean of early hidden states on one 136-token example, **not a semantic retrieval benchmark**. The CSA/HCA implementations remain simplified, not exact DeepSeek V4 variants. The CPU Mamba recurrence has not been numerically verified against the official CUDA kernel.

## Before running

Download `iq_mamba3_csa_hca_phase4c.zip` into `~/Downloads` from the chat link. The folder is created only after extracting that ZIP.

```bash
cd ~/Downloads
ls -lh iq_mamba3_csa_hca_phase4c.zip || exit 1
unzip -o iq_mamba3_csa_hca_phase4c.zip || exit 1
cd iq_mamba3_csa_hca_phase4c || exit 1
ls -l iq_hybrid_cpu.py iq_calibrate_cpu.py openshift_attention_calibration.yaml || exit 1

oc create configmap iq-mamba3-csa-hca-phase4c \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --dry-run=client -o yaml | oc apply -f -

oc delete job iq-mamba3-csa-hca-calibration-v3 --ignore-not-found
oc apply -f openshift_attention_calibration.yaml
oc logs -f job/iq-mamba3-csa-hca-calibration-v3
```

The namespace has **zero GPU quota**. This job requests CPU and runs in a restricted OpenShift container, with Python dependencies in `/scratch/pylibs` and the original Hugging Face Mamba-3 MIMO 1.5B checkpoint under `/scratch/hf`.

## Reading the logs

- `memory_projection_line_search`: corrected zero-projection warm start for CSA/HCA.
- `memory_warmstart_complete`: actual proxy improvements, accepted steps and Q gradients.
- `gain_calibration_trial`: actual teacher-KL and top-1 checks at each candidate bound.
- `gain_calibration_selected`: selected **nonzero** bound, no further modification of new attention weights.
- `post_calibration_memory_check`: raw and *deployed bounded* memory proxy values.
- `heldout_generation`: real three-token independent greedy continuations.
- `quality_gate`: accept/reject. Only `accepted: true` writes a checkpoint.

If the gate passes, the pod prints `[artifact]` and stays alive for ten minutes. In a second terminal:

```bash
POD=$(oc get pods -l job-name=iq-mamba3-csa-hca-calibration-v3 -o jsonpath='{.items[0].metadata.name}')
oc cp "$POD:/scratch/iq_attention_pilot.pt" "$HOME/Downloads/iq_attention_pilot.pt"
```

If the gate fails, **do not promote this checkpoint**, even if its short generated examples look coherent.
