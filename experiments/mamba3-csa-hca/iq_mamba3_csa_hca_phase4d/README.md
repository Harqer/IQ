# IQ Mamba-3 MIMO + CSA/HCA Phase 4d — persistent artifact fix

Phase 4c passed its quality gate on the user's CPU-only Red Hat OpenShift namespace, but it saved its ~84 MB attention-only checkpoint to `/scratch` on an `emptyDir`. That filesystem is ephemeral and `oc cp` cannot retrieve it once the pod reaches Succeeded.

Phase 4d **does not change the attention model or its training code**. It reruns the same Phase 4c code and criteria but saves any accepted checkpoint to a new, dedicated 1Gi PVC `iq-mamba3-attention-artifacts` at `/checkpoint/iq_attention_pilot.pt`. It checks write permissions before installing dependencies. The PVC survives job completion. A reader pod is created only AFTER the training job finishes, avoiding a potential ReadWriteOnce multi-node attach conflict.

## Run from your Linux terminal

Download `iq_mamba3_csa_hca_phase4d.zip` to `~/Downloads` first.

```bash
set -e
cd "$HOME/Downloads"
test -s iq_mamba3_csa_hca_phase4d.zip
unzip -o iq_mamba3_csa_hca_phase4d.zip
cd iq_mamba3_csa_hca_phase4d

oc create configmap iq-mamba3-csa-hca-phase4c \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --dry-run=client -o yaml | oc apply -f -

oc apply -f pvc.yaml
oc apply -f openshift_attention_calibration.yaml
oc logs -f job/iq-mamba3-csa-hca-calibration-v4
```

If PVC provisioning prevents scheduling, use `oc get pvc iq-mamba3-attention-artifacts` and `oc describe pvc iq-mamba3-attention-artifacts`. If the write check fails, the job reports that immediately rather than downloading dependencies and the model. PVC creation requires a default StorageClass or an explicit `storageClassName` and enough remaining namespace storage quota. The last user-provided quota reported 70Gi/80Gi requested.

## Copy after job succeeds (no time window)

```bash
oc wait --for=condition=complete job/iq-mamba3-csa-hca-calibration-v4 --timeout=900s
oc apply -f artifact_reader.yaml
oc wait --for=condition=Ready pod/iq-mamba3-attention-reader --timeout=180s
mkdir -p "$HOME/IQ/checkpoints/mamba3-csa-hca"
oc exec iq-mamba3-attention-reader -- ls -lh /artifacts/iq_attention_pilot.pt
oc cp iq-mamba3-attention-reader:/artifacts/iq_attention_pilot.pt \
  "$HOME/IQ/checkpoints/mamba3-csa-hca/iq_attention_pilot.pt" -c reader
sha256sum "$HOME/IQ/checkpoints/mamba3-csa-hca/iq_attention_pilot.pt"
oc exec iq-mamba3-attention-reader -- sha256sum /artifacts/iq_attention_pilot.pt
```

Check that the local and PVC SHA-256 values match. The reader pod consumes a tiny CPU/memory request while active. When the copy is verified:

```bash
oc delete pod iq-mamba3-attention-reader
```

**Do not delete the PVC** until you have independently verified the local copy. The training job can complete without the reader; the artifact is not tied to a live pod. The original pretrained Mamba-3 model weights are not modified. The saved checkpoint is an attention-only calibration pilot, not proof of semantic retrieval quality or official CUDA numerical parity.
