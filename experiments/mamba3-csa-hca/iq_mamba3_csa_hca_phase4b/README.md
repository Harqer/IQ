# IQ Mamba-3 MIMO + CSA/HCA Phase 4b (CPU OpenShift)

This fixes the actual Phase 4 failure: the optimizer **increased** an already tiny
memory-proxy MSE, so `memory_proxy_improved=false`. The previous reason text
incorrectly blamed Top-1/KL. Phase 4b uses target-normalized losses, controlled
backtracking with full optimizer/weight rollback, post-step measurements,
and specific `failed_checks` while preserving all language gates.

What this **does not** do: prove DeepSeek-equivalent CSA/HCA, improve retrieval
semantics, or validate CPU-vs-CUDA Mamba3 kernel parity. The reconstruction
proxy is deliberately limited; do not deploy a candidate for production.

## Run from Linux

```bash
cd ~/Downloads
unzip -o iq_mamba3_csa_hca_phase4b.zip
cd iq_mamba3_csa_hca_phase4b
oc create configmap iq-mamba3-csa-hca-phase4b \
  --from-file=iq_hybrid_cpu.py --from-file=iq_calibrate_cpu.py \
  --dry-run=client -o yaml | oc apply -f -
oc apply -f openshift_attention_calibration.yaml
oc logs -f job/iq-mamba3-csa-hca-calibration-v2
```

The 1.5B foundation is loaded by strict state-dict matching and never modified.
Only the 2 attention modules (about 21M parameters) are trained. An accepted
candidate must improve the normalized memory proxy by at least 1% for BOTH
modules AND preserve top-1 and KL thresholds for held-out continuations.

## Recover a candidate artifact if accepted

The `/scratch` directory uses **ephemeral emptyDir storage**. The pod stays alive
for ten minutes after saving a candidate so you can copy it to your Linux
machine. Run in a second terminal, while logs say `[artifact]`:

```bash
POD=$(oc get pod -l job-name=iq-mamba3-csa-hca-calibration-v2 \
  -o jsonpath='{.items[0].metadata.name}')
oc cp "$POD:/scratch/iq_attention_pilot.pt" "$HOME/Downloads/iq_attention_pilot.pt"
```

No checkpoint means the quality gate rejected it; check `failed_checks`.
The pod will exit automatically once the ten-minute extraction window expires.
