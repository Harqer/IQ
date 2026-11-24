# IQ Mamba-3 MIMO + CSA/HCA experimental Phase 4: CPU warm-start

## Scope and caveats

**Uses the original `state-spaces/mamba3-mimo-1.5b` pretrained weights**, original
128256-token vocabulary and all 24 full Mamba blocks. It is not the 235-layer IQ
checkpoint. It does not widen or transplant any Mamba pretrained tensors.

CSA/HCA here are **compressed-attention-inspired demonstrators**, not a
bitwise or architectural reproduction of DeepSeek V4. Unlike an already
trained hybrid, the newly inserted attention projections start untrained.

The CPU Mamba recurrence implements the public MIMO reference equations but
has **not been numerically cross-validated against the official CUDA/TileLang
kernel**. Phase 2 showed readable pretrained text from this CPU reference;
this is not proof of production-kernel parity.

## What's new

1. Pretrained Mamba weights are frozen and hashed, before and after.
2. Cache original hidden states after blocks 8 and 16.
3. Warm-start the added CSA/HCA-style attention modules with a *simple known
   historical-pool reconstruction proxy*. This trains both compressed paths
   on the original hidden width (2048), but **does not prove semantic retrieval**.
4. Run a bounded language-preservation step with original-Mamba teacher KL
   and next-token CE; gradients pass through frozen downstream Mamba blocks.
5. Check actual active next-token behavior on two held-out 3-token completions,
   versus the original model, with a 0.01-nat KL budget and unchanged greedy
   token selections. Save *only the two attention modules* if these narrow
   gates pass. These gates are NOT an overall language-quality certification.

The script never writes to the original model checkpoint.

## Execute on Red Hat OpenShift (CPU-only quota)

After downloading the ZIP:

```bash
cd ~/Downloads
unzip -o iq_mamba3_csa_hca_phase4.zip
cd iq_mamba3_csa_hca_phase4

oc create configmap iq-mamba3-csa-hca-phase4 \
  --from-file=iq_hybrid_cpu.py \
  --from-file=iq_calibrate_cpu.py \
  --dry-run=client -o yaml | oc apply -f -

oc delete job iq-mamba3-csa-hca-calibration --ignore-not-found
oc apply -f openshift_attention_calibration.yaml
oc logs -f job/iq-mamba3-csa-hca-calibration
```

When a `attention_checkpoint_saved` event appears, copy the **new attention
weights only**, while the completed job's pod is still available:

```bash
POD=$(oc get pod -l job-name=iq-mamba3-csa-hca-calibration \
  -o jsonpath='{.items[0].metadata.name}')
oc cp "$P:/scratch/iq_attention_pilot.pt" ./iq_attention_pilot.pt
```

If the result says `attention_checkpoint_rejected`, there is no accepted
checkpoint. This is intentional; do not silently promote a degraded hybrid.

## Runtime caveats

- OpenShift job requests no GPU; max memory 22 GiB and 3 CPUs, inside the
  previously reported project quotas.
- Full-model CPU teacher + gradient-through-frozen-backbone execution is
  expensive, even for one very short language-calibration step. Model load,
  hidden activation capture and generation may take significant time.
- The warm-start is supervised by a *proxy memory target*, not a qualified
  retrieval benchmark or production training dataset.
- The two held-out prompts measure only a narrow language-regression gate.
- For real CSA/HCA capability, subsequent supervised retrieval training and
  a production-equivalent implementation remain necessary.
- PIP uses `/scratch/pylibs` because OpenShift prevents installs into `/.local`.
