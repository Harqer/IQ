# Bias-Sanitized Weight Transfer

IQ may flatten a measured political-asymmetry representation before donor activation maps are used for weight transport. This is a transfer-time filter, not a blanket rewrite of the source checkpoint.

## Preservation contract

- Preserve coding and general NLP by default.
- Do not remove political knowledge, entities, languages, or viewpoints merely because they are politically related.
- Only target a counterfactually measured asymmetric behavior that passes causal intervention testing.
- High-magnitude activation dimensions may be protected from erasure; magnitude alone is never evidence of bias.
- A fitted filter is a candidate only. It cannot be used for transfer until activation, causal, and capability gates pass.
- `GLM-5.3-BF16` is a text checkpoint, so coding + NLP are mandatory preservation metrics.
- If a multimodal GLM checkpoint is used, keep its vision tower/projector out of the text-bias edit scope and add at least one required vision/multimodal capability metric before approval.

## Flow

```text
paired counterfactual prompts
  -> donor activation capture
  -> raw magnitude-outlier protection
  -> pair centering
  -> shrinkage LEACE candidate
  -> held-out leakage/distortion gate
  -> causal intervention validation
  -> coding/NLP/(vision) capability gate
  -> approved donor-side activation filter
  -> coordinate/operator transport
```

Pair centering subtracts the midpoint of each matched pair before fitting, reducing shared topic, language, and scale nuisance. LEACE is fitted with covariance shrinkage and covariance-trace protection. BF16/FP16 activations are projected in FP32 and cast back afterward.

## Fit a candidate

Use existing IQ capture artifacts produced by `TorchActivationCapture` / `save_capture_records`.

```bash
python -m iq_transfer.cli bias-fit \
  --fit-a captures/political_a_fit \
  --fit-b captures/political_b_fit \
  --validation-a captures/political_a_validation \
  --validation-b captures/political_b_validation \
  --tap layer.40.residual_out \
  --target-concept political_asymmetry \
  --space-name donor.layer.40.residual_out \
  --protect-magnitude-outliers \
  --output artifacts/glm53-political-filter
```

For a multimodal source, add a protected modality and an explicit required multimodal metric:

```bash
  --protected-modality vision \
  --required-capability-metric coding \
  --required-capability-metric nlp \
  --required-capability-metric vision_vqa
```

The command reports `"approved": false` intentionally.

## Approval

Approval requires all three evidence classes:

1. **Activation gate** — target leakage is materially reduced while held-out representation distortion remains within policy.
2. **Causal gate** — temporarily applying the filter reduces the measured political asymmetry and does not materially perturb matched controls.
3. **Capability gate** — baseline versus filtered evaluations satisfy every required capability metric.

```python
approval = approve_bias_filter(
    artifact,
    metrics,
    activation_gate=BiasFilterGate(),
    causal_validation=causal_validation,
    baseline_capabilities=baseline_scores,
    filtered_capabilities=filtered_scores,
    capability_rules=(
        CapabilityMetricRule("coding", max_degradation=...),
        CapabilityMetricRule("nlp", max_degradation=...),
    ),
)
```

For a multimodal checkpoint, include its required vision metric in both score dictionaries and in `capability_rules`.

Only an artifact paired with its matching `BiasFilterApproval` may be passed to `sanitize_activation_pair()`; fingerprint mismatch fails closed.

## Evaluation corpus

The political audit corpus should be balanced and multilingual, with matched semantic pairs that swap country, party, ideology, institution, or policy stance while holding task form constant. Keep separate fit, causal-validation, transfer-validation, and capability-regression sets. Do not reuse the same prompts for fitting and final acceptance.

The first objective is flattening unjustified asymmetry with the smallest representation change possible. If LEACE cannot meet the preservation gates, reject the filter rather than relaxing the coding/NLP/multimodal gates.
