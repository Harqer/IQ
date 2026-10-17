# Bias-filtered weight transfer

IQ may filter a source model's **causally validated asymmetric representation**
before transporting it into IQ. The source checkpoint is not rewritten.

## Preservation contract

The filter must preserve:

- general NLP/language capability;
- coding and repository-engineering capability;
- reasoning/tool behavior used by the source;
- native multimodal capability when the source has one.

Standard GLM-5.3 is a text-generation model. GLM-5.3-Flash is the multimodal
GLM-5.3 variant and has a separate vision tower plus image/video tokens. Do not
apply a text bias filter to the vision tower unless an independent multimodal
audit demonstrates the same causal feature there.

## Pipeline

```text
paired counterfactual calibration prompts
  -> donor activation capture
  -> protect known super-activation/superweight coordinates
  -> robust median/MAD scaling
  -> shrinkage covariance
  -> whiten
  -> identify the concept-correlated subspace
  -> causal intervention validation
  -> erase only validated directions
  -> fit IQ coordinate/operator maps
  -> capability regression gate
  -> transfer
```

`iq_transfer.bias_filter.fit_bias_filter()` implements the transfer-time
whiten/project/unwhiten filter. It follows the LEACE geometry but keeps a
protected-coordinate set for high-impact outliers that must not be silently
edited.

A fitted filter is **not promotable** merely because a probe score falls.
Run the source model with and without the temporary activation intervention and
measure the requested behavioral asymmetry. Then apply `CapabilityGate`.

Default promotion budget:

- at least 30% reduction in the targeted asymmetry;
- <=1% absolute NLP regression;
- <=1% absolute coding regression;
- <=1% absolute multimodal regression when the source is multimodal.

If the filter fails any preservation gate, reject that direction rather than
increasing erasure strength.

## Political-bias evaluation

Use symmetric prompt pairs and report behavior rather than an ideological
target. Examples include country/entity swaps, support/oppose framing swaps,
and multilingual equivalents. Keep language, sentiment, refusal tendency,
answer length, and generic safety behavior as nuisance controls so the filter
does not confuse them with the targeted asymmetry.

The filter should flatten asymmetric treatment, not delete factual political,
historical, cultural, geographic, or linguistic knowledge.
