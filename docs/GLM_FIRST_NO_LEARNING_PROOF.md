# GLM-5.3 -> Mamba-3 MIMO: mathematical no-learning verdict

**Decision:** Weight-only, nonzero, function-preserving GLM-to-Mamba *replacement* is **NOT established**. The existing code cannot correctly claim it. Original GLM inference **can** be preserved without training by retaining the entire GLM computation and leaving every new Mamba branch at zero gate. In that case Mamba performs no useful language-model work.

## 1. Unconditional rank-nullity obstruction

Let x be an arbitrary GLM residual vector in R^6144 and A be the current donor input adapter of shape [2048, 6144]. Then:

```text
rank(A) <= 2048
dim(ker A) = 6144 - rank(A) >= 4096
for any v != 0 in ker(A): A(x+v) = A(x)
```

Thus A is not globally injective, no left-inverse exists on all of R^6144, and the Mamba donor cannot distinguish two residuals that differ only in v. The result is unconditional regardless of all pretrained weight values or compute used.

**Nuance:** GLM residual information is not lost if x is passed through an unmodified residual skip. This proof applies to representation transfer into Mamba, not to a zero-gated additive path.

## 2. What zero gating actually guarantees

If the original GLM decoder block maps x to G(x), and IQ adds an independent Mamba contribution:

```text
new_state = G(x) + gate * P_out( Mamba3Block( P_in(G(x)) ) )
```

Then gate = 0 yields new_state = G(x), assuming the implementation avoids NaN propagation, side effects, changes in dropout/cache and dtype drift. The actual IQ reference implementation also bypasses the donor when disabled. No training is necessary to preserve the existing GLM computation with all Mamba contributions absent.

When the gate is nonzero, logits after downstream GLM layers may change, and token probabilities/behavior are no longer guaranteed to match the GLM teacher. Original weight tensors can all remain byte-identical and the composite function can still change.

## 3. Pretrained Mamba semantics need more than copied tensor weights

The official 1.5B donor uses **24 mixer-plus-gated-MLP blocks in order**, each consuming representations from the previous Mamba donor block. IQ attaches these 24 pretrained blocks as independent GLM residual branches, where each instead consumes P_in(GLM_layer_i_hidden). This breaks the donor's learned inter-block hidden representation chain even though each *individual* donor block retains its pretrained weights and mathematical operations.

Consequently:

- Mamba donor weights retained: **YES** (assuming hash/loading validation).
- Mamba donor within-block operations retained: **CONDITIONAL** on pinned upstream runtime/kernel and numerical reference tests.
- Mamba donor original inter-block activation space retained: **NO** in the current interleaved GLM branch architecture.
- Full GLM pretrained computations retained: **YES only when branches are neutral**, and verified in the zero-gated reference tests.
- Useful nonzero recurrent contribution that preserves GLM's output exactly: **NOT PROVEN**.
- Replacing original GLM attention and MoE solely by tensor copying: **NOT CERTIFIABLE**.

## 4. Could a no-gradient method make the interface work?

A deterministic alignment or closed-form projection can be tested using paired GLM and Mamba hidden states for identical text and a strictly held-out evaluation set. This is **calibration**, even if no gradient optimizer is used. It does not remove the 6144->2048 global information bottleneck or prove semantic equivalence on arbitrary sequences. It may reduce average mismatch on observed states but must be measured.

Alternatively, three parallel 2048-wide lanes could span 6144 dimensions if their concatenated adapter is full rank. That only removes the *linear dimension* obstruction; it does NOT reproduce GLM attention dynamics or automatically align independently pretrained Mamba block representations. Widening the donor to 6144 changes the recipient and needs a mathematically verified remapping of all MIMO tensor layouts and a new functional test.

## 5. The criteria for avoiding any learning

A claimed no-learning feature must demonstrate ALL the following:

1. Matching tokenizer IDs/embeddings/LM head and GLM attention/MoE residual semantics across the full target vocabulary and supported context.
2. Complete, correct tensor name/shape/layout/permutation maps and donor checkpoint hashes.
3. A **mathematical function-preserving mapping** for every altered operation, not just equal tensor shapes.
4. GLM reference logit and layer-state parity on controlled forward/cached decode fixtures. Finite passing prompts are evidence but not a proof of universal equality.
5. Correct Mamba recurrent state evolution across the intended architecture and identical behavior with chunked/step decoding.

At present, requirements 3 and 5 are **NOT MET** for a nonzero recurrent replacement. The current branch is an exact-preservation **baseline**, not a verified accelerated conversion.

## 6. Offline proof and numerical probes added

- `iq_transfer/glm_first_no_learning.py`: dimension-bound certificate plus read-only teacher/branch logit comparison with max delta, mean delta, KL, top-1 agreement and strict source/ID/finite checks.
- `tests/test_glm_first_no_learning.py`: proves lower-bound nullity 4096, constructs an explicit 8->4 nullspace counterexample, confirms exact zero-gate identity and detects altered logits from an untrained nonzero branch.

```bash
python -m unittest discover -s tests -p 'test_glm_first_no_learning.py' -v
```

**No paid inference, GLM-5.3 weight downloads, checkpoint overwrites, or training were run to establish this theorem.** A real-source activation probe remains unavailable until the original GLM teacher and full Mamba donor weights are loaded on a suitable runtime.

## 7. Correction: original Mamba inter-block connectivity is now implemented

The original GLMFirstMambaHybrid class inserted 24 independent Mamba donor branches at different GLM layers. That preserved their individual mixer+MLP operations but not their pretrained inter-block hidden-state flow.

The new iq_model/glm_first_serial_mamba.py provides a distinct GLMFirstSerialMambaHybrid implementation that:

- Runs all 24 complete donor blocks in order, propagating the upstream (hidden_states, residual) pair and applying the official terminal Mamba norm_f.
- Retains GLM's original tokenizer, embedding/output weights, all 78 original Transformer blocks and residual graph.
- Inserts one 6144-to-2048 input adapter and one 2048-to-6144 output adapter, with an initially zero residual gate.
- Freezes original pretrained weights and blocks unsupported fused normalization, cached decoding and unvalidated sequence lengths.
- Requires official GLM lexical fingerprint verification in its production constructor; synthetic dimensions are rejected from that constructor.

At gate zero, original GLM behavior is unchanged. In an enabled branch, Mamba's internal donor-to-donor connectivity is preserved. **The adapter still has a minimum 4096-dimensional nullspace, does not establish the donor's original embedding distribution and does not turn Mamba into an exact GLM attention replacement.**

The independent-branch implementation remains for controlled comparison. Do not label it pretrained-Mamba inter-block-equivalent. The serial design is a sidecar rather than a replacement for the 78 GLM Transformer layers; no inference acceleration or quality preservation of a nonzero branch is claimed.
