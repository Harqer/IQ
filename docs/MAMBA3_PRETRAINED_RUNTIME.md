# IQ: exact pretrained Mamba-3 MIMO inference lane

## Decision and model boundary

Chosen pretrained base: `state-spaces/mamba3-mimo-1.5b`, pinned to
`bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec`. It has 24 Mamba-3
MIMO rank-4 blocks (including their gated MLPs), 2048 hidden width,
128 state size, a tied 128256-token head, and 2048-token training context.
The tokenizer is **meta-llama/Llama-3.1-8B**. Its neural weights are NOT
used: only the tokenizer and special-token assignments are reused.

`iq_model/mamba3_pretrained.py` runs all original source blocks in their
original order. It loads the original checkpoint directly, without
GLM/Transformer attention, trained adapters, weight tiling, identity-gap
blocks, or uninitialized IQ head projections. Source weights are
checked against the existing pinned SHA-256 in `iq_transfer/mamba3_direct.py`
by default. Upstream `from_pretrained` applies strict tensor-key loading.
The IQ-owned wrapper exposes the original model as `model.core`.

**This is an experimentally isolated, native-pretrained baseline, NOT
the 235-layer GLM-MoE/CSA/HCA experimental IQ checkpoint.** The latter
is retained unchanged. Whether it can be transformed into an equivalent,
coherent pretrained hybrid remains unproven.

## Run when a BF16-capable CUDA GPU is already available

```bash
# Install CUDA-enabled torch and the IQ pinned native Mamba runtime
bash scripts/install_model_runtime.sh

# Hugging Face login is needed if you lack access to Meta's gated tokenizer
hf auth login

python scripts/iq_mamba3_native_inference.py --preflight
python scripts/iq_mamba3_native_inference.py \
  --prompt "The next number after 1, 2, 3 is" \
  --max-new-tokens 16 --parity --eval
```

The runner does not provision paid GPU compute. It intentionally rejects
CPU, SM75/T4/RTX2060S (no native BF16), SISO substitutions, out-of-vocab
tokens, a broken tied head, missing blocks, and requests exceeding the
official 2048-token context. It reports generated text, timing, and
cached-versus-full-prefix logit errors at the first two decode positions.

The `--skip-sha256` flag is an **explicit diagnostic opt-out**, not the
default. Source checkpoint: ~3GB on disk plus runtime memory. The
official checkpoint is a base causal LM, not an instruction-tuned or
coding-benchmarked assistant; the `--eval` probes produce raw
arithmetic/reasoning/code continuations for manual quality review and
are not reported as passed benchmarks.

## Validation gates

1. CPU structural tests: native configuration, full 24-block order,
   recurrent rank, tied lexical tensors, tokenizer ID safety, and context
   boundaries. These tests do NOT demonstrate pretrained quality.
2. BF16 CUDA: pinned 3GB checkpoint downloaded and SHA-256 matched;
   strict upstream state_dict load; native fused MIMO kernel present.
3. Actual recurrent execution: two cached decoding scores versus uncached
   whole-prefix reference scores (error and tolerance recorded).
4. Actual generation: nonempty English/code continuation, measured
   end-to-end tokens/second, and representative coding/reasoning probes.
5. Expand IQ only after these pass. Any additional MoE, attention, or
   GLM-derived branch must be zero-effect by default until quality and
   parity are measured. An untrained cross-architecture transfer does
   not inherit the source model's reasoning ability.

No GPU inference was executed by the source-only changes; a complete
success report requires the runtime to print the real outputs from step 4.

Official sources:
- https://huggingface.co/state-spaces/mamba3-mimo-1.5b
- https://github.com/state-spaces/mamba/tree/main/mamba_ssm/models
