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

The official pinned upstream TileLang source at
`e9594ce1c732d97440f0332fdc43170a2294dbfa` contains an
unguarded V tile load for short or non-multiple-of-16 prefills.
Earlier IQ H200 diagnostics found related illegal CUDA addresses.
Therefore **the runner refuses any pretrained GPU inference unless
the installed source contains the explicitly bounded, zero-filled tail
guard**. This source patch does not alter the donor's neural weights;
it is a kernel memory-safety fix, not proof of numerical correctness.

A source-based install into a disposable, **already CUDA-enabled**
Python environment with compatible TileLang/TVM dependencies:

```bash
# Use a CUDA torch+toolkit venv and IQ dependencies first.
# Repo root must be the current working directory.
python -m scripts.iq_mamba3_native_inference --preflight

# Prepare the exact source in a temporary checkout; DO NOT use latest.
git clone https://github.com/state-spaces/mamba.git /tmp/iq-mamba3-native
git -C /tmp/iq-mamba3-native checkout e9594ce1c732d97440f0332fdc43170a2294dbfa
python -m scripts.patch_iq_native_mimo_tail \
    --patch-checkout /tmp/iq-mamba3-native

# Ensure only the intended kernel source has changed.
git -C /tmp/iq-mamba3-native diff --name-only

# Build from the source checkout with existing pinned dependencies.
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation \
    /tmp/iq-mamba3-native
python -m scripts.patch_iq_native_mimo_tail --check-installed

# Authenticate for the model and Meta's gated official tokenizer if required.
hf auth login

python -m scripts.iq_mamba3_native_inference \
    --prompt "The next number after 1, 2, 3 is" \
    --max-new-tokens 16 --parity --eval
```

This procedure requires a working compatible CUDA toolchain and may
fail if the dependency solver uses incompatible TileLang/TVM versions.
Do not treat the source patch, a successful import, or unit tests as proof
of safe native CUDA prefill. End-to-end native GPU testing remains required.
The official pretrained checkpoint is ~3GB on disk and the 1.5B model
requires additional runtime and kernel memory. The released model is a
base causal LM, not an instruction-tuned coding assistant. The `--eval`
probes are raw continuations, not scored capability benchmarks.

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
