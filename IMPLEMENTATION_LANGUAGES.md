# IQ Implementation Language Boundaries

This policy separates research compatibility from the intended production/runtime architecture.

## Priority order

Use the language that best matches the workload rather than forcing one language across the stack.

```text
accelerator tensor hot path?       -> Mojo / MAX
systems/runtime ownership?         -> Rust
distributed control-plane service? -> Go
upstream ML research dependency?   -> Python
```

## Mojo / MAX

Preferred for stable accelerator-facing compute after an architecture component wins its ablation:

- fused recurrent kernels;
- Mamba/SSM optimized operators where MAX support is practical;
- NSA/MLA/attention kernels;
- FFN/norm/residual fusion;
- MoE routing/dispatch kernels;
- custom tensor operators and accelerator memory/layout work.

Do not rewrite a correct upstream Triton/CUDA implementation before the architecture is validated.

## Rust

Default systems language:

- checkpoint and safetensors tooling;
- tokenizer/runtime data structures;
- model-state/cache ownership;
- transfer/distillation infrastructure outside tensor kernels;
- dataset preprocessing and deterministic evaluation;
- inference runtime/server internals;
- artifact/version/schema contracts;
- concurrency-sensitive code.

## Go

Control plane only:

- experiment/job orchestration;
- remote worker coordination;
- training/inference service APIs;
- telemetry;
- artifact/job metadata;
- cluster scheduling integration.

Go should not own core tensor math.

## Python

Research compatibility layer:

- PyTorch reference model;
- official Mamba/FLA/Triton research code;
- architecture ablations;
- autograd-heavy distillation/training;
- notebooks and exploratory evaluation.

Python implementations may remain authoritative while the research is moving. Once a component is stable, move only the production/runtime responsibility to Mojo/Rust and retain a Python numerical reference for equivalence tests.
