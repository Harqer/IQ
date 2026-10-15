# IQ's first native Megatron Core model builder (TP1 / PP1 / EP)

## Specification

The authoritative NVIDIA sources are:

- [Bridge: contribute a model](https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html)
- [Bridge: builder-backed model configurations](https://docs.nvidia.com/nemo/megatron-bridge/nightly/bridge-guide.html)
- [Megatron Core: heterogeneous pipeline layout](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/pipeline_parallel_layout.html)
- [Megatron Core: TP/PP/EP semantics](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

NVIDIA is migrating model families to **`ModelConfig` and
`ModelBuilder`**. IQ is not an ordinary GPT transformer: each of its 78
blocks combines **native Mamba-3 MIMO and IQ CSA** with sequential dense
or SiLU-gated Stable LatentMoE. A stock GPT layer spec would change its
computation; do not call it a compatible training provider.

## What is implemented

1. `iq_training/hf_model_config.py` defines a real Hugging Face
   `PretrainedConfig` with `model_type="iq_mamba3_csa"`. Its
   `iq_hybrid` field is the **full existing IQHybridConfig**. It
   validates the fingerprint plus flat vocab/hidden/head fields. Use
   `register_iq_hf_config()` before `AutoConfig.from_pretrained` in
   a fresh process. This **does not** register a fake HF causal LM class.
2. `iq_training/mcore_builder.py` implements NVIDIA's actual
   `ModelConfig`, `ModelBuilder` and `MegatronModule` contracts.
   Its `build_model` constructs **the existing IQHybridForCausalLM**,
   not a speculative GPT reimplementation. Model config sets the real
   78-layer widths/256 MoE experts/top-8 EP topology. It obtains the
   **actual** `ProcessGroupCollection.ep`, and validates its size.
3. This first runtime supports **TP=1 and PP=1 only**, with native
   expert-parallel execution from the existing `ExpertParallelStableLatentMoE`.
   Unsupported TP/PP/VPP, DDP/FSDP/wrappers and missing CUDA fail
   **before** model allocation. IQ's own expert-parallel gradient
   synchronization and native `torch_dist` checkpoint are still
   separate; do not call `bridge.get_model(load_weights=True)`.
4. `iq_training/pipeline_contract.py` maps 78 global composite blocks
   to caller-supplied contiguous PP stages and a documented Megatron Core
   `PipelineParallelLayerLayout`. It rejects a PP boundary when a
   **DSA IndexShare full-indexer** and its consumer layers would be
   separated. It identifies cross-stage **Block AttnRes** as an
   independent unsatisfied dependency.
5. `scripts/plan_iq_megatron_model.py` exports the actual IQ HF
   configuration, deterministic fingerprint and exact MCore stage
   layout, including explicit blockers, on an ordinary CPU.
6. CI verifies the current official `megatron-core==0.19.0`
   `ModelBuilder` API and pipeline layout parser, plus Transformers
   config and fail-closed behavior. A CPU test **does not** instantiate
   Mamba-3 CUDA weights or prove model conversion.

## Execute the config/partition validation today, without GPU

```bash
python -m scripts.plan_iq_megatron_model \
  --ep-size 8 --pp-cuts 0,6,10,78 \
  --output /tmp/iq-megatron-contract
```

The generated `hf/config.json` contains the actual canonical
`IQHybridConfig` and its fingerprint.
`pipeline-contract.json` reports all global layer owners,
dense/MoE/indexer blocks, validated boundary cuts, proposed MCore
layer list, and the **blocking runtime requirements**. An output
directory is never overwritten.

A cut `0,3,78` fails because GLM's layer-2 DSA top-k source is needed
by layers 3–5. `0,6,10,78` is a *legal ownership plan*, not a
functioning PP model: IQ's Block AttnRes, pipeline activations,
`set_input_tensor`, sharded optimizer and checkpoint state still need
implementation.

## Instantiate IQ's actual native builder (CUDA only)

Only on an NVIDIA CUDA worker with initialized model-parallel process
groups and enough memory for the **complete** TP1/PP1 IQ backbone:

```python
from iq_training.mcore_builder import IQMegatronModelConfig
from megatron.core.process_groups_config import ProcessGroupCollection

cfg = IQMegatronModelConfig(
    iq_hybrid=...,  # exact verified IQHybridConfig.to_dict(), not a GLM config
    expert_model_parallel_size=8,
)
builder = cfg.get_builder_cls()(cfg)
models = builder.build_distributed_models(
    pg_collection=your_initialized_pg_collection,
    wrap_with_ddp=False, mixed_precision_wrapper=None,
)
```

This returns a real NVIDIA `MegatronModule` around IQ's own graph.
**No automatic HF weight transplantation** occurs. If the graph cannot
fit, fails CUDA Mamba-3 operator requirements, or process groups are
absent, initialization fails. Do not use this on the CPU laptop.

## Gaps that must remain explicit before full Megatron Bridge support

- The IQ HF wrapper model class and registered bidirectional
  `MegatronModelBridge`/`MegatronMappingRegistry` are **not
  complete**. Source GLM/Nemotron donor mappings involve learned
  6144→4096 transformations: using stock `AutoMapping` would claim
  weight equivalence that has not been demonstrated.
- The custom fused block does not implement the pipeline
  `set_input_tensor` and PP stage ownership protocol. Global
  `BlockAttentionResidual` must preserve its recurrence across
  stages; IndexShare state must travel safely or stages must own
  whole indexer groups.
- Tensor-parallel sharding of native Mamba-3, IQ CSA, fusion and other
  weights is not implemented. The current EP-only grouped gradient and
  checkpoint adapters must be extended to TP/PP/DP.
- Full 78-stage pretrained conversion, tested GLM attention distillation,
  post-restart convergence and independent teacher/student evaluation
  have not been executed. No completed pretrained IQ checkpoint exists.

**The model-identity/builder work is real and testable, but this is not yet
NVIDIA's complete convert/train/resume bridge workflow.**
