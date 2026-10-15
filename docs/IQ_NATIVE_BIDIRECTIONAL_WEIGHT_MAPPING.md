# Reversible IQ-native weight mappings: real NVIDIA registry, strict scope

Official references:
- [NVIDIA Megatron Bridge technical conversion details](https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-tech-details.html)
- [NVIDIA contributing a model](https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html)
- [NVIDIA `MegatronMappingRegistry` source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/mapping_registry.py)
- [NVIDIA `DirectMapping` source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/param_mapping.py)

## Implemented

`iq_training/native_weight_registry.py` creates a deterministic global
source/target tensor inventory from the **actual** PyTorch modules, covering
`named_parameters(remove_duplicate=False)` and `named_buffers`. It
validates **all** names, shapes, dtypes and parameter/buffer kinds.

For the same unsharded IQ architecture:
- Hugging Face model wrapper key: `model.layers.6.attention.<tensor>`
- Megatron module wrapper key: `iq.layers.6.attention.<tensor>`

The only difference is the wrapper prefix; `DirectMapping` is appropriate.
The resulting object is NVIDIA's **actual**
`MegatronMappingRegistry`, with a separate registered `DirectMapping`
for each verified tensor and both source/reverse lookups exercised.

The default production preflight also verifies all 78 IQ composite stages
through `plan_global_inventory`, including 3 dense and 75 MoE stages,
and compares exact model fingerprints. No `allow_missing_keys`.

`iq_training/native_weight_transfer.py` performs actual in-place tensor
copies using the verified mapping plan, validates the source and target
tensors are materialized, and checks bitwise value equality after each copy.
It accepts a direction `hf_to_megatron` or `megatron_to_hf`.

**This is IQ-native weight transfer only**, which is reversible at
**TP=PP=EP=1**. It does NOT merge GLM-5.3, Nemotron, Mamba-3 donors
or translate source vocabularies. It also does NOT yet serialize a complete
new Hugging Face IQ model checkpoint, support optimizer states, or register
a full `MegatronModelBridge` for AutoBridge discovery. A source IQ HF
model class is a separately required integration milestone.

## CPU acceptance

The CPU unit tests use *real* PyTorch parameters and buffers (but tiny,
non-pretrained weights), verify missing/shape-mismatched tensors fail,
verify full-78 model coverage is required in production mode, and execute
actual import/export steps through the NVIDIA registry.

A dedicated CI job pins `megatron-bridge==0.6.2` and
`megatron-core==0.19.0`, ensuring registry behavior is validated against
the installed NVIDIA packages, not stand-in implementations.

```bash
python -m unittest discover -s tests -p "test_iq_native_weight_registry.py" -v
```

The ordinary IQ CPU tests do not require NVIDIA's Bridge installation;
the dedicated `native-bridge-weight-registry` job does.

## Remaining hard blockers

1. A complete HF `PreTrainedModel` wrapper for IQ's 78-block CUDA
   architecture and a source-native checkpoint format with strict
   load/save semantics.
2. Register a fully functioning `MegatronModelBridge` source→target
   mapping and test supported `AutoBridge` import/export. Registry
   lookup tests alone are **not** end-to-end Bridge conversion.
3. Global expert weights need source global-expert ID→EP shard-index
   mappings, plus TP/PP sharding and gradient/optimizer checkpoint
   contracts; the current direct registry refuses every sharded topology.
4. Cross-model GLM-5.3→IQ transfers need *non-identity* calibrated
   transformations, attention distillation, tokenizer realignment and
   Mamba-3-specific operator training. They are NOT bidirectionally
   reversible pretrained transfers.
5. Actual donor tensor provenance, GPU training and full-model recovery
   need external model data and validation. None have been executed here.

No bucket objects, model donor files or user billing resources were touched.
