"""Exact 78-block IQ pipeline ownership and MCore layout contract.

NVIDIA source:
https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/pipeline_parallel_layout.html
https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.pipeline_parallel_layer_layout.html

This only maps logical stages; it never claims current IQHybridForCausalLM
implements MegatronModule.set_input_tensor, the pipeline schedule, or
pipeline-state transfers. It refuses hidden IndexShare cross-stage edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from iq_model.hybrid import IQHybridConfig


class IQPipelineContractError(ValueError):
    pass


@dataclass(frozen=True)
class IQPipelineStage:
    pp_rank: int
    start_layer: int
    end_layer: int
    owns_embedding: bool
    owns_output: bool
    dense_blocks: tuple[int, ...]
    moe_blocks: tuple[int, ...]
    mamba3_blocks: tuple[int, ...]
    index_owner_blocks: tuple[int, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(range(self.start_layer, self.end_layer))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pp_rank": self.pp_rank,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "owns_embedding": self.owns_embedding,
            "owns_output": self.owns_output,
            "dense_blocks": list(self.dense_blocks),
            "moe_blocks": list(self.moe_blocks),
            "mamba3_blocks": list(self.mamba3_blocks),
            "index_owner_blocks": list(self.index_owner_blocks),
        }


@dataclass(frozen=True)
class IQPipelinePartition:
    config_fingerprint: str
    stages: tuple[IQPipelineStage, ...]
    blocked_capabilities: tuple[str, ...]

    @property
    def pp_size(self) -> int:
        return len(self.stages)

    def megatron_layer_layout(self) -> list[list[str]]:
        """Documented MCore layout tokens; NOT an executable IQ layer spec."""
        return [
            (["embedding"] if stage.owns_embedding else [])
            + ["decoder"] * len(stage.layers)
            + (["loss"] if stage.owns_output else [])
            for stage in self.stages
        ]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": "iq_mcore_pp_contract_v1",
            "config_fingerprint": self.config_fingerprint,
            "pp_size": self.pp_size,
            "stages": [stage.to_dict() for stage in self.stages],
            "mcore_pipeline_model_parallel_layout": self.megatron_layer_layout(),
            "blocked_capabilities": list(self.blocked_capabilities),
            "executable_as_megatron_bridge_provider": False,
            "contains_model_weights": False,
        }
        data = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        result["sha256"] = sha256(data).hexdigest()
        return result

    def require_mcore_training_ready(self) -> None:
        if self.blocked_capabilities:
            raise IQPipelineContractError(
                "IQ pipeline cannot be executed as a Megatron Bridge model: "
                + ", ".join(self.blocked_capabilities)
            )
        raise IQPipelineContractError(
            "IQ's current builder supports only TP=1/PP=1; a stage layout "
            "alone is not a registered full MegatronModelBridge mapping or "
            "pipeline-capable model"
        )


def partition_iq_78(
    config: IQHybridConfig,
    *,
    boundaries: tuple[int, ...] = (0, 78),
) -> IQPipelinePartition:
    """Statically analyze global 78-block ownership and causal IndexShare edges.

    Boundaries are exact, caller-supplied 0-based cuts. Never infer a
    compute-balanced split without measured activation/FLOP costs.
    """
    schedule = config.schedule
    if (
        len(schedule.layers) != 78
        or len(schedule.parallel_positions) != 78
        or config.parallel_dense_layers != (0, 1, 2)
        or schedule.moe_positions != tuple(range(3, 78))
        or len(config.parallel_index_sources) != 78
        or config.stable_moe is None
        or config.stable_moe.num_experts != 256
        or config.stable_moe.top_k != 8
        or config.mamba3.num_layers != 78
    ):
        raise IQPipelineContractError(
            "pipeline contract requires IQ's complete 78-block GLM recipe"
        )
    if (
        not isinstance(boundaries, tuple)
        or len(boundaries) < 2
        or any(type(i) is not int for i in boundaries)
        or boundaries[0] != 0
        or boundaries[-1] != 78
        or tuple(sorted(set(boundaries))) != boundaries
    ):
        raise IQPipelineContractError(
            "pipeline boundaries must be sorted, unique, start at 0 and end at 78"
        )

    sources = config.parallel_index_sources
    stages = []
    for rank, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end <= begin:
            raise IQPipelineContractError("empty pipeline ranks are not supported for IQ")
        violating = [
            (layer, sources[layer])
            for layer in range(begin, end)
            if sources[layer] < begin or sources[layer] >= end
        ]
        if violating:
            raise IQPipelineContractError(
                f"pipeline rank {rank} breaks DSA IndexShare: "
                f"layer/source pairs {violating[:6]}"
            )
        stages.append(IQPipelineStage(
            pp_rank=rank,
            start_layer=begin, end_layer=end,
            owns_embedding=rank == 0,
            owns_output=rank == len(boundaries) - 2,
            dense_blocks=tuple(i for i in range(begin, end) if i < 3),
            moe_blocks=tuple(i for i in range(begin, end) if i >= 3),
            mamba3_blocks=tuple(range(begin, end)),
            index_owner_blocks=tuple(
                i for i in range(begin, end) if sources[i] == i
            ),
        ))
    all_layers = [
        layer for stage in stages for layer in stage.layers
    ]
    if all_layers != list(range(78)):
        raise IQPipelineContractError("pipeline ownership is incomplete/overlapping")

    blockers = [
        "full MegatronModelBridge bidirectional mapping is not registered",
        "IQ runtime pipeline stage handoff/pre_process/post_process is absent",
        "IQ optimizer/checkpoint TP/PP mapping has not been implemented",
    ]
    if len(stages) > 1:
        if config.attnres is not None:
            blockers.append(
                "Block AttnRes requires cross-stage residual-state communication"
            )
        if config.reasoning is not None:
            blockers.append(
                "reasoning recurrence requires first/last-stage ownership contract"
            )
        if config.multimodal is not None:
            blockers.append(
                "multimodal memory requires explicit stage ownership/communication"
            )
    return IQPipelinePartition(
        config_fingerprint=config.fingerprint,
        stages=tuple(stages), blocked_capabilities=tuple(blockers),
    )
