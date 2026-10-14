"""Deterministic, rank-safe IQ 78-block transfer inventory.

This is a metadata-level preflight, NOT a weight-conversion implementation.
No tensor values or fake source weights are generated. The actual NVIDIA
GLM5Bridge owns source checkpoint import/shard collectives. IQ-specific
tensor mappings remain subject to operator alignment and output parity.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Iterable, Literal
import json
import re

from iq_model.hybrid import IQHybridConfig

_NAME = re.compile(r"^layers\.(\d+)\.([^.]++)(?:\.|$)")


class ParallelMappingError(ValueError):
    pass


@dataclass(frozen=True)
class ParameterRecord:
    global_name: str
    shape: tuple[int, ...]
    dtype: str
    kind: Literal["parameter", "buffer"]
    pp_rank: int
    tp_rank: int
    ep_rank: int

    def __post_init__(self):
        if not self.global_name or any(x <= 0 for x in self.shape):
            raise ParallelMappingError("parameter name/shape must be valid")
        if self.kind not in ("parameter", "buffer"):
            raise ParallelMappingError("inventory entry must be parameter or buffer")
        if any(r < 0 for r in (self.pp_rank, self.tp_rank, self.ep_rank)):
            raise ParallelMappingError("parallelism ranks cannot be negative")


def collect_local_inventory(model, *, pp_rank: int = 0, tp_rank: int = 0,
                            ep_rank: int = 0,
                            local_to_global: dict[int, int] | None = None
                            ) -> list[ParameterRecord]:
    """Collect real named_parameters/buffers metadata, not weight tensors."""
    result = []
    for kind, iterator in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in iterator:
            match = _NAME.match(name)
            if match and local_to_global is not None:
                local = int(match.group(1))
                if local not in local_to_global:
                    raise ParallelMappingError(f"missing global PP index for {name}")
                name = "layers." + str(local_to_global[local]) + name[match.end(1):]
            elif match and local_to_global is None and pp_rank != 0:
                raise ParallelMappingError(
                    "nonzero PP ranks require explicit local_to_global layer indices"
                )
            result.append(ParameterRecord(
                global_name=name, shape=tuple(tensor.shape),
                dtype=str(tensor.dtype), kind=kind,
                pp_rank=pp_rank, tp_rank=tp_rank, ep_rank=ep_rank,
            ))
    return sorted(result, key=lambda x: (
        x.global_name, x.kind, x.pp_rank, x.tp_rank, x.ep_rank
    ))


def gather_rank_metadata(local: list[ParameterRecord], *, process_group=None
                         ) -> list[ParameterRecord]:
    """Collect metadata only; every participating rank receives same global order."""
    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise ParallelMappingError("torch.distributed is required") from exc
    if not dist.is_available() or not dist.is_initialized():
        if any(r.pp_rank or r.tp_rank or r.ep_rank for r in local):
            raise ParallelMappingError(
                "cannot gather multiple ranks without initialized torch.distributed"
            )
        return sorted(local, key=lambda x: (
            x.global_name, x.kind, x.pp_rank, x.tp_rank, x.ep_rank
        ))
    world_size = dist.get_world_size(group=process_group)
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local, group=process_group)
    if any(not isinstance(part, list) for part in gathered):
        raise ParallelMappingError("distributed parameter metadata is malformed")
    combined: list[ParameterRecord] = []
    for part in gathered:
        for item in part:
            if not isinstance(item, ParameterRecord):
                raise ParallelMappingError("non-parameter metadata in distributed gather")
            combined.append(item)
    return sorted(combined, key=lambda x: (
        x.global_name, x.kind, x.pp_rank, x.tp_rank, x.ep_rank
    ))


def mapping_kind(name: str, config: IQHybridConfig) -> tuple[str, str]:
    """Return source and exact required transfer method, never a silent copy."""
    match = _NAME.match(name)
    if match:
        layer = int(match.group(1))
        if layer >= len(config.schedule.layers):
            raise ParallelMappingError(f"parameter targets nonexistent IQ layer {layer}")
        group = match.group(2)
        if group == "mamba":
            return ("mamba3-mimo-1.5b", "verified_24_to_78_native_overlay")
        if group == "attention":
            return ("glm53", "MLA_DSA_coordinate_transport_and_indexshare_alignment")
        if group == "dense_ffn":
            if layer not in config.parallel_dense_layers:
                raise ParallelMappingError("dense tensor appears outside first three stages")
            return ("glm53", "dense_SwiGLU_6144_to_4096_transport")
        if group == "moe":
            if layer in config.parallel_dense_layers:
                raise ParallelMappingError("MoE tensor appears in GLM dense stage")
            return ("glm53", "full_256_expert_and_router_transport")
        if group in ("fusion", "expert_gate"):
            return ("iq", "new_trainable_residual_neutral_initialization")
        if group == "norm":
            return ("iq", "pretrained_Mamba_norm_or_aligned_norm")
        raise ParallelMappingError(f"unclassified IQ layer parameter: {name}")
    if name.startswith(("embed_tokens.", "lm_head.")):
        return ("nemotron3-super", "lexical_tokenizer_and_weight_tie_validation")
    if name.startswith(("norm.", "attnres.", "reasoning.", "multimodal.",
                        "reasoning_injector.", "energy_critic.")):
        return ("iq", "configuration_owned_or_independently_calibrated")
    raise ParallelMappingError(f"unclassified IQ destination: {name}")


def plan_global_inventory(records: Iterable[ParameterRecord],
                          config: IQHybridConfig) -> dict:
    """Strictly account for complete 78 blocks before serializing conversion tasks."""
    if (
        len(config.schedule.layers) != 78
        or config.parallel_dense_layers != (0, 1, 2)
        or config.stable_moe is None
        or config.stable_moe.num_experts != 256
        or config.stable_moe.top_k != 8
        or len(config.parallel_index_sources) != 78
    ):
        raise ParallelMappingError("destination is not the full GLM 78-block IQ geometry")
    ordered = sorted(records, key=lambda item: (
        item.global_name, item.kind, item.pp_rank, item.tp_rank, item.ep_rank
    ))
    if not ordered:
        raise ParallelMappingError("empty destination parameter inventory")
    if len(set((x.global_name, x.kind, x.pp_rank, x.tp_rank, x.ep_rank)
               for x in ordered)) != len(ordered):
        raise ParallelMappingError("duplicate rank-local parameter or buffer")
    names = {x.global_name for x in ordered}
    families = {i: set() for i in range(78)}
    tasks = []
    for item in ordered:
        source, transform = mapping_kind(item.global_name, config)
        match = _NAME.match(item.global_name)
        if match:
            families[int(match.group(1))].add(match.group(2))
        tasks.append({
            **asdict(item),
            "shape": list(item.shape),
            "source_family": source,
            "transformation": transform,
            "verified": False,
        })
    for layer, seen in families.items():
        expected = {"mamba", "attention", "fusion", "expert_gate", "norm"}
        expected.add("dense_ffn" if layer < 3 else "moe")
        if not expected.issubset(seen):
            raise ParallelMappingError(
                f"IQ block {layer} missing parameter families: {sorted(expected - seen)}"
            )
        if ("moe" in seen) == (layer < 3) or ("dense_ffn" in seen) == (layer >= 3):
            raise ParallelMappingError(f"misplaced dense/MoE parameter at block {layer}")
    payload = {
        "schema_version": 1,
        "target_config_fingerprint": config.fingerprint,
        "destination_parameter_count": len(ordered),
        "source_counts": dict(sorted(Counter(
            task["source_family"] for task in tasks
        ).items())),
        "tasks": tasks,
        "status": "preflight_only_not_weight_converted",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = sha256(raw).hexdigest()
    return payload
