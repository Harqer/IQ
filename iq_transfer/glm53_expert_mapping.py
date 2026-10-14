"""Exact GLM5Bridge-HF -> IQ full-78 expert ownership map.

No tensor values and no speculative attention mappings are generated.
These task records describe actual source/destination names and ownership,
not numerical equivalence. Use the existing calibrated expert converter.

NVIDIA source:
https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Iterator



@dataclass(frozen=True)
class GLM53TensorTransferTask:
    source: str
    destination: str
    owner: int
    stage: int
    action: str  # transform via calibrated projection, never raw copy


def iter_glm53_expert_transfer_tasks(ep_size: int) -> Iterator[GLM53TensorTransferTask]:
    """Every source expert is owned exactly once; replicated IQ tensors once."""
    from .glm53_expert_shards import expert_range
    expert_range(0, ep_size)  # Validate divisibility and EP size.
    for block in range(78):
        source = f"model.layers.{block}.mlp"
        target = f"layers.{block}"
        if block < 3:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield GLM53TensorTransferTask(
                    f"{source}.{proj}.weight",
                    f"{target}.dense_ffn.1.{proj}.weight",
                    0, block, "dense_coordinate_transport",
                )
            continue
        for src, dest, action in (
            ("gate.weight", "router.weight", "router_coordinate_transport"),
            ("gate.e_score_correction_bias", "routing_bias", "router_bias_calibration"),
            ("shared_experts.gate_proj.weight", "shared_experts.0.gate_proj.weight", "shared_expert_transport"),
            ("shared_experts.up_proj.weight", "shared_experts.0.up_proj.weight", "shared_expert_transport"),
            ("shared_experts.down_proj.weight", "shared_experts.0.down_proj.weight", "shared_expert_transport"),
        ):
            yield GLM53TensorTransferTask(
                f"{source}.{src}", f"{target}.moe.moe.{dest}", 0, block,
                action,
            )
        for ep_rank in range(ep_size):
            owned = expert_range(ep_rank, ep_size)
            for expert in owned:
                local = expert - owned.start
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield GLM53TensorTransferTask(
                        f"{source}.experts.{expert}.{proj}.weight",
                        f"{target}.moe.moe.routed_experts.{local}.{proj}.weight",
                        ep_rank, block, "routed_expert_latent_transport",
                    )


def glm53_expert_transfer_manifest(ep_size: int) -> dict:
    """Deterministic global task order with explicit replicated-vs-sharded keys."""
    tasks = list(iter_glm53_expert_transfer_tasks(ep_size))
    by_source = [t.source for t in tasks]
    if len(by_source) != len(set(by_source)):
        raise ValueError("duplicate GLM-5.3 tensor source")
    # Destination names are rank-local: uniqueness is per (owning rank, key).
    owned_destination = [(t.owner, t.destination) for t in tasks]
    if len(owned_destination) != len(set(owned_destination)):
        raise ValueError("duplicate IQ rank-local destination")
    if sum(t.action == "routed_expert_latent_transport" for t in tasks) != 75 * 256 * 3:
        raise ValueError("GLM full routed-expert count not preserved")
    ordered = sorted(tasks, key=lambda t: (t.source, t.owner, t.destination))
    records = [asdict(item) for item in ordered]
    data = {
        "artifact": "iq_glm53_full_expert_transfer_tasks_v1",
        "stage_count": 78,
        "expert_parallel_size": ep_size,
        "routed_experts": 256,
        "routed_projection_tasks": 75 * 256 * 3,
        "tasks": records,
        "all_actions_require_calibration": True,
        "attention_mapping_complete": False,
        "mamba3_mapping_complete": False,
        "conversion_executed": False,
    }
    digest = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["sha256"] = sha256(digest).hexdigest()
    return data
