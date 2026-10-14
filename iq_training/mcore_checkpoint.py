"""IQ-native *real* Megatron Core torch_dist training checkpoint adapter.

Checkpointing follows NVIDIA's ShardedTensor/from_rank_offsets, ShardedObject,
dist_checkpointing.save/load APIs. This is NOT the Megatron Bridge training
loop: IQ is a custom torch.nn.Module, not a Megatron GPTModel. Only same-size
expert-parallel (EP) topology with world_size == EP is supported. No TP/PP/DP
resharding, no zero optimizer and no missing optimizer states are silently
accepted. Strict post-save restart uses the same exact model/optimizer graph.

Official API references:
https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.dist_checkpointing.mapping.html
https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.dist_checkpointing.serialization.html
https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import random
import re
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from iq_model.mlp.expert_parallel import ExpertParallelStableLatentMoE


class IQDistributedCheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class IQCheckpointProgress:
    iteration: int
    consumed_samples: int
    run_id: str
    config_fingerprint: str

    def __post_init__(self):
        if self.iteration < 0 or self.consumed_samples < 0:
            raise IQDistributedCheckpointError("training progress must be non-negative")
        if not self.run_id or not self.config_fingerprint:
            raise IQDistributedCheckpointError("run ID and IQ config fingerprint are required")


def _core():
    # NVIDIA MCore 0.19's torch_dist writer unconditionally synchronizes CUDA
    # while preparing checkpoint tensors. CPU-only runners cannot exercise
    # this native path; do not silently fall back to torch.save().
    if not torch.cuda.is_available():
        raise IQDistributedCheckpointError(
            "NVIDIA Megatron Core 0.19 torch_dist checkpointing requires "
            "an actual CUDA driver/GPU; run the native GPU acceptance job"
        )
    try:
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    except ImportError as exc:
        raise IQDistributedCheckpointError(
            "megatron-core is required for real torch_dist checkpointing"
        ) from exc
    return dist_checkpointing, ShardedObject, ShardedTensor


def _ep_world(model: nn.Module) -> tuple[int, int, ExpertParallelStableLatentMoE]:
    if not dist.is_available() or not dist.is_initialized():
        raise IQDistributedCheckpointError("distributed process group must be initialized")
    ep_modules = [m for m in model.modules() if isinstance(m, ExpertParallelStableLatentMoE)]
    if not ep_modules:
        raise IQDistributedCheckpointError("no genuine IQ EP-sharded expert module found")
    first = ep_modules[0]
    world, rank = dist.get_world_size(), dist.get_rank()
    if world != first.ep_size or rank != first.ep_rank:
        raise IQDistributedCheckpointError(
            "checkpointing presently requires world_size == expert_parallel_size, "
            "and EP group == default WORLD; TP/PP/DP checkpointing is not implemented"
        )
    for layer in ep_modules:
        if layer.ep_size != world or layer.ep_rank != rank:
            raise IQDistributedCheckpointError("inconsistent EP group membership")
    return rank, world, first


def _canonical_names(model: nn.Module, *, rank: int) -> dict[str, str]:
    """Map only local routed expert positions to immutable global expert IDs."""
    replacements: dict[str, str] = {}
    for prefix, module in model.named_modules():
        if not isinstance(module, ExpertParallelStableLatentMoE):
            continue
        for local in range(module.local_experts):
            local_prefix = f"{prefix}.routed_experts.{local}."
            global_prefix = f"{prefix}.routed_experts.{module.global_expert_id(local)}."
            for name in model.state_dict():
                if name.startswith(local_prefix):
                    if name in replacements:
                        raise IQDistributedCheckpointError("duplicate local routed expert")
                    replacements[name] = global_prefix + name[len(local_prefix):]
    mapped = {
        name: replacements.get(name, name)
        for name in model.state_dict()
    }
    if len(set(mapped.values())) != len(mapped):
        raise IQDistributedCheckpointError("global expert naming collided")
    if not replacements:
        raise IQDistributedCheckpointError("no EP-sharded expert parameters discovered")
    return mapped


def _sharded_state(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    progress: IQCheckpointProgress, *, scheduler: Any,
    dataloader_state: Mapping[str, Any],
    load_template: bool = False,
    saved_tensor_keys: set[str] | None = None,
) -> dict[str, Any]:
    _, ShardedObject, ShardedTensor = _core()
    rank, world, _ = _ep_world(model)
    if type(optimizer) is not torch.optim.AdamW:
        raise IQDistributedCheckpointError(
            "exact AdamW optimizer is currently the only resumable IQ optimizer"
        )
    if scheduler is None or not hasattr(scheduler, "state_dict") or not hasattr(scheduler, "load_state_dict"):
        raise IQDistributedCheckpointError("training checkpoint requires a scheduler")
    if not isinstance(dataloader_state, Mapping):
        raise IQDistributedCheckpointError("training checkpoint requires dataloader state")
    model_config = getattr(model, "config", None)
    if model_config is not None:
        actual_fingerprint = getattr(model_config, "fingerprint", None)
        if actual_fingerprint != progress.config_fingerprint:
            raise IQDistributedCheckpointError(
                "checkpoint target fingerprint differs from actual IQ model config"
            )
    mapped = _canonical_names(model, rank=rank)
    state = model.state_dict()
    pieces: dict[str, Any] = {"model": {}, "optimizer": {}}
    for local_name, tensor in sorted(state.items()):
        global_name = mapped[local_name]
        owner_only = global_name != local_name
        pieces["model"][local_name] = ShardedTensor.from_rank_offsets(
            "iq.model." + global_name, tensor.detach().contiguous(),
            replica_id=0 if owner_only else rank,
        )

    local_params = dict(model.named_parameters())
    if set(local_params) - set(mapped):
        raise IQDistributedCheckpointError("unmapped trainable parameters")
    param_ids = {id(param): local_name for local_name, param in local_params.items()}
    param_groups: list[dict[str, Any]] = []
    steps: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        if group.get("capturable", False) or group.get("fused", False):
            raise IQDistributedCheckpointError(
                "capturable/fused AdamW requires separately verified optimizer checkpointing"
            )
        names = []
        for parameter in group["params"]:
            local_name = param_ids.get(id(parameter))
            if local_name is None:
                raise IQDistributedCheckpointError("AdamW has a parameter not owned by IQ model")
            names.append(local_name)
            state_values = optimizer.state.get(parameter)
            if state_values and set(state_values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise IQDistributedCheckpointError(
                    f"unsupported AdamW state structure for {local_name}"
                )
            if state_values:
                step = state_values["step"]
                if not isinstance(step, torch.Tensor) or step.numel() != 1:
                    raise IQDistributedCheckpointError("AdamW step must be scalar tensor")
                steps[local_name] = step.detach().cpu().clone()
            canonical = mapped[local_name]
            requested = [
                moment for moment in ("exp_avg", "exp_avg_sq")
                if (
                    (load_template and saved_tensor_keys is not None and
                     f"iq.optimizer.{canonical}.{moment}" in saved_tensor_keys)
                    or (not load_template and bool(state_values))
                )
            ]
            if load_template and len(requested) not in (0, 2):
                raise IQDistributedCheckpointError("checkpoint has incomplete AdamW moments")
            for moment in requested:
                data = (
                    torch.empty_like(parameter, memory_format=torch.preserve_format)
                    if load_template else state_values[moment]
                )
                if not isinstance(data, torch.Tensor) or data.shape != parameter.shape:
                    raise IQDistributedCheckpointError("invalid AdamW moment shape")
                pieces["optimizer"][f"{local_name}.{moment}"] = (
                    ShardedTensor.from_rank_offsets(
                        f"iq.optimizer.{canonical}.{moment}",
                        data.detach().contiguous(),
                        replica_id=0 if canonical != local_name else rank,
                    )
                )
        if not names:
            raise IQDistributedCheckpointError("empty optimizer parameter group")
        param_groups.append({
            "names": names,
            "options": {key: value for key, value in group.items() if key != "params"},
        })

    payload = {
        "progress": {
            "iteration": progress.iteration,
            "consumed_samples": progress.consumed_samples,
            "run_id": progress.run_id,
            "config_fingerprint": progress.config_fingerprint,
        },
        "optimizer_steps": steps,
        "param_groups": param_groups,
        "scheduler": scheduler.state_dict(),
        "dataloader": dict(dataloader_state),
        "rng_torch": torch.get_rng_state(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "rank": rank,
        "world": world,
        "name_map": mapped,
    }
    pieces["per_rank_state"] = ShardedObject(
        "iq.training_state", payload,
        global_shape=(world,), global_offset=(rank,)
    )
    return pieces


def save_training_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, *,
    scheduler: Any, dataloader_state: Mapping[str, Any],
    progress: IQCheckpointProgress, base_dir: str | Path,
) -> Path:
    """Save a fully resumable *same-EP-world* IQ torch_dist checkpoint.

    A new iter directory and atomic top-level tracker are written. The tracker
    is advanced on rank zero ONLY after all ranks complete MCore serialization.
    """
    dcp, _, _ = _core()
    rank, world, _ = _ep_world(model)
    payload = _sharded_state(
        model, optimizer, progress, scheduler=scheduler,
        dataloader_state=dataloader_state
    )
    base = Path(base_dir).resolve()
    iteration = base / f"iter_{progress.iteration:07d}"
    # NVIDIA MCore serializer expects a *pre-existing empty directory*.
    # Every rank must agree on existence BEFORE rank zero creates it.
    exists = torch.tensor(int(iteration.exists()), dtype=torch.int32)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise IQDistributedCheckpointError(
            "refusing to overwrite existing checkpoint iteration"
        )
    if rank == 0:
        base.mkdir(parents=True, exist_ok=True)
        iteration.mkdir(mode=0o700, exist_ok=False)
    dist.barrier()
    dcp.save(
        payload, str(iteration), verify_integrity=True,
        content_metadata={
            "format": "iq_mcore_torch_dist_v1",
            "ep_size": world,
            "config_fingerprint": progress.config_fingerprint,
        },
    )
    dist.barrier()
    if rank == 0:
        tracker = base / "iq_latest.json"
        if tracker.exists():
            previous = json.loads(tracker.read_text(encoding="utf-8"))
            if int(previous["iteration"]) >= progress.iteration:
                raise IQDistributedCheckpointError("checkpoint iteration must increase")
        tmp = base / ".iq_latest.json.tmp"
        tmp.write_text(json.dumps({
            "format": "iq_mcore_torch_dist_v1",
            "iteration": progress.iteration,
            "ep_size": world,
            "config_fingerprint": progress.config_fingerprint,
        }, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, tracker)
    dist.barrier()
    return iteration


def load_training_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, *,
    scheduler: Any, base_dir: str | Path,
    expected_run_id: str, expected_fingerprint: str,
    iteration: int | None = None,
) -> tuple[IQCheckpointProgress, dict[str, Any]]:
    """Strictly load model, AdamW optimizer, scheduler and per-rank RNG/data.

    Caller must pass returned dataloader state into its real resumable
    dataloader before processing any new batch.
    """
    dcp, _, _ = _core()
    rank, world, _ = _ep_world(model)
    base = Path(base_dir).resolve()
    tracker_path = base / "iq_latest.json"
    if iteration is None:
        if not tracker_path.is_file():
            raise IQDistributedCheckpointError("missing IQ checkpoint tracker")
        tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
        if tracker["format"] != "iq_mcore_torch_dist_v1" or tracker["ep_size"] != world:
            raise IQDistributedCheckpointError("IQ checkpoint format or EP world mismatched")
        if tracker["config_fingerprint"] != expected_fingerprint:
            raise IQDistributedCheckpointError("IQ checkpoint target config changed")
        iteration = int(tracker["iteration"])
    if iteration < 0:
        raise IQDistributedCheckpointError("invalid checkpoint iteration")
    directory = base / f"iter_{iteration:07d}"
    if not directory.is_dir():
        raise IQDistributedCheckpointError("requested checkpoint iteration does not exist")

    # Populate guidance using real local model/optimizer memory before loading.
    candidate = IQCheckpointProgress(iteration, 0, expected_run_id, expected_fingerprint)
    saved_tensor_keys = set(dcp.load_tensors_metadata(str(directory)))
    template = _sharded_state(
        model, optimizer, candidate, scheduler=scheduler,
        dataloader_state={}, load_template=True,
        saved_tensor_keys=saved_tensor_keys,
    )
    metadata = dcp.load_content_metadata(str(directory))
    if not isinstance(metadata, dict) or (
        metadata.get("format") != "iq_mcore_torch_dist_v1"
        or metadata.get("ep_size") != world
        or metadata.get("config_fingerprint") != expected_fingerprint
    ):
        raise IQDistributedCheckpointError("unsupported IQ MCore checkpoint metadata")
    restored = dcp.load(
        template, str(directory), verify_integrity=True,
        strict="raise_all",
    )
    state = restored["per_rank_state"]
    if isinstance(state, dict):
        payload = state
    elif hasattr(state, "data"):
        payload = state.data
    else:
        raise IQDistributedCheckpointError("invalid per-rank checkpoint state")
    saved = payload["progress"]
    if (
        saved["iteration"] != iteration
        or saved["run_id"] != expected_run_id
        or saved["config_fingerprint"] != expected_fingerprint
        or payload["rank"] != rank or payload["world"] != world
    ):
        raise IQDistributedCheckpointError("checkpoint progress/rank provenance mismatch")

    if payload["name_map"] != _canonical_names(model, rank=rank):
        raise IQDistributedCheckpointError("local-to-global expert parameter topology changed")
    local_params = dict(model.named_parameters())
    expected_groups = [
        [next((n for n, p in local_params.items() if p is param), None)
         for param in group["params"]]
        for group in optimizer.param_groups
    ]
    if expected_groups != [entry["names"] for entry in payload["param_groups"]]:
        raise IQDistributedCheckpointError("optimizer parameter group topology changed")

    # Build all new states before model/optimizer mutation.
    model_state = {}
    for name, entry in restored["model"].items():
        tensor = entry.data if hasattr(entry, "data") else entry
        if tuple(tensor.shape) != tuple(model.state_dict()[name].shape):
            raise IQDistributedCheckpointError("checkpoint model tensor shape changed")
        model_state[name] = tensor
    optimizer_states = {}
    for name, parameter in local_params.items():
        step = payload["optimizer_steps"].get(name)
        moment_keys = [f"{name}.{key}" for key in ("exp_avg", "exp_avg_sq")]
        present = [key in restored["optimizer"] for key in moment_keys]
        if present == [False, False] and step is None:
            # AdamW lazily initializes never-selected expert weights.
            continue
        if present != [True, True] or step is None:
            raise IQDistributedCheckpointError(
                f"AdamW step/moments incomplete for {name}"
            )
        state = {"step": step}
        for moment, key in zip(("exp_avg", "exp_avg_sq"), moment_keys):
            value = restored["optimizer"][key]
            tensor = value.data if hasattr(value, "data") else value
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise IQDistributedCheckpointError("AdamW optimizer shard shape mismatch")
            state[moment] = tensor.to(parameter.device)
        optimizer_states[parameter] = state
    scheduler_state = payload["scheduler"]
    if not isinstance(payload.get("dataloader"), dict):
        raise IQDistributedCheckpointError("checkpoint missing dataloader state")
    model.load_state_dict(model_state, strict=True)
    optimizer.state.clear()
    optimizer.state.update(optimizer_states)
    for group, item in zip(optimizer.param_groups, payload["param_groups"]):
        group.update(item["options"])
    scheduler.load_state_dict(scheduler_state)
    torch.set_rng_state(payload["rng_torch"])
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    if torch.cuda.is_available() and payload["rng_cuda"]:
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    dist.barrier()
    return (
        IQCheckpointProgress(**saved),
        dict(payload["dataloader"])
    )
