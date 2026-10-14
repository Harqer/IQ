"""Synchronize replicated (non-routed-expert) gradients over the IQ EP group.

NVIDIA Megatron Core distinguishes dense/non-expert parameter replicas from
routed experts. This helper handles the EP-only IQ torch.nn.Module path;
do not combine with an existing DDP/MCore gradient-reduction pass or use
for unsupported TP/PP/DP configurations.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
from torch import nn

from iq_model.mlp.expert_parallel import ExpertParallelStableLatentMoE


class IQGradientSyncError(RuntimeError):
    pass


def reduce_replicated_ep_gradients(
    model: nn.Module, *,
    ep_group: dist.ProcessGroup,
    reduction: Literal["sum", "mean"],
) -> tuple[str, ...]:
    """Reduce all replicated gradients, skipping every local routed expert.

    Caller must already scale each rank's loss correctly. 'sum' assumes
    tokens were weighted using global normalization; 'mean' averages the
    rank-level gradients, suitable only for equal local token weighting.
    """
    if reduction not in ("sum", "mean"):
        raise IQGradientSyncError("explicit reduction='sum' or 'mean' required")
    if not dist.is_initialized():
        raise IQGradientSyncError("expert-parallel distributed group not initialized")
    size = dist.get_world_size(group=ep_group)
    if size < 2:
        raise IQGradientSyncError("replicated gradient reduction needs EP >1")
    local_prefixes: list[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, ExpertParallelStableLatentMoE):
            if module.ep_group is not ep_group:
                raise IQGradientSyncError("module expert group does not match reduction group")
            local_prefixes.append(module_name + ".routed_experts.")
    if not local_prefixes:
        raise IQGradientSyncError("no sharded expert modules in IQ model")
    shared = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(tuple(local_prefixes))
    ]
    shared.sort(key=lambda pair: pair[0])
    metadata = [(name, tuple(p.shape), str(p.dtype)) for name, p in shared]
    rank_meta: list[object] = [None] * size
    dist.all_gather_object(rank_meta, metadata, group=ep_group)
    if any(part != metadata for part in rank_meta):
        raise IQGradientSyncError("replicated parameter inventory differs among EP ranks")

    # For parameters not used in any rank's graph, preserve grad=None so
    # AdamW doesn't spuriously initialize its moments or decay those weights.
    local_present = [p.grad is not None for _, p in shared]
    gathered: list[object] = [None] * size
    dist.all_gather_object(gathered, local_present, group=ep_group)
    if any(not isinstance(flags, list) or len(flags) != len(shared) for flags in gathered):
        raise IQGradientSyncError("EP gradient presence metadata is malformed")
    active = [
        any(flags[index] for flags in gathered)
        for index in range(len(shared))
    ]
    for (name, parameter), needs_reduce in zip(shared, active):
        if not needs_reduce:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise IQGradientSyncError(f"sparse replicated gradient unsupported: {name}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=ep_group)
        if reduction == "mean":
            parameter.grad.div_(size)
    return tuple(name for (name, _), enabled in zip(shared, active) if enabled)
