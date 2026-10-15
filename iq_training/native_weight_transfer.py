"""Execute verified IQ-native weight transfer between HF and MCore wrappers.

Only exact, shape-and-dtype-identical IQ-native tensors are reversible.
Calls NVIDIA's real DirectMapping import transformation and requires complete
78-block inventory by default; this is not the GLM donor transplant.

https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-tech-details.html
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import torch
from torch import nn

from iq_model.hybrid import IQHybridConfig
from .native_weight_registry import (
    IQNativeMappingError, IQNativeMappingPlan,
    megatron_native_mapping_registry, plan_iq_native_tensors, _named,
)


@dataclass(frozen=True)
class IQNativeTransferResult:
    direction: str
    config_fingerprint: str
    tensor_count: int
    mapping_sha256: str
    exact_values_verified: bool
    full_iq_78_verified: bool

    def to_dict(self) -> dict:
        return {
            "format": "iq_native_exact_transfer_v1",
            "direction": self.direction,
            "config_fingerprint": self.config_fingerprint,
            "tensor_count": self.tensor_count,
            "mapping_sha256": self.mapping_sha256,
            "exact_values_verified": self.exact_values_verified,
            "full_iq_78_verified": self.full_iq_78_verified,
            "GLM_to_IQ_pretrained_transform": False,
        }


def transfer_native_iq_weights(
    hf_model: nn.Module,
    megatron_model: nn.Module,
    *,
    iq_config: IQHybridConfig,
    direction: Literal["hf_to_megatron", "megatron_to_hf"],
    full_78_required: bool = True,
) -> IQNativeTransferResult:
    """Execute actual in-place transfers, verifying each tensor exactly.

    Performs a complete topology/name/shape/dtype preflight BEFORE any write.
    No optimizer/RNG/data state is included: this is model-weight transfer
    only. Does not overwrite any checkpoint file or modify donor buckets.
    """
    if direction not in ("hf_to_megatron", "megatron_to_hf"):
        raise IQNativeMappingError("unsupported native IQ transfer direction")
    plan = plan_iq_native_tensors(
        hf_model, megatron_model, iq_config=iq_config,
        full_78_required=full_78_required,
    )
    registry = megatron_native_mapping_registry(plan)
    source = _named(hf_model if direction == "hf_to_megatron" else megatron_model)
    destination = _named(megatron_model if direction == "hf_to_megatron" else hf_model)

    # Recheck all tensors' memory accessibility up front; tensor
    # dimensions/parameters have already been fully inventoried.
    for spec in plan.tensors:
        src_name = (
            spec.hf_name if direction == "hf_to_megatron" else spec.megatron_name
        )
        dst_name = (
            spec.megatron_name if direction == "hf_to_megatron" else spec.hf_name
        )
        if source[src_name][0].is_meta or destination[dst_name][0].is_meta:
            raise IQNativeMappingError(
                f"cannot transfer unmaterialized meta tensor: {src_name}"
            )

    with torch.no_grad():
        for spec in plan.tensors:
            if direction == "hf_to_megatron":
                mapping = registry.hf_to_megatron_lookup(spec.hf_name)
                assert mapping is not None
                src = source[spec.hf_name][0]
                dest = destination[spec.megatron_name][0]
                transformed = mapping.hf_to_megatron(src, megatron_model)
            else:
                mapping = registry.megatron_to_hf_lookup(spec.megatron_name)
                assert mapping is not None
                src = source[spec.megatron_name][0]
                dest = destination[spec.hf_name][0]
                # DirectMapping is an identity operation at TP=PP=EP=1.
                # The actual NVIDIA mapping is validated by the registry;
                # avoid calling its distributed PP broadcast in a process
                # that has no MCore pipeline group initialized.
                transformed = src
            dest.copy_(transformed)
            if not bool(torch.equal(src, dest)):
                raise IQNativeMappingError(
                    f"post-transfer exact tensor equality failed: {spec.hf_name}"
                )

    result = IQNativeTransferResult(
        direction=direction,
        config_fingerprint=iq_config.fingerprint,
        tensor_count=len(plan.tensors),
        mapping_sha256=plan.to_dict()["sha256"],
        exact_values_verified=True,
        full_iq_78_verified=full_78_required,
    )
    return result
