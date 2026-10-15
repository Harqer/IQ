"""Verified reversible IQ-native HF <-> MCore parameter mapping.

Important distinction:
  * IQ-native -> IQ-native with the SAME architecture and tensor shapes:
    exact, reversible `DirectMapping` is valid at TP=PP=EP=1.
  * GLM-5.3 -> IQ: different width, MLA/CSA and tokenizer; these
    mappings MUST NOT be reused for donor transplantation.

Official NVIDIA MegatronMappingRegistry and DirectMapping:
https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/mapping_registry.py
https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/param_mapping.py
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import torch
from torch import nn

from iq_model.hybrid import IQHybridConfig
from iq_transfer.parallel_parameter_plan import (
    ParameterRecord,
    plan_global_inventory,
)


class IQNativeMappingError(ValueError):
    pass


@dataclass(frozen=True)
class IQNativeTensorSpec:
    """Exact source/target named parameter or buffer correspondence."""

    hf_name: str
    megatron_name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hf_name": self.hf_name,
            "megatron_name": self.megatron_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class IQNativeMappingPlan:
    config_fingerprint: str
    tensors: tuple[IQNativeTensorSpec, ...]
    topology: tuple[int, int, int]
    complete_iq_78_inventory: bool

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "iq_native_reversible_weight_mapping_v1",
            "config_fingerprint": self.config_fingerprint,
            "topology": {
                "tp": self.topology[0], "pp": self.topology[1],
                "ep": self.topology[2],
            },
            "tensor_count": len(self.tensors),
            "tensors": [x.to_dict() for x in self.tensors],
            "complete_iq_78_inventory": self.complete_iq_78_inventory,
            "source_model": "IQHFCausalLM_only",
            "destination_model": "IQMegatronModule.iq",
            "source_is_glm53": False,
            "requires_measured_GLM_coordinate_transform": False,
            "native_weight_transfer_executed": False,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["sha256"] = sha256(raw).hexdigest()
        return payload


def _named(model: nn.Module) -> dict[str, tuple[torch.Tensor, str]]:
    result: dict[str, tuple[torch.Tensor, str]] = {}
    for kind, iterator in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in iterator:
            if not name or name in result:
                raise IQNativeMappingError(f"duplicate or empty {kind} name: {name}")
            if any(d <= 0 for d in tensor.shape):
                raise IQNativeMappingError(f"invalid tensor dimensions: {name}")
            result[name] = (tensor, kind)
    return result


def _normalized_inventory(
    target: Mapping[str, tuple[torch.Tensor, str]],
) -> list[ParameterRecord]:
    return [
        ParameterRecord(
            global_name=name.removeprefix("iq."),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            kind=kind,
            pp_rank=0, tp_rank=0, ep_rank=0,
        )
        for name, (tensor, kind) in sorted(target.items())
    ]


def plan_iq_native_tensors(
    hf_model: nn.Module,
    megatron_model: nn.Module,
    *,
    iq_config: IQHybridConfig,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    full_78_required: bool = True,
) -> IQNativeMappingPlan:
    """Read *actual* named parameters/buffers from both model instances.

    Import/export is only reversible for matching IQ models. Never hide
    unknown or missing parameters with allow_missing_keys or heuristics.
    Use full_78_required=False only in miniature unit tests, not conversion.
    """
    if (tensor_parallel_size, pipeline_parallel_size, expert_parallel_size) != (1, 1, 1):
        raise IQNativeMappingError(
            "direct IQ-native mapping requires TP=PP=EP=1; "
            "distributed expert/global shards need distinct ownership mappings"
        )
    if iq_config.expert_parallel_size != 1:
        raise IQNativeMappingError(
            "IQ config has sharded experts; global HF expert export is unavailable"
        )
    hf = _named(hf_model)
    native = _named(megatron_model)
    if not hf or not native:
        raise IQNativeMappingError("both models must expose real named tensors")
    if any(not key.startswith("model.") for key in hf):
        raise IQNativeMappingError("HF IQ model tensors must start with model.")
    if any(not key.startswith("iq.") for key in native):
        raise IQNativeMappingError("MCore IQ tensors must start with iq.")
    # Require source and target to have precisely the same key set, not just
    # the intersection; a missing Mamba, attention or expert key is fatal.
    source_suffixes = {key.removeprefix("model.") for key in hf}
    target_suffixes = {key.removeprefix("iq.") for key in native}
    missing_native = sorted(source_suffixes - target_suffixes)
    missing_hf = sorted(target_suffixes - source_suffixes)
    if missing_native or missing_hf:
        raise IQNativeMappingError(
            "IQ source/target tensor inventories differ: "
            f"missing native={missing_native[:4]} missing HF={missing_hf[:4]}"
        )
    specs: list[IQNativeTensorSpec] = []
    for suffix in sorted(source_suffixes):
        hf_name, native_name = f"model.{suffix}", f"iq.{suffix}"
        source_tensor, source_kind = hf[hf_name]
        native_tensor, native_kind = native[native_name]
        if (tuple(source_tensor.shape) != tuple(native_tensor.shape)
            or source_tensor.dtype != native_tensor.dtype
            or source_kind != native_kind):
            raise IQNativeMappingError(
                f"non-identical IQ-native tensor {suffix}: "
                f"HF {tuple(source_tensor.shape)}/{source_tensor.dtype}/{source_kind}, "
                f"MCore {tuple(native_tensor.shape)}/{native_tensor.dtype}/{native_kind}"
            )
        specs.append(IQNativeTensorSpec(
            hf_name=hf_name, megatron_name=native_name,
            shape=tuple(native_tensor.shape),
            dtype=str(native_tensor.dtype), kind=native_kind,
        ))
    if full_78_required:
        # Use the pre-existing strict inventory preflight on the *actual*
        # MCore parameters. Every one of the 78 stages must expose its
        # Mamba/attention/fusion/norm and dense-or-MoE parameter families.
        plan_global_inventory(_normalized_inventory(native), iq_config)
        actual_config = getattr(getattr(megatron_model, "iq", None), "config", None)
        if (not isinstance(actual_config, IQHybridConfig)
            or actual_config.fingerprint != iq_config.fingerprint):
            raise IQNativeMappingError("MCore IQ model architecture fingerprint differs")
        hf_hybrid = getattr(getattr(hf_model, "config", None), "validated_iq_hybrid", None)
        if not callable(hf_hybrid) or hf_hybrid().fingerprint != iq_config.fingerprint:
            raise IQNativeMappingError("HF IQ model architecture fingerprint differs")
    return IQNativeMappingPlan(
        config_fingerprint=iq_config.fingerprint,
        tensors=tuple(specs),
        topology=(1, 1, 1),
        complete_iq_78_inventory=full_78_required,
    )


def megatron_native_mapping_registry(plan: IQNativeMappingPlan):
    """Construct NVIDIA's *real* bidirectional registry for verified pairs.

    Only use `DirectMapping` when all names/shapes/dtypes were checked above.
    No floating-point resizing, inferred Mamba2-to-3 transform or CSA rewrite.
    """
    if plan.topology != (1, 1, 1):
        raise IQNativeMappingError("registered DirectMapping requires unsharded IQ weights")
    if not plan.tensors:
        raise IQNativeMappingError("cannot register empty IQ-native mapping")
    try:
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import DirectMapping
    except ImportError as exc:
        raise IQNativeMappingError(
            "install NVIDIA Megatron-Bridge to use its actual mapping registry"
        ) from exc
    registry = MegatronMappingRegistry(*[
        DirectMapping(megatron_param=spec.megatron_name, hf_param=spec.hf_name)
        for spec in plan.tensors
    ])
    for spec in plan.tensors:
        forward = registry.megatron_to_hf_lookup(spec.megatron_name)
        reverse = registry.hf_to_megatron_lookup(spec.hf_name)
        if (forward is None or reverse is None
            or forward.hf_param != spec.hf_name
            or reverse.megatron_param != spec.megatron_name):
            raise IQNativeMappingError(
                f"NVIDIA registry did not resolve both directions: {spec.hf_name}"
            )
    return registry
