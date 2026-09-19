from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .manifest import DonorManifest
from .plan import TransportPlan
from .slots import TargetAssignment, TargetSlot, TransferMethod


class PhiPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhiLayerMapIds:
    attn_in: str
    q: str
    kv: str
    attn_out: str
    mlp_in: str
    mlp_hidden: str
    mlp_out: str

    def __post_init__(self) -> None:
        empty = [name for name, value in self.__dict__.items() if not str(value).strip()]
        if empty:
            raise PhiPipelineError(f"Phi layer map ids must be non-empty: {', '.join(empty)}")


@dataclass(frozen=True)
class PhiDensePlanSpec:
    donor_id: str
    donor_manifest: DonorManifest
    calibration_manifest_hash: str
    recipient_config_hash: str
    recipient_vocab_size: int
    recipient_hidden_size: int
    recipient_intermediate_size: int
    recipient_num_attention_heads: int
    recipient_num_key_value_heads: int
    recipient_head_dim: int
    layer_mapping: tuple[tuple[int, int], ...]
    layer_maps: tuple[tuple[int, PhiLayerMapIds], ...]
    embedding_map_id: str
    final_residual_map_id: str
    plan_id: str = "phi-dense-iq-v1"

    def __post_init__(self) -> None:
        if self.donor_manifest.donor_id != self.donor_id:
            raise PhiPipelineError("donor_id does not match donor manifest")
        required_text = {
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "recipient_config_hash": self.recipient_config_hash,
            "embedding_map_id": self.embedding_map_id,
            "final_residual_map_id": self.final_residual_map_id,
            "plan_id": self.plan_id,
        }
        empty = [name for name, value in required_text.items() if not str(value).strip()]
        if empty:
            raise PhiPipelineError(f"Phi plan fields must be non-empty: {', '.join(empty)}")
        ints = {
            "recipient_vocab_size": self.recipient_vocab_size,
            "recipient_hidden_size": self.recipient_hidden_size,
            "recipient_intermediate_size": self.recipient_intermediate_size,
            "recipient_num_attention_heads": self.recipient_num_attention_heads,
            "recipient_num_key_value_heads": self.recipient_num_key_value_heads,
            "recipient_head_dim": self.recipient_head_dim,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise PhiPipelineError(f"positive recipient dimensions required: {', '.join(bad)}")
        if self.recipient_num_attention_heads * self.recipient_head_dim != self.recipient_hidden_size:
            raise PhiPipelineError("recipient Q projection width must equal hidden_size")
        if self.recipient_num_attention_heads % self.recipient_num_key_value_heads != 0:
            raise PhiPipelineError("recipient Q heads must be divisible by KV heads")

        mapping = dict(self.layer_mapping)
        maps = dict(self.layer_maps)
        if len(mapping) != len(self.layer_mapping):
            raise PhiPipelineError("target layer mapping contains duplicate target layers")
        if len(maps) != len(self.layer_maps):
            raise PhiPipelineError("target layer map ids contain duplicate target layers")
        if set(mapping) != set(maps):
            raise PhiPipelineError("every mapped target layer must have exactly one PhiLayerMapIds record")
        source_layers = [source for _, source in self.layer_mapping]
        if len(source_layers) != len(set(source_layers)):
            raise PhiPipelineError("Phi proof requires unique donor layers per target layer")
        ordered_targets = sorted(mapping)
        ordered_sources = [mapping[target] for target in ordered_targets]
        if ordered_sources != sorted(ordered_sources):
            raise PhiPipelineError("Phi layer mapping must be monotonic")
        if ordered_sources and (
            ordered_sources[0] < 0 or ordered_sources[-1] >= self.donor_manifest.num_layers
        ):
            raise PhiPipelineError("source layer mapping is outside donor layer range")


def _recipient_config_fields(config: Any) -> dict[str, int | str]:
    fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "fingerprint",
    )
    missing = [name for name in fields if not hasattr(config, name)]
    if missing:
        raise PhiPipelineError(f"recipient config is missing fields: {', '.join(missing)}")
    return {name: getattr(config, name) for name in fields}


def build_phi_dense_plan_spec(
    *,
    recipient_config: Any,
    donor_manifest: DonorManifest,
    calibration_manifest_hash: str,
    layer_mapping: Mapping[int, int],
    layer_maps: Mapping[int, PhiLayerMapIds],
    embedding_map_id: str,
    final_residual_map_id: str,
    donor_id: str = "phi",
    plan_id: str = "phi-dense-iq-v1",
) -> PhiDensePlanSpec:
    fields = _recipient_config_fields(recipient_config)
    return PhiDensePlanSpec(
        donor_id=donor_id,
        donor_manifest=donor_manifest,
        calibration_manifest_hash=calibration_manifest_hash,
        recipient_config_hash=str(fields["fingerprint"]),
        recipient_vocab_size=int(fields["vocab_size"]),
        recipient_hidden_size=int(fields["hidden_size"]),
        recipient_intermediate_size=int(fields["intermediate_size"]),
        recipient_num_attention_heads=int(fields["num_attention_heads"]),
        recipient_num_key_value_heads=int(fields["num_key_value_heads"]),
        recipient_head_dim=int(fields["head_dim"]),
        layer_mapping=tuple(sorted((int(t), int(s)) for t, s in layer_mapping.items())),
        layer_maps=tuple(sorted((int(t), ids) for t, ids in layer_maps.items())),
        embedding_map_id=embedding_map_id,
        final_residual_map_id=final_residual_map_id,
        plan_id=plan_id,
    )


def build_phi_dense_transport_plan(spec: PhiDensePlanSpec) -> TransportPlan:
    assignments: list[TargetAssignment] = []
    h = spec.recipient_hidden_size
    i = spec.recipient_intermediate_size
    q = spec.recipient_num_attention_heads * spec.recipient_head_dim
    kv = spec.recipient_num_key_value_heads * spec.recipient_head_dim

    assignments.extend(
        [
            TargetAssignment(
                target_module_path="embed_tokens",
                target_slot=TargetSlot.EMBEDDING,
                target_shape=(spec.recipient_vocab_size, h),
                transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                source_donor_id=spec.donor_id,
                source_operator="model.embed_tokens.weight",
                output_map_id=spec.embedding_map_id,
            ),
            TargetAssignment(
                target_module_path="lm_head",
                target_slot=TargetSlot.LM_HEAD,
                target_shape=(spec.recipient_vocab_size, h),
                transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                source_donor_id=spec.donor_id,
                source_operator="lm_head.weight",
                input_map_id=spec.final_residual_map_id,
            ),
            TargetAssignment(
                target_module_path="norm",
                target_slot=TargetSlot.NORM_SCALE,
                target_shape=(h,),
                transfer_method=TransferMethod.RECIPIENT_NATIVE,
            ),
        ]
    )

    source_by_target = dict(spec.layer_mapping)
    maps_by_target = dict(spec.layer_maps)
    for target_layer in sorted(source_by_target):
        source_layer = source_by_target[target_layer]
        maps = maps_by_target[target_layer]
        prefix = f"blocks.{target_layer}"
        assignments.extend(
            [
                TargetAssignment(
                    target_module_path=f"{prefix}.input_norm",
                    target_slot=TargetSlot.NORM_SCALE,
                    target_shape=(h,),
                    transfer_method=TransferMethod.RECIPIENT_NATIVE,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.attn.q_proj",
                    target_slot=TargetSlot.ATTN_Q,
                    target_shape=(q, h),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="attn.q",
                    input_map_id=maps.attn_in,
                    output_map_id=maps.q,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.attn.k_proj",
                    target_slot=TargetSlot.ATTN_K,
                    target_shape=(kv, h),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="attn.k",
                    input_map_id=maps.attn_in,
                    output_map_id=maps.kv,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.attn.v_proj",
                    target_slot=TargetSlot.ATTN_V,
                    target_shape=(kv, h),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="attn.v",
                    input_map_id=maps.attn_in,
                    output_map_id=maps.kv,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.attn.o_proj",
                    target_slot=TargetSlot.ATTN_O,
                    target_shape=(h, q),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="attn.o",
                    input_map_id=maps.q,
                    output_map_id=maps.attn_out,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.post_attention_norm",
                    target_slot=TargetSlot.NORM_SCALE,
                    target_shape=(h,),
                    transfer_method=TransferMethod.RECIPIENT_NATIVE,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.mlp.gate_proj",
                    target_slot=TargetSlot.MLP_GATE,
                    target_shape=(i, h),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="mlp.gate",
                    input_map_id=maps.mlp_in,
                    output_map_id=maps.mlp_hidden,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.mlp.up_proj",
                    target_slot=TargetSlot.MLP_UP,
                    target_shape=(i, h),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="mlp.up",
                    input_map_id=maps.mlp_in,
                    output_map_id=maps.mlp_hidden,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.mlp.down_proj",
                    target_slot=TargetSlot.MLP_DOWN,
                    target_shape=(h, i),
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=spec.donor_id,
                    source_layer=source_layer,
                    source_operator="mlp.down",
                    input_map_id=maps.mlp_hidden,
                    output_map_id=maps.mlp_out,
                ),
            ]
        )

    return TransportPlan(
        plan_id=spec.plan_id,
        donor_fingerprints=((spec.donor_id, spec.donor_manifest.fingerprint),),
        assignments=tuple(assignments),
        calibration_manifest_hash=spec.calibration_manifest_hash,
        iq_config_hash=spec.recipient_config_hash,
    )
