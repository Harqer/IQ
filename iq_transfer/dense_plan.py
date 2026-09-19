from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from .manifest import DonorManifest
from .plan import TransportPlan
from .slots import TargetAssignment, TargetSlot, TransferMethod


class DensePlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class DenseLayerMapIds:
    residual_in: str
    q: str
    k: str
    v: str
    attn_in: str
    residual_out: str
    gate: str
    up: str
    mlp_in: str

    def __post_init__(self) -> None:
        values = (
            self.residual_in,
            self.q,
            self.k,
            self.v,
            self.attn_in,
            self.residual_out,
            self.gate,
            self.up,
            self.mlp_in,
        )
        if any(not value.strip() for value in values):
            raise DensePlanError("all dense layer coordinate-map ids must be non-empty")


def build_phi_dense_transport_plan(
    model_config: Any,
    donor_manifest: DonorManifest,
    *,
    layer_mapping: Mapping[int, int],
    layer_maps: Mapping[int, DenseLayerMapIds],
    calibration_manifest_hash: str,
    embedding_map_id: str,
    lm_head_map_id: str,
    donor_id: str = "phi",
    plan_id: str = "phi-dense-iq",
) -> TransportPlan:
    if donor_manifest.donor_id != donor_id:
        raise DensePlanError(
            f"donor id mismatch: manifest={donor_manifest.donor_id!r}, requested={donor_id!r}"
        )
    if not calibration_manifest_hash.strip() or not embedding_map_id.strip() or not lm_head_map_id.strip():
        raise DensePlanError("calibration and lexical map ids must be non-empty")

    required_layers = set(range(int(model_config.num_hidden_layers)))
    if set(layer_mapping) != required_layers:
        raise DensePlanError(
            f"layer_mapping must cover every IQ layer exactly: expected={sorted(required_layers)}, "
            f"got={sorted(layer_mapping)}"
        )
    if set(layer_maps) != required_layers:
        raise DensePlanError(
            f"layer_maps must cover every IQ layer exactly: expected={sorted(required_layers)}, "
            f"got={sorted(layer_maps)}"
        )
    bad_source = {
        source_layer
        for source_layer in layer_mapping.values()
        if source_layer < 0 or source_layer >= donor_manifest.num_layers
    }
    if bad_source:
        raise DensePlanError(f"source layer indices outside donor range: {sorted(bad_source)}")

    hidden = int(model_config.hidden_size)
    head_dim = int(model_config.head_dim)
    q_dim = int(model_config.num_attention_heads) * head_dim
    kv_dim = int(model_config.num_key_value_heads) * head_dim
    intermediate = int(model_config.intermediate_size)
    vocab = int(model_config.vocab_size)

    assignments: list[TargetAssignment] = [
        TargetAssignment(
            target_module_path="embed_tokens",
            target_slot=TargetSlot.EMBEDDING,
            target_shape=(vocab, hidden),
            transfer_method=TransferMethod.OPERATOR_TRANSPORT,
            source_donor_id=donor_id,
            source_operator="model.embed_tokens.weight",
            output_map_id=embedding_map_id,
        ),
        TargetAssignment(
            target_module_path="lm_head",
            target_slot=TargetSlot.LM_HEAD,
            target_shape=(vocab, hidden),
            transfer_method=TransferMethod.OPERATOR_TRANSPORT,
            source_donor_id=donor_id,
            source_operator="lm_head.weight",
            input_map_id=lm_head_map_id,
        ),
    ]

    for target_layer in sorted(required_layers):
        source_layer = int(layer_mapping[target_layer])
        maps = layer_maps[target_layer]
        prefix = f"blocks.{target_layer}"
        specs = (
            ("attn.q_proj", TargetSlot.ATTN_Q, (q_dim, hidden), "attn.q", maps.residual_in, maps.q),
            ("attn.k_proj", TargetSlot.ATTN_K, (kv_dim, hidden), "attn.k", maps.residual_in, maps.k),
            ("attn.v_proj", TargetSlot.ATTN_V, (kv_dim, hidden), "attn.v", maps.residual_in, maps.v),
            ("attn.o_proj", TargetSlot.ATTN_O, (hidden, q_dim), "attn.o", maps.attn_in, maps.residual_out),
            ("mlp.gate_proj", TargetSlot.MLP_GATE, (intermediate, hidden), "mlp.gate", maps.residual_in, maps.gate),
            ("mlp.up_proj", TargetSlot.MLP_UP, (intermediate, hidden), "mlp.up", maps.residual_in, maps.up),
            ("mlp.down_proj", TargetSlot.MLP_DOWN, (hidden, intermediate), "mlp.down", maps.mlp_in, maps.residual_out),
        )
        for suffix, slot, shape, role, input_map, output_map in specs:
            assignments.append(
                TargetAssignment(
                    target_module_path=f"{prefix}.{suffix}",
                    target_slot=slot,
                    target_shape=shape,
                    transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                    source_donor_id=donor_id,
                    source_layer=source_layer,
                    source_operator=role,
                    input_map_id=input_map,
                    output_map_id=output_map,
                )
            )
        assignments.extend(
            (
                TargetAssignment(
                    target_module_path=f"{prefix}.input_norm",
                    target_slot=TargetSlot.NORM_SCALE,
                    target_shape=(hidden,),
                    transfer_method=TransferMethod.RECIPIENT_NATIVE,
                ),
                TargetAssignment(
                    target_module_path=f"{prefix}.post_attention_norm",
                    target_slot=TargetSlot.NORM_SCALE,
                    target_shape=(hidden,),
                    transfer_method=TransferMethod.RECIPIENT_NATIVE,
                ),
            )
        )

    assignments.append(
        TargetAssignment(
            target_module_path="norm",
            target_slot=TargetSlot.NORM_SCALE,
            target_shape=(hidden,),
            transfer_method=TransferMethod.RECIPIENT_NATIVE,
        )
    )

    config_hash = getattr(model_config, "fingerprint", None)
    if not isinstance(config_hash, str) or not config_hash:
        raise DensePlanError("model_config must expose a non-empty fingerprint")

    return TransportPlan(
        plan_id=plan_id,
        donor_fingerprints=((donor_id, donor_manifest.fingerprint),),
        assignments=tuple(assignments),
        calibration_manifest_hash=calibration_manifest_hash,
        iq_config_hash=config_hash,
    )
