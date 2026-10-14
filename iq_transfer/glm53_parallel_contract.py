"""Pinned GLM-5.3 architecture topology, independent of checkpoint tensor IO.

Source of truth: GLM-5.3 config.json at
zai-org/GLM-5.3@aca966e4e02791568aa6a4ced368624b3d897f42
and NVIDIA-NeMo/Megatron-Bridge GLM5Bridge. The appended MTP block is
separate from the 78 decoder blocks and is not silently imported into IQ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any
from hashlib import sha256
import json

GLM53_CHECKPOINT_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
GLM53_DECODER_BLOCKS = 78
GLM53_DENSE_STAGES = (0, 1, 2)
GLM53_ROUTED_EXPERTS = 256
GLM53_EXPERTS_PER_TOKEN = 8
GLM53_RESIDUAL_WIDTH = 6144


class GLM53TopologyError(ValueError):
    pass


def expected_indexer_types() -> tuple[str, ...]:
    # Literal publisher IndexShare pattern, not an alternating CSA/HCA design.
    return tuple(
        "full" if i < 3 or (i - 2) % 4 == 0 else "shared"
        for i in range(GLM53_DECODER_BLOCKS)
    )


def index_source_layers(indexer_types: tuple[str, ...]) -> tuple[int, ...]:
    if len(indexer_types) != GLM53_DECODER_BLOCKS:
        raise GLM53TopologyError("exactly 78 GLM DSA indexer stages are required")
    owner: int | None = None
    result = []
    for i, kind in enumerate(indexer_types):
        if kind == "full":
            owner = i
        elif kind != "shared":
            raise GLM53TopologyError(f"unrecognized GLM indexer type at {i}: {kind}")
        if owner is None:
            raise GLM53TopologyError("DSA shared indexer precedes all full indexers")
        result.append(owner)
    return tuple(result)


@dataclass(frozen=True)
class GLM53Topology:
    indexer_types: tuple[str, ...]
    index_source_blocks: tuple[int, ...]
    dense_blocks: tuple[int, ...]
    sparse_blocks: tuple[int, ...]
    num_routed_experts: int
    top_k: int
    donor_hidden_size: int
    iq_hidden_size: int
    megatron_physical_layers: int

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "GLM53Topology":
        required = {
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": GLM53_DECODER_BLOCKS,
            "first_k_dense_replace": len(GLM53_DENSE_STAGES),
            "n_routed_experts": GLM53_ROUTED_EXPERTS,
            "num_experts_per_tok": GLM53_EXPERTS_PER_TOKEN,
            "n_shared_experts": 1,
            "hidden_size": GLM53_RESIDUAL_WIDTH,
            "intermediate_size": 12288,
            "moe_intermediate_size": 2048,
            "index_topk": 2048,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
        }
        for name, expected in required.items():
            if config.get(name) != expected:
                raise GLM53TopologyError(
                    f"GLM-5.3 checkpoint topology mismatch: {name} must be {expected!r}"
                )
        raw_index = config.get("indexer_types")
        if not isinstance(raw_index, list):
            raise GLM53TopologyError("GLM-5.3 must provide 78 indexer_types")
        kinds = tuple(raw_index)
        if kinds != expected_indexer_types():
            raise GLM53TopologyError("GLM-5.3 DSA IndexShare pattern changed")
        raw_ffn = config.get("mlp_layer_types")
        if not isinstance(raw_ffn, list):
            raise GLM53TopologyError("GLM-5.3 must provide mlp_layer_types")
        ffn = tuple(raw_ffn)
        if ffn != ("dense",) * 3 + ("sparse",) * 75:
            raise GLM53TopologyError("GLM-5.3 dense/MoE stage order differs")
        if config.get("num_nextn_predict_layers") != 1:
            raise GLM53TopologyError("GLM-5.3 appended MTP layer must be accounted separately")
        return cls(
            indexer_types=kinds,
            index_source_blocks=index_source_layers(kinds),
            dense_blocks=GLM53_DENSE_STAGES,
            sparse_blocks=tuple(range(3, 78)),
            num_routed_experts=256,
            top_k=8,
            donor_hidden_size=6144,
            iq_hidden_size=4096,
            megatron_physical_layers=156,
        )

    def export_manifest(self) -> dict[str, Any]:
        stages = [
            {
                "block": i,
                "megatron_attention_physical": 2 * i,
                "megatron_feedforward_physical": 2 * i + 1,
                "ffn_kind": "dense" if i < 3 else "moe",
                "indexer_kind": self.indexer_types[i],
                "index_source_block": self.index_source_blocks[i],
                "destination_prefix": f"layers.{i}",
                "weight_equivalent": False,
            }
            for i in range(78)
        ]
        payload = {
            "donor_revision": GLM53_CHECKPOINT_REVISION,
            "donor_decoder_blocks": 78,
            "megatron_physical_layers": 156,
            "iq_parallel_blocks": 78,
            "donor_hidden_size": self.donor_hidden_size,
            "iq_hidden_size": self.iq_hidden_size,
            "routed_experts_per_moe": 256,
            "routed_top_k": 8,
            "dense_blocks": list(self.dense_blocks),
            "moe_blocks": list(self.sparse_blocks),
            "stages": stages,
            "not_transferred": ["MTP", "GLM lexical embeddings", "GLM-5.3 logits"],
            "requires": ["coordinate_transport", "DSA_indexshare_parity",
                         "full_expert_tensor_mapping", "activation_alignment"],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["fingerprint"] = sha256(serialized.encode()).hexdigest()
        return payload
