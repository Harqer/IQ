"""NVIDIA GLM5Bridge HF-side MLA/DSA source contract for IQ transfer.

This inventories the *source* operator and shape, not a fabricated tensor
mapping into IQ's fundamentally different CSA compressor/output structure.
GLM uses 78 MLA blocks and full DSA indexers only on the explicit
IndexShare owner layers. Sources:
- https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Any, Mapping

from .glm53_parallel_contract import GLM53Topology


class GLM53AttentionContractError(ValueError):
    pass


@dataclass(frozen=True)
class GLM53AttentionGeometry:
    hidden: int
    heads: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_head: int
    index_heads: int
    index_head_dim: int

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "GLM53AttentionGeometry":
        required = {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "indexer_rope_interleave": True,
            "rope_interleave": True,
            "attention_bias": False,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise GLM53AttentionContractError(
                    f"GLM-5.3 MLA/DSA changed {key}: expected {expected!r}, "
                    f"actual {config.get(key)!r}"
                )
        rope = config.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_theta") != 8000000:
            raise GLM53AttentionContractError("GLM-5.3 RoPE theta differs from publisher")
        return cls(
            hidden=6144, heads=64, q_lora=2048, kv_lora=512,
            qk_nope=192, qk_rope=64, v_head=256,
            index_heads=32, index_head_dim=128,
        )

    def source_shapes_for_block(self, block: int, *, indexer_full: bool) -> dict[str, tuple[int, ...]]:
        if not 0 <= block < 78:
            raise GLM53AttentionContractError("only 78 decoder attention blocks exist")
        base = f"model.layers.{block}.self_attn"
        shapes: dict[str, tuple[int, ...]] = {
            f"{base}.q_a_proj.weight": (self.q_lora, self.hidden),
            f"{base}.q_a_layernorm.weight": (self.q_lora,),
            f"{base}.q_b_proj.weight": (
                self.heads * (self.qk_nope + self.qk_rope), self.q_lora
            ),
            f"{base}.kv_a_proj_with_mqa.weight": (
                self.kv_lora + self.qk_rope, self.hidden
            ),
            f"{base}.kv_a_layernorm.weight": (self.kv_lora,),
            f"{base}.kv_b_proj.weight": (
                self.heads * (self.qk_nope + self.v_head), self.kv_lora
            ),
            f"{base}.o_proj.weight": (self.hidden, self.heads * self.v_head),
        }
        if indexer_full:
            shapes.update({
                f"{base}.indexer.wq_b.weight": (
                    self.index_heads * self.index_head_dim, self.q_lora
                ),
                f"{base}.indexer.wk.weight": (self.index_head_dim, self.hidden),
                f"{base}.indexer.weights_proj.weight": (self.index_heads, self.hidden),
                f"{base}.indexer.k_norm.weight": (self.index_head_dim,),
                f"{base}.indexer.k_norm.bias": (self.index_head_dim,),
            })
        return shapes


def attention_source_manifest(
    topology: GLM53Topology, geometry: GLM53AttentionGeometry,
) -> dict[str, Any]:
    source = []
    for block in range(78):
        full = topology.indexer_types[block] == "full"
        for name, shape in geometry.source_shapes_for_block(
            block, indexer_full=full
        ).items():
            source.append({
                "block": block, "source": name, "shape": list(shape),
                "operator": "dsa_indexer" if ".indexer." in name else "mla",
                "index_source": topology.index_source_blocks[block],
                "destination": None,
                "status": "requires_source_operator_alignment",
            })
    source.sort(key=lambda item: item["source"])
    payload = {
        "schema": 1,
        "decoder_blocks": 78,
        "full_indexer_blocks": sum(t == "full" for t in topology.indexer_types),
        "source_tensor_count": len(source),
        "model_target": "IQ_78_fused_Mamba3_parallel_CSA",
        "source_tensors": source,
        "converted": False,
        "weight_equivalent": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = sha256(raw).hexdigest()
    return payload
