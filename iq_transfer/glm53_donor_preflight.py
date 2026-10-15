"""Metadata-only GLM-5.3 checkpoint preflight before any IQ tensor conversion.

Source tensor names come from NVIDIA GLM5Bridge.mapping_registry(), NOT
guessed from destination keys. Publisher config is checked by
GLM53Topology.from_hf_config. No model weights are loaded.

https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py
https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .glm53_parallel_contract import GLM53Topology
from .glm53_attention_contract import GLM53AttentionGeometry


class GLM53DonorPreflightError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GLM53DonorInventory:
    """Complete key/file header audit; never a source-weight equivalence claim."""

    model_dir: str
    bf16_required: bool
    required_keys: int
    checkpoint_keys: int
    layer_counts: tuple[int, ...]
    referenced_shards: int
    config_sha256: str
    index_sha256: str
    checkpoint_identity: str
    verified_headers: bool
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": "glm53_donor_index_preflight_v1",
            "model_dir": self.model_dir,
            "bf16_required": self.bf16_required,
            "required_keys": self.required_keys,
            "checkpoint_keys": self.checkpoint_keys,
            "layer_counts": list(self.layer_counts),
            "referenced_shards": self.referenced_shards,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "checkpoint_identity": self.checkpoint_identity,
            "verified_headers": self.verified_headers,
            "complete": self.complete,
            "converted_weights": False,
            "publisher_revision_verified": False,
        }


def required_glm53_hf_keys(topology: GLM53Topology) -> tuple[str, ...]:
    """NVIDIA GLM5Bridge HF-side tensor keys for all 78 decoder stages.

    MTP at model.layers.78 is intentionally excluded. Shared indexer stages
    reuse top-k selection; require own indexer weights only for full stages.
    """
    required: set[str] = {
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    }
    for layer in range(78):
        att = f"model.layers.{layer}.self_attn"
        mlp = f"model.layers.{layer}.mlp"
        required.update((
            f"model.layers.{layer}.input_layernorm.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
            f"{att}.q_a_proj.weight",
            f"{att}.q_b_proj.weight",
            f"{att}.q_a_layernorm.weight",
            f"{att}.kv_a_proj_with_mqa.weight",
            f"{att}.kv_b_proj.weight",
            f"{att}.kv_a_layernorm.weight",
            f"{att}.o_proj.weight",
        ))
        if topology.indexer_types[layer] == "full":
            required.update((
                f"{att}.indexer.wq_b.weight",
                f"{att}.indexer.wk.weight",
                f"{att}.indexer.weights_proj.weight",
                f"{att}.indexer.k_norm.weight",
                f"{att}.indexer.k_norm.bias",
            ))
        if layer in topology.dense_blocks:
            for name in ("gate_proj", "up_proj", "down_proj"):
                required.add(f"{mlp}.{name}.weight")
        else:
            required.update((
                f"{mlp}.gate.weight",
                f"{mlp}.gate.e_score_correction_bias",
            ))
            for name in ("gate_proj", "up_proj", "down_proj"):
                required.add(f"{mlp}.shared_experts.{name}.weight")
            for expert in range(256):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    required.add(f"{mlp}.experts.{expert}.{name}.weight")
    return tuple(sorted(required))


def _validate_shard(root: Path, name: str) -> Path:
    normalized = PurePosixPath(name)
    if (
        not name or name.startswith("/") or "\\" in name
        or len(normalized.parts) != 1 or normalized.name != name
        or name in (".", "..")
    ):
        raise GLM53DonorPreflightError("unsafe safetensors index shard path")
    path = root / name
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise GLM53DonorPreflightError(f"missing or unsafe donor shard {name}")
    return path


def expected_glm53_tensor_shape(
    key: str, *, vocab_size: int,
    attention_geometry: GLM53AttentionGeometry | None = None,
) -> tuple[int, ...] | None:
    """Publisher GLM5Bridge dense/expert tensor geometry, no tensor loading."""
    if attention_geometry is not None and key.startswith("model.layers."):
        pieces = key.split(".")
        if len(pieces) > 3 and pieces[2].isdigit() and pieces[3] == "self_attn":
            block = int(pieces[2])
            if 0 <= block < 78:
                return attention_geometry.source_shapes_for_block(
                    block, indexer_full=True
                ).get(key)
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return (vocab_size, 6144)
    if key == "model.norm.weight":
        return (6144,)
    if key.endswith(".input_layernorm.weight") or key.endswith(".post_attention_layernorm.weight"):
        return (6144,)
    parts = key.split(".")
    if len(parts) < 5 or parts[:2] != ["model", "layers"]:
        return None
    try:
        layer = int(parts[2])
    except ValueError:
        return None
    if not 0 <= layer < 78 or parts[3] != "mlp":
        return None
    tail = ".".join(parts[4:])
    if layer < 3:
        if tail in ("gate_proj.weight", "up_proj.weight"):
            return (12288, 6144)
        if tail == "down_proj.weight":
            return (6144, 12288)
        return None
    if tail == "gate.weight":
        return (256, 6144)
    if tail == "gate.e_score_correction_bias":
        return (256,)
    if tail.startswith("shared_experts."):
        projection = tail.removeprefix("shared_experts.")
    elif tail.startswith("experts."):
        expert_parts = tail.split(".")
        if len(expert_parts) != 4:
            return None
        if not expert_parts[1].isdigit() or not 0 <= int(expert_parts[1]) < 256:
            return None
        projection = ".".join(expert_parts[2:])
    else:
        return None
    if projection in ("gate_proj.weight", "up_proj.weight"):
        return (2048, 6144)
    if projection == "down_proj.weight":
        return (6144, 2048)
    return None


def audit_glm53_donor(
    checkpoint_dir: str | Path, *,
    require_bf16: bool,
    verify_headers: bool = True,
    checkpoint_identity: str,
) -> GLM53DonorInventory:
    """Audit *real* config and indexed safetensors. Never touch the model data.

    Check a complete contiguous expert key inventory, header shapes/dtypes,
    and path-safe existent shards; pass only BF16/FP32 for the IQ native
    expert transport. Unverified declared publisher ID is provenance only.
    """
    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise GLM53DonorPreflightError("donor checkpoint directory does not exist")
    if not checkpoint_identity or checkpoint_identity not in (
        "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16"
    ):
        raise GLM53DonorPreflightError("specify the actual GLM-5.3 publisher checkpoint ID")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise GLM53DonorPreflightError(
            "complete indexed GLM checkpoint requires config.json and "
            "model.safetensors.index.json"
        )
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    topology = GLM53Topology.from_hf_config(config)
    attention_geometry = GLM53AttentionGeometry.from_hf_config(config)
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise GLM53DonorPreflightError("checkpoint vocabulary size must be explicit")
    with index_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in weight_map.items()
    ):
        raise GLM53DonorPreflightError("invalid safetensors weight_map")
    needed = required_glm53_hf_keys(topology)
    missing = sorted(set(needed) - weight_map.keys())
    if missing:
        raise GLM53DonorPreflightError(
            f"GLM5Bridge donor missing {len(missing)} mandatory tensor keys; "
            f"first: {missing[:8]}"
        )
    all_shards = sorted(set(weight_map.values()))
    shard_paths = {name: _validate_shard(root, name) for name in all_shards}
    if require_bf16 and checkpoint_identity != "zai-org/GLM-5.3-BF16":
        # Exact source ID alone is not a dtype guarantee: header verification
        # below is still mandatory before the native IQ converter is used.
        if not verify_headers:
            raise GLM53DonorPreflightError(
                "BF16 conversion requires scanning actual checkpoint headers"
            )
    if require_bf16 and not verify_headers:
        raise GLM53DonorPreflightError(
            "BF16 conversion requires validating safetensors header dtypes"
        )
    layer_counts = [0] * 78
    for key in needed:
        if key.startswith("model.layers."):
            index = int(key.split(".")[2])
            layer_counts[index] += 1
    if verify_headers:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise GLM53DonorPreflightError("safetensors required for header checks") from exc
        keys_by_shard: dict[str, list[str]] = {name: [] for name in all_shards}
        for key in needed:
            keys_by_shard[weight_map[key]].append(key)
        for shard, path in shard_paths.items():
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                actual_keys = set(reader.keys())
                indexed_keys = {
                    name for name, indexed_shard in weight_map.items()
                    if indexed_shard == shard
                }
                if actual_keys != indexed_keys:
                    missing_from_file = sorted(indexed_keys - actual_keys)
                    extra_from_file = sorted(actual_keys - indexed_keys)
                    raise GLM53DonorPreflightError(
                        f"safetensors header/index mismatch {shard}: "
                        f"missing {missing_from_file[:3]} unexpected {extra_from_file[:3]}"
                    )
                for key in keys_by_shard[shard]:
                    view = reader.get_slice(key)
                    shape = tuple(view.get_shape())
                    dtype = view.get_dtype()
                    if any(d <= 0 for d in shape):
                        raise GLM53DonorPreflightError(f"invalid tensor geometry: {key}")
                    expected = expected_glm53_tensor_shape(
                        key, vocab_size=vocab_size,
                        attention_geometry=attention_geometry,
                    )
                    if expected is not None and tuple(shape) != expected:
                        raise GLM53DonorPreflightError(
                            f"GLM5Bridge geometry mismatch {key}: "
                            f"actual {shape} expected {expected}"
                        )
                    if key.endswith("e_score_correction_bias"):
                        if shape != (256,) or dtype not in ("F32", "BF16"):
                            raise GLM53DonorPreflightError(
                                f"router correction bias mismatched: {key}"
                            )
                    elif require_bf16 and dtype not in ("BF16", "F32"):
                        raise GLM53DonorPreflightError(
                            f"GLM donor tensor {key} has {dtype}; "
                            "requires real BF16 data or NVIDIA Bridge dequantization"
                        )
    return GLM53DonorInventory(
        model_dir=str(root),
        bf16_required=require_bf16,
        required_keys=len(needed), checkpoint_keys=len(weight_map),
        layer_counts=tuple(layer_counts),
        referenced_shards=len(all_shards),
        config_sha256=_sha256(config_path),
        index_sha256=_sha256(index_path),
        checkpoint_identity=checkpoint_identity,
        verified_headers=verify_headers,
        complete=verify_headers,
    )
