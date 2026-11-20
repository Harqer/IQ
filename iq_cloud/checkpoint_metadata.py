"""Pure metadata checks for the 539 GB original IQ235 checkpoint.

Never opens SafeTensors weight payloads or modifies the HF bucket. Does NOT
establish tensor integrity, functional weight alignment, or inference readiness.
"""
from __future__ import annotations
import json
from pathlib import Path

from .glm53_lexical import inspect_lexical_boundary

EXPECTED_SHARDS = 20283
EXPECTED_KEYS = 63164
EXPECTED_BYTES = 539388468802
GLOBAL = "model-global.safetensors"
BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "IQ/runs/glm53-full78-local-20261025a"

class IQ235MetadataError(ValueError):
    pass

def inspect_index(config, expected_source, glm_profile, index, marker,
                  *, counts=(EXPECTED_KEYS, EXPECTED_SHARDS, EXPECTED_BYTES)):
    if config != expected_source:
        raise IQ235MetadataError("bucket/source config mismatch")
    lexical = inspect_lexical_boundary(expected_source, glm_profile)
    if (marker.get("inference_ready") is not False
            or marker.get("tokenizer_packaged") is not False
            or marker.get("end_to_end_knowledge_retention_proven") is not False
            or marker.get("mapped_layers") != 78
            or marker.get("mapped_routed_expert_slots") != 19200
            or marker.get("routed_experts_per_layer") != 256
            or marker.get("status") != "FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED"):
        raise IQ235MetadataError("compiled source status/provenance mismatch")
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise IQ235MetadataError("invalid checkpoint index")
    if len(weight_map) != counts[0] or metadata.get("total_size") != counts[2]:
        raise IQ235MetadataError("index tensor count or total bytes mismatch")
    shards = set()
    for key, name in weight_map.items():
        if (not isinstance(key, str) or not key or not isinstance(name, str)
                or not name.startswith("model-") or not name.endswith(".safetensors")
                or "/" in name or "\\" in name or name in {".", ".."}):
            raise IQ235MetadataError("invalid indexed tensor or shard path")
        shards.add(name)
    if len(shards) != counts[1]:
        raise IQ235MetadataError("indexed shard count mismatch")
    for key in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
        if weight_map.get(key) != GLOBAL:
            raise IQ235MetadataError("lexical/global tensor location mismatch")
    return {
        "status": "source_metadata_verified_only",
        "model_layers": lexical["physical_layers"],
        "source_vocab": lexical["source_vocab"],
        "glm_target_vocab": lexical["glm_target_vocab"],
        "indexed_shards": len(shards),
        "indexed_tensor_keys": len(weight_map),
        "indexed_total_bytes": metadata["total_size"],
        "weight_payloads_read": False,
        "weight_payloads_verified": False,
        "lexical_alignment_verified": False,
        "inference_ready": False,
    }

def inspect_bucket(source_config, glm_profile):
    """Read only bounded JSON metadata over HfFileSystem; no safetensors files."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = f"hf://buckets/{BUCKET}/{RUN}"
    def get(name, cap):
        with fs.open(f"{base}/{name}", "rb") as stream:
            raw = stream.read(cap + 1)
        if len(raw) > cap:
            raise IQ235MetadataError("metadata JSON exceeds bounded read")
        return json.loads(raw)
    config = get("iq_config.json", 100000)
    index = get("model.safetensors.index.json", 8000000)
    marker = get("COMPILE_COMPLETE.json", 100000)
    report = inspect_index(config, source_config, glm_profile, index, marker)
    report["global_shard_exists"] = bool(
        fs.info(f"{base}/{GLOBAL}").get("size", 0) > 0
    )
    if not report["global_shard_exists"]:
        raise IQ235MetadataError("original global lexical shard unavailable")
    return report
