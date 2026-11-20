"""Pure metadata checks for the 539 GB original IQ235 checkpoint.

Never opens SafeTensors weight payloads or modifies the HF bucket. Does NOT
establish tensor integrity, functional weight alignment, or inference readiness.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path

from .glm53_lexical import inspect_lexical_boundary

EXPECTED_SHARDS = 20283
EXPECTED_KEYS = 63164
EXPECTED_BYTES = 539388468802
# Recorded in the immutable HF device-map metadata, verified against the
# existing IQ235 checkpoint index/config. Not safetensors payload digests.
EXPECTED_INDEX_SHA256 = "0367412e31bb391a5c892497b565164a4b7954611250d1df1f0a986a8194b2bd"
EXPECTED_CONFIG_SHA256 = "93f351e2bdd116ea5c83792cd1e494ce8dd904cb19c3bcac8c3cbecb1271a799"
EXPECTED_GLOBAL_BYTES = 2101354776
# No trusted 20,283-shard payload checksum manifest has been reviewed.
# This remains None until a real remote integrity audit and human approval.
PINNED_SOURCE_SHARD_MANIFEST_SHA256: str | None = None
GLOBAL = "model-global.safetensors"
BUCKET = "kofecheckmate/iq-glm53-all256"
RUN = "IQ/runs/glm53-full78-local-20261025a"

class IQ235MetadataError(ValueError):
    pass


def check_pinned_metadata_sha256(config_bytes: bytes, index_bytes: bytes,
                                 *, config_digest: str = EXPECTED_CONFIG_SHA256,
                                 index_digest: str = EXPECTED_INDEX_SHA256) -> None:
    """Check original JSON bytes against historical recorded digest values."""
    if sha256(config_bytes).hexdigest() != config_digest:
        raise IQ235MetadataError("compiled config SHA-256 mismatch")
    if sha256(index_bytes).hexdigest() != index_digest:
        raise IQ235MetadataError("checkpoint index SHA-256 mismatch")


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
        return raw
    config_bytes = get("iq_config.json", 100000)
    index_bytes = get("model.safetensors.index.json", 8000000)
    check_pinned_metadata_sha256(config_bytes, index_bytes)
    marker_bytes = get("COMPILE_COMPLETE.json", 100000)
    config = json.loads(config_bytes)
    index = json.loads(index_bytes)
    marker = json.loads(marker_bytes)
    report = inspect_index(config, source_config, glm_profile, index, marker)
    global_size = fs.info(f"{base}/{GLOBAL}").get("size")
    if global_size != EXPECTED_GLOBAL_BYTES:
        raise IQ235MetadataError("original global lexical shard size mismatch")
    report["global_shard_size_bytes"] = global_size
    report["pinned_config_sha256_checked"] = True
    report["pinned_index_sha256_checked"] = True
    return report
