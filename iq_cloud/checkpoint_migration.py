"""Build an executable-weight-loader *plan* from IQ235 checkpoint metadata.

The pinned Hugging Face checkpoint contains 63164 tensor names over 20283
shards. Only the source embed_tokens.weight and lm_head.weight have different
shapes in the GLM-5.3 vocabulary profile; the backbone and norm retain their
original names and geometry.

This module NEVER reads a safetensors payload, loads neural weights, starts
GPU compute, changes an index, or claims full-checkpoint verification.
A separate trusted loader must check every tensor's real header, value
integrity, and final device placement before full-model lexical training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
from typing import Any, Mapping

from .checkpoint_metadata import (
    GLOBAL, EXPECTED_BYTES, EXPECTED_KEYS, EXPECTED_SHARDS,
    IQ235MetadataError, inspect_index,
)
from .glm53_lexical import (
    GLM_VOCAB, IQ_WIDTH, SOURCE_IQ_VOCAB, inspect_lexical_boundary,
)

_LEXICAL = frozenset(("embed_tokens.weight", "lm_head.weight"))
_LAYER_PATTERN = re.compile(r"^layers\.(\d+)\.")
_DEVICE_COUNT = 8


def build_migration_plan(
    original_config: Mapping[str, Any],
    target_config: Mapping[str, Any],
    source_index: Mapping[str, Any],
    compile_marker: Mapping[str, Any],
    *,
    test_counts: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Validate the entire source index and plan a lossless BODY reuse.

    Lexical *candidates* are separate from trained weights. This inventory
    cannot prove that any source tensor or initialized target value is valid.
    """
    counts = (EXPECTED_KEYS, EXPECTED_SHARDS, EXPECTED_BYTES)
    if test_counts is not None:
        if (len(test_counts) != 3 or any(type(x) is not int or x <= 0
                                        for x in test_counts)):
            raise IQ235MetadataError("invalid synthetic index count expectations")
        counts = test_counts
    inspected = inspect_index(
        original_config, original_config, target_config,
        source_index, compile_marker, counts=counts,
    )
    lexical = inspect_lexical_boundary(original_config, target_config)
    map_ = source_index["weight_map"]
    if not all(key in map_ for key in _LEXICAL):
        raise IQ235MetadataError("source lexical tensor keys are missing")
    # The source config differs from the GLM profile in the two lexical rows
    # only. A production loader must verify EVERY OTHER tensor shape in situ.
    layers = original_config["schedule"]["layers"]
    if len(layers) != 235:
        raise IQ235MetadataError("source topology is not full235")
    per_device: Counter[int] = Counter()
    keys_per_shard: Counter[str] = Counter()
    shard_devices: dict[str, set[int]] = defaultdict(set)
    selected_nonlexical = 0
    for key, shard in map_.items():
        if key in _LEXICAL:
            continue
        match = _LAYER_PATTERN.match(key)
        if match is not None:
            layer = int(match.group(1))
            if not 0 <= layer < len(layers):
                raise IQ235MetadataError("indexed source parameter layer is out of range")
            device = layer % _DEVICE_COUNT
        elif key.startswith("layers."):
            raise IQ235MetadataError("malformed source physical layer parameter name")
        else:
            device = 0
        keys_per_shard[shard] += 1
        shard_devices[shard].add(device)
        per_device[device] += 1
        selected_nonlexical += 1
    if selected_nonlexical != counts[0] - 2:
        raise IQ235MetadataError("nonlexical source tensor inventory incomplete")
    if map_["embed_tokens.weight"] != GLOBAL or map_["lm_head.weight"] != GLOBAL:
        raise IQ235MetadataError("source lexical payloads not in original global shard")
    return {
        "status": "iq235_source_to_glm53_metadata_plan_only",
        "source_vocab_size": SOURCE_IQ_VOCAB,
        "target_vocab_size": GLM_VOCAB,
        "hidden_size": IQ_WIDTH,
        "source_index_tensor_keys": inspected["indexed_tensor_keys"],
        "source_index_shards": inspected["indexed_shards"],
        "source_index_total_bytes": inspected["indexed_total_bytes"],
        "reused_nonlexical_source_key_count": selected_nonlexical,
        "replaced_original_lexical_keys": sorted(_LEXICAL),
        "replacement_embedding_shape": lexical["target_embedding_shape"],
        "replacement_head_shape": lexical["target_lm_head_shape"],
        "target_tensor_key_count": selected_nonlexical + 2,
        "body_tensor_keys_by_planned_gpu": {
            str(device): per_device[device] for device in range(_DEVICE_COUNT)
        },
        "body_shards_referenced": len(keys_per_shard),
        "multi_device_shards_in_metadata": sum(
            1 for owners in shard_devices.values() if len(owners) > 1
        ),
        "all_shard_headers_verified": False,
        "all_payload_sha256_verified": False,
        "all_source_tensors_injected": False,
        "target_lexical_initialized": False,
        "target_lexical_trained": False,
        "readiness_for_training": False,
        "inference_ready": False,
        "GPU_compute_started": False,
        "source_weights_downloaded": False,
        "source_bucket_modified": False,
    }


def inspect_remote_migration_plan(
    source_config: Mapping[str, Any], glm_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Use only bounded JSON reads, never safetensors bodies."""
    from huggingface_hub import HfFileSystem
    from .checkpoint_metadata import (
        BUCKET, RUN, check_pinned_metadata_sha256,
    )

    base = f"hf://buckets/{BUCKET}/{RUN}"
    fs = HfFileSystem()
    def read_json_bytes(name: str, limit: int) -> bytes:
        with fs.open(f"{base}/{name}", "rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise IQ235MetadataError(f"checkpoint metadata exceeds size limit: {name}")
        return raw

    config_bytes = read_json_bytes("iq_config.json", 100_000)
    index_bytes = read_json_bytes("model.safetensors.index.json", 8_000_000)
    marker_bytes = read_json_bytes("COMPILE_COMPLETE.json", 100_000)
    check_pinned_metadata_sha256(config_bytes, index_bytes)
    config = json.loads(config_bytes)
    if config != source_config:
        raise IQ235MetadataError("source HF configuration differs from pinned original")
    plan = build_migration_plan(
        config, glm_profile, json.loads(index_bytes), json.loads(marker_bytes)
    )
    plan["source_metadata_fingerprints_verified"] = True
    return plan
