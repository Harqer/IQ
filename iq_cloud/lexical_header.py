"""Bounded SafeTensors header inspection; no tensor-payload deserialization.

The IQ235 checkpoint is stored in a Hugging Face bucket. This module reads
only the global SafeTensors header through fsspec's small-block remote reader
to prove the source embedding and output head shapes. A remote backend may
buffer adjacent bytes; it must not download an entire checkpoint shard.
"""
from __future__ import annotations

import json
import math
import struct
from typing import BinaryIO

from .checkpoint_metadata import (
    BUCKET, RUN, GLOBAL, EXPECTED_GLOBAL_BYTES, IQ235MetadataError,
)

MAX_HEADER_BYTES = 65536
EXPECTED_SHAPES = {
    "embed_tokens.weight": (128256, 4096),
    "lm_head.weight": (128256, 4096),
    "norm.weight": (4096,),
}
BF16_BYTES = 2


def inspect_global_header(file_obj: BinaryIO, *, file_size: int) -> dict[str, object]:
    """Inspect the header only; validate all source lexical tensors exactly."""
    if type(file_size) is not int or file_size <= 8:
        raise IQ235MetadataError("invalid global shard size")
    prefix = file_obj.read(8)
    if len(prefix) != 8:
        raise IQ235MetadataError("truncated global SafeTensors prefix")
    header_length = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_length <= MAX_HEADER_BYTES:
        raise IQ235MetadataError("global SafeTensors header length out of bounds")
    raw_header = file_obj.read(header_length)
    if len(raw_header) != header_length:
        raise IQ235MetadataError("truncated global SafeTensors header")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IQ235MetadataError("malformed global SafeTensors header") from exc
    if not isinstance(header, dict):
        raise IQ235MetadataError("global SafeTensors header not a JSON object")
    keys = set(header) - {"__metadata__"}
    if keys != set(EXPECTED_SHAPES):
        raise IQ235MetadataError("global lexical tensor names mismatch")
    spans: list[tuple[int, int]] = []
    summary: dict[str, dict[str, object]] = {}
    for name, expected_shape in EXPECTED_SHAPES.items():
        record = header[name]
        if not isinstance(record, dict):
            raise IQ235MetadataError(f"invalid lexical tensor metadata: {name}")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (record.get("dtype") != "BF16"
                or not isinstance(shape, list)
                or tuple(shape) != expected_shape
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(i) is not int for i in offsets)):
            raise IQ235MetadataError(f"lexical tensor shape/dtype/offset mismatch: {name}")
        start, end = offsets
        nbytes = math.prod(expected_shape) * BF16_BYTES
        if start < 0 or end - start != nbytes:
            raise IQ235MetadataError(f"lexical tensor byte span mismatch: {name}")
        spans.append((start, end))
        summary[name] = {"dtype": "BF16", "shape": list(shape), "payload_bytes": nbytes}
    current = 0
    for start, end in sorted(spans):
        if start != current:
            raise IQ235MetadataError("overlapping or gapped global SafeTensors data offsets")
        current = end
    if current + 8 + header_length != file_size:
        raise IQ235MetadataError("global SafeTensors header/file size mismatch")
    return {
        "status": "source_glm_lexical_incompatible_without_alignment",
        "original_lexical_tensors": summary,
        "header_bytes_inspected": header_length + 8,
        "original_file_bytes": file_size,
        "target_glm_embedding_shape": [154880, 4096],
        "target_glm_head_shape": [154880, 4096],
        "trained_glm_lexical_weights_present": False,
        "tensor_payloads_decoded": False,
        "pretrained_inference_ready": False,
    }


def inspect_remote_global_lexical_shard(source_config: dict) -> dict[str, object]:
    """Read a bounded, small-block remote header without fetching model weights."""
    if (source_config.get("model", {}).get("vocab_size") != 128256
            or source_config.get("model", {}).get("hidden_size") != 4096):
        raise IQ235MetadataError("IQ235 original lexical configuration mismatch")
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(block_size=4096)
    path = f"hf://buckets/{BUCKET}/{RUN}/{GLOBAL}"
    size = fs.info(path).get("size")
    if size != EXPECTED_GLOBAL_BYTES:
        raise IQ235MetadataError("original global lexical shard size mismatch")
    with fs.open(path, "rb", block_size=4096) as stream:
        report = inspect_global_header(stream, file_size=size)
    report["remote_hf_bucket"] = BUCKET
    report["remote_original_shard"] = GLOBAL
    report["source_shards_downloaded"] = False
    return report
