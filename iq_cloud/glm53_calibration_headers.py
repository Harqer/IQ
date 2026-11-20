"""Bounded header-only audit of EXISTING GLM->IQ residual coordinate maps.

These are weight-only, diagnostics-free 6144->4096 maps. Metadata geometry
does not prove lexical semantic compatibility, low reconstruction error, or
that a new pretrained GLM-tokenized checkpoint exists. Never read matrix data.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import struct
from typing import BinaryIO, Mapping

from .checkpoint_metadata import BUCKET, IQ235MetadataError

CALIBRATION = "calibration78/glm53-full78-20261024-c"
MAP_SHAPE = (6144, 4096)
MAP_DTYPE = "F64"
MAP_FILE_BYTES = 201326680
MAX_MAP_HEADER_BYTES = 2048
MAP_NAMES = ("lexical-input", "final-output")
EXPECTED_SPACES = {
    "lexical-input": (
        "glm53.layer.0.residual", "iq.mamba_foundation.layer.0.residual"
    ),
    "final-output": (
        "glm53.layer.77.residual", "iq.mamba_foundation.layer.77.residual"
    ),
}


def inspect_coordinate_map_header(
    stream: BinaryIO, *, file_size: int,
    source_dims: int = MAP_SHAPE[0], target_dims: int = MAP_SHAPE[1],
) -> dict[str, object]:
    """Accept exactly one F64 coordinate tensor, without reading its data."""
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise IQ235MetadataError("truncated coordinate-map SafeTensors prefix")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= MAX_MAP_HEADER_BYTES:
        raise IQ235MetadataError("unexpected coordinate-map header length")
    raw = stream.read(length)
    if len(raw) != length:
        raise IQ235MetadataError("truncated coordinate-map SafeTensors header")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IQ235MetadataError("invalid coordinate-map JSON header") from exc
    if not isinstance(header, dict) or set(header) != {"matrix"}:
        raise IQ235MetadataError("coordinate-map tensor keys disagree")
    info = header["matrix"]
    if (not isinstance(info, dict)
            or info.get("dtype") != MAP_DTYPE
            or info.get("shape") != [source_dims, target_dims]
            or info.get("data_offsets") != [0, source_dims * target_dims * 8]):
        raise IQ235MetadataError("coordinate-map shape/dtype/offsets disagree")
    if type(file_size) is not int or file_size != 8 + length + source_dims * target_dims * 8:
        raise IQ235MetadataError("coordinate-map SafeTensors file-size mismatch")
    return {
        "dtype": MAP_DTYPE,
        "matrix_shape": [source_dims, target_dims],
        "header_bytes_inspected": 8+length,
        "matrix_payload_bytes": source_dims*target_dims*8,
        "matrix_payload_read": False,
    }


def inspect_map_metadata(
    name: str, raw_json: bytes, expected_checksum: str
) -> dict[str, object]:
    """Verify archived metadata JSON and provenance, not calibration quality."""
    if name not in MAP_NAMES:
        raise IQ235MetadataError("unsupported coordinate-map name")
    if sha256(raw_json).hexdigest() != expected_checksum:
        raise IQ235MetadataError("archived coordinate-map metadata checksum mismatch")
    try:
        metadata = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IQ235MetadataError("invalid coordinate-map metadata JSON") from exc
    if (not isinstance(metadata, dict)
            or metadata.get("schema_version") != 1
            or metadata.get("diagnostics") is not None
            or metadata.get("ridge") != 1e-12
            or (metadata.get("source_space"), metadata.get("target_space"))
                != EXPECTED_SPACES[name]):
        raise IQ235MetadataError("unexpected coordinate-map calibration provenance")
    return {
        "source_space": metadata["source_space"],
        "target_space": metadata["target_space"],
        "calibration_diagnostics_available": False,
        "ridge": metadata["ridge"],
        "calibration_quality_validated": False,
    }


def inspect_remote_coordinate_maps() -> dict[str, object]:
    """Inspect JSON metadata plus a few dozen SafeTensors header bytes remotely."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(block_size=4096)
    root = f"hf://buckets/{BUCKET}/{CALIBRATION}"
    def read_small(path: str, max_bytes: int) -> bytes:
        with fs.open(path, "rb", block_size=4096) as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise IQ235MetadataError("calibration metadata exceeds bounded read")
        return raw
    archive = json.loads(read_small(f"{root}/calibration_sha256.json", 150000))
    manifest = json.loads(read_small(f"{root}/glm53_calibration.json", 60000))
    if (manifest.get("schema_version") != 2
            or manifest.get("calibration_mode") != "weight_only"
            or manifest.get("source_layers") != 78
            or manifest.get("lexical_input_map") != "lexical-input"
            or manifest.get("final_output_map") != "final-output"
            or len(manifest.get("stages", [])) != 78):
        raise IQ235MetadataError("unexpected archival calibration schema")
    records: dict[str, object] = {}
    for name in MAP_NAMES:
        json_name = name + ".json"
        data_name = name + ".safetensors"
        checksum = archive.get("files", {}).get(json_name)
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise IQ235MetadataError("archived map checksum missing")
        metadata = inspect_map_metadata(
            name, read_small(f"{root}/{json_name}", 2048), checksum
        )
        full = f"{root}/{data_name}"
        size = fs.info(full).get("size")
        if size != MAP_FILE_BYTES:
            raise IQ235MetadataError("coordinate-map remote file size mismatch")
        with fs.open(full, "rb", block_size=4096) as stream:
            header = inspect_coordinate_map_header(stream, file_size=size)
        records[name] = {**metadata, **header}
    return {
        "status": "archived_weight_only_coordinate_map_geometry_verified",
        "calibration_run": CALIBRATION,
        "calibration_mode": "weight_only",
        "coordinate_maps": records,
        "independent_activation_validation_present": False,
        "glm_tokenized_lexical_overlay_present": False,
        "matrix_payloads_decoded": False,
        "checkpoint_weights_downloaded": False,
        "pretrained_inference_ready": False,
    }
