"""Read-only, whole-checkpoint integrity gate; CPU/file I/O only, no CUDA allocation.

Validates SHA256SUMS over EVERY shard, the COMPLETE.json index digest, index
coverage, and safetensors headers/offsets/dtype/shape without loading weights.
SHA-256 covers the complete shard payloads, not just file names and lengths.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

# Dtype bit counts follow the safetensors specification.
_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "U16": 16, "I16": 16,
    "U32": 32, "I32": 32, "U64": 64, "I64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8, "F8_E8M0": 8,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "C64": 64,
}
_HASH_LINE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024 * 1024


class IQIntegrityError(ValueError):
    """Refuse a checkpoint with unverified, malformed or corrupted tensors."""


@dataclass(frozen=True)
class IQIntegrityReport:
    shard_count: int
    tensor_count: int
    shard_bytes: int
    verified_sha256_count: int
    verified_header_count: int
    index_sha256: str


def _object_no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IQIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json(data: bytes, label: str):
    try:
        item = json.loads(data.decode("utf-8"), object_pairs_hook=_object_no_duplicate_keys)
    except (ValueError, UnicodeError) as exc:
        raise IQIntegrityError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(item, dict):
        raise IQIntegrityError(f"{label}: JSON root must be an object")
    return item


def _regular(root: Path, name: str) -> Path:
    if (
        not isinstance(name, str) or not name
        or name != Path(name).name or name in (".", "..")
        or "\\" in name or "/" in name
    ):
        raise IQIntegrityError(f"unsafe file name: {name!r}")
    file = root / name
    if file.is_symlink() or not file.is_file():
        raise IQIntegrityError(f"missing or symlinked checkpoint file: {name}")
    return file


def _checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=_CHUNK_BYTES) as stream:
        for block in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    content = _regular(root, "SHA256SUMS").read_text(encoding="utf-8")
    checksums = {}
    for line in content.splitlines():
        found = _HASH_LINE.fullmatch(line)
        if not found:
            raise IQIntegrityError("SHA256SUMS contains a malformed entry")
        digest, filename = found.groups()
        _regular(root, filename)
        if not filename.endswith(".safetensors") or filename in checksums:
            raise IQIntegrityError(f"invalid/duplicate manifest shard: {filename}")
        checksums[filename] = digest
    if not checksums:
        raise IQIntegrityError("empty SHA256SUMS")
    return checksums


def _header(path: Path, expected: set[str]) -> int:
    """Verify every declared tensor offset, dtype and shape, with no data read."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise IQIntegrityError(f"{path.name}: truncated safetensors prefix")
        header_size = int.from_bytes(header_size_bytes, "little")
        if not 2 <= header_size <= min(_MAX_HEADER_BYTES, size - 8):
            raise IQIntegrityError(f"{path.name}: invalid safetensors header length")
        header = _json(handle.read(header_size), path.name)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
    ):
        raise IQIntegrityError(f"{path.name}: invalid safetensors metadata")
    if set(header) != expected:
        raise IQIntegrityError(
            f"{path.name}: tensor-index disagreement "
            f"(missing={len(expected - set(header))}, extra={len(set(header) - expected)})"
        )
    payload_bytes = size - 8 - header_size
    ranges = []
    for key, item in header.items():
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise IQIntegrityError(f"{path.name}: malformed tensor header: {key}")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if dtype not in _BITS:
            raise IQIntegrityError(f"{path.name}: unsupported dtype for {key}: {dtype}")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise IQIntegrityError(f"{path.name}: malformed tensor shape: {key}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int or x < 0 for x in offsets)):
            raise IQIntegrityError(f"{path.name}: malformed tensor data offsets: {key}")
        begin, end = offsets
        if begin > end or end > payload_bytes:
            raise IQIntegrityError(f"{path.name}: out-of-range tensor data: {key}")
        element_count = math.prod(shape)
        bits = element_count * _BITS[dtype]
        if bits % 8 != 0 or (end - begin) != bits // 8:
            raise IQIntegrityError(f"{path.name}: tensor byte count mismatch: {key}")
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise IQIntegrityError(f"{path.name}: overlapping/gapped tensor payloads")
        cursor = end
    if cursor != payload_bytes:
        raise IQIntegrityError(f"{path.name}: unaccounted safetensors payload bytes")
    return len(header)


def verify_iq_checkpoint_integrity(
    root: str | Path, *, telemetry=None
) -> IQIntegrityReport:
    """Fail closed on any missing or corrupt byte, tensor metadata or hash.

    Read-only, CPU-only. Hashes full shard payloads in bounded-size chunks,
    so a 33+GB checkpoint does not need to fit in system RAM.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise IQIntegrityError("checkpoint root is not a directory")
    index_bytes = _regular(root, "model.safetensors.index.json").read_bytes()
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    index = _json(index_bytes, "model.safetensors.index.json")
    complete = _json(_regular(root, "COMPLETE.json").read_bytes(), "COMPLETE.json")
    if complete.get("index_sha256") != index_hash:
        raise IQIntegrityError("COMPLETE.json index SHA-256 mismatch")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or any(
        not isinstance(k, str) or not k or not isinstance(v, str)
        for k, v in weight_map.items()
    ):
        raise IQIntegrityError("invalid checkpoint weight_map")
    by_shard: dict[str, set[str]] = {}
    for tensor, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(tensor)
    expected_shards = set(by_shard)
    checksums = _manifest(root)
    if set(checksums) != expected_shards:
        raise IQIntegrityError(
            f"SHA256SUMS/index mismatch: missing={len(expected_shards - set(checksums))}, "
            f"extra={len(set(checksums) - expected_shards)}"
        )
    if type(complete.get("shards")) is not int or complete["shards"] != len(expected_shards):
        raise IQIntegrityError("COMPLETE.json shard count mismatch")
    declared_total = index.get("metadata", {}).get("total_size") if isinstance(index.get("metadata"), dict) else None
    if type(declared_total) is not int:
        raise IQIntegrityError("checkpoint index missing integer total_size")
    total_size = 0
    tensor_count = 0
    for position, shard in enumerate(sorted(expected_shards), 1):
        file = _regular(root, shard)
        total_size += file.stat().st_size
        tensors = _header(file, by_shard[shard])
        tensor_count += tensors
        if telemetry:
            telemetry.event("checkpoint.integrity.shard.start", shard=shard, index=position, tensors=tensors)
        actual_sha = _checksum_file(file)
        if actual_sha != checksums[shard]:
            raise IQIntegrityError(f"SHA-256 mismatch for {shard}")
        if telemetry:
            telemetry.event("checkpoint.integrity.shard.verified", shard=shard, index=position, tensors=tensors)
    if total_size != declared_total:
        raise IQIntegrityError(
            f"shard sizes {total_size} do not match index declared total_size {declared_total}"
        )
    if telemetry:
        telemetry.event("checkpoint.integrity.complete", shards=len(expected_shards),
                        tensors=tensor_count, sha256_verified=True, bytes=total_size)
    return IQIntegrityReport(
        shard_count=len(expected_shards), tensor_count=tensor_count,
        shard_bytes=total_size, verified_sha256_count=len(expected_shards),
        verified_header_count=len(expected_shards), index_sha256=index_hash
    )
