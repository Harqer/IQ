"""Fail-closed streaming SafeTensors verification on an EPHEMERAL remote worker.

Only a single shard is staged at a time. The original source remains in its
Hugging Face bucket. No code path from this module can write to that bucket.
A pinned digest manifest, external to the model index, is mandatory; metadata
alone cannot establish that the 539 GB model is intact.

The caller MUST run this on an authorized remote worker with enough scratch
space for the largest individual shard, not on the user's local computer.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import tempfile
from typing import BinaryIO, Iterator, Mapping

MAX_HEADER = 8 * 1024 * 1024
MAX_SHARD = 4 * 1024 ** 3
CHUNK = 4 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Data widths are bytes; do not infer dtype using torch's native alignment.
ITEM_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class RemoteShardError(ValueError):
    """An unverified, corrupt, incomplete or dangerous checkpoint shard."""


def _read_exact(reader: BinaryIO, count: int) -> bytes:
    payload = reader.read(count)
    if len(payload) != count:
        raise RemoteShardError("truncated SafeTensors header")
    return payload


def verify_safetensors_header(
    reader: BinaryIO, *, file_size: int,
    expected_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, dict]:
    """Check real header geometry and contiguous byte spans WITHOUT payload reads."""
    if type(file_size) is not int or not 8 < file_size <= MAX_SHARD:
        raise RemoteShardError("shard size invalid or exceeds worker cap")
    prefix = _read_exact(reader, 8)
    header_len = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_len <= min(MAX_HEADER, file_size - 8):
        raise RemoteShardError("SafeTensors header length invalid")
    try:
        header = json.loads(_read_exact(reader, header_len))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RemoteShardError("invalid SafeTensors header JSON") from exc
    if not isinstance(header, dict):
        raise RemoteShardError("SafeTensors header must be an object")
    if "__metadata__" in header:
        meta = header.pop("__metadata__")
        if not isinstance(meta, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in meta.items()
        ):
            raise RemoteShardError("invalid SafeTensors auxiliary metadata")
    if not header or (expected_keys is not None and set(header) != set(expected_keys)):
        raise RemoteShardError("SafeTensors tensor names disagree with index")
    offset = 0
    spans = []
    for name, record in header.items():
        if (not isinstance(name, str) or not name
                or not isinstance(record, dict)
                or record.get("dtype") not in ITEM_BYTES):
            raise RemoteShardError("invalid SafeTensors tensor name or dtype")
        dims, loc = record.get("shape"), record.get("data_offsets")
        if (not isinstance(dims, list)
                or any(type(d) is not int or d < 0 for d in dims)
                or not isinstance(loc, list) or len(loc) != 2
                or any(type(v) is not int or v < 0 for v in loc)):
            raise RemoteShardError("invalid SafeTensors tensor dimensions/offsets")
        start, stop = loc
        if stop < start or stop - start != math.prod(dims) * ITEM_BYTES[record["dtype"]]:
            raise RemoteShardError("SafeTensors tensor payload byte count mismatch")
        spans.append((start, stop))
    for start, stop in sorted(spans):
        if start != offset:
            raise RemoteShardError("SafeTensors tensors overlap or have gaps")
        offset = stop
    if 8 + header_len + offset != file_size:
        raise RemoteShardError("SafeTensors header/file byte length mismatch")
    return header


def validate_digest_manifest(
    manifest: Mapping[str, object],
    indexed_shards: set[str],
    *,
    source_index_digest: str,
) -> dict[str, str]:
    """Require independently anchored checksum expectations for ALL source shards.

    Supplying hashes based on a current live download is a *snapshot*, not
    independent evidence that the weights match the original donor. A trusted
    review must pin this manifest before enabling production loading.
    """
    if (not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != 1
            or manifest.get("source_index_sha256") != source_index_digest
            or manifest.get("sha256_algorithm") != "sha256"
            or manifest.get("model_source_status") != "compiled_not_inference_validated"):
        raise RemoteShardError("untrusted or incompatible digest manifest")
    hashes = manifest.get("shards_sha256")
    if not isinstance(hashes, dict) or set(hashes) != indexed_shards:
        raise RemoteShardError("digest manifest must cover every indexed shard exactly")
    for name, digest in hashes.items():
        if (not isinstance(name, str) or not name.startswith("model-")
                or not name.endswith(".safetensors")
                or "/" in name or "\\" in name
                or not isinstance(digest, str) or SHA256.fullmatch(digest) is None):
            raise RemoteShardError("unsafe shard name or invalid SHA-256 digest")
    return dict(hashes)


@contextmanager
def verified_remote_shard(
    filesystem, uri: str, expected_sha256: str, *,
    expected_keys: set[str] | frozenset[str],
    scratch_dir: str | None = None,
    max_shard_bytes: int = MAX_SHARD,
    chunk_bytes: int = CHUNK,
) -> Iterator[tuple[Path, dict[str, dict]]]:
    """Stream one shard into ephemeral remote scratch; verify before yielding.

    A caller cannot read/use the staged model tensor before its checksum,
    length, and header are fully verified. Scratch is deleted on exit/failure.
    """
    if (not isinstance(uri, str) or not uri.startswith("hf://buckets/")
            or not uri.endswith(".safetensors") or ".." in uri.split("/")
            or not isinstance(expected_sha256, str)
            or SHA256.fullmatch(expected_sha256) is None):
        raise RemoteShardError("unsafe source URI or invalid pinned digest")
    if (type(max_shard_bytes) is not int or not 8 < max_shard_bytes <= MAX_SHARD
            or type(chunk_bytes) is not int or not 0 < chunk_bytes <= CHUNK):
        raise RemoteShardError("invalid streaming bounds")
    info = filesystem.info(uri)
    size = info.get("size")
    if type(size) is not int or not 8 < size <= max_shard_bytes:
        raise RemoteShardError("source shard size is missing or exceeds cap")
    with tempfile.TemporaryDirectory(dir=scratch_dir, prefix="iq235-shard-") as tmp:
        staged = Path(tmp) / "model.safetensors"
        digest = sha256()
        copied = 0
        with filesystem.open(uri, "rb") as remote, staged.open("xb") as output:
            while copied < size:
                chunk = remote.read(min(chunk_bytes, size - copied))
                if not chunk:
                    raise RemoteShardError("remote stream ended before expected size")
                copied += len(chunk)
                digest.update(chunk)
                output.write(chunk)
            if remote.read(1):
                raise RemoteShardError("remote stream exceeds recorded shard size")
        if copied != size or digest.hexdigest() != expected_sha256:
            raise RemoteShardError("remote shard length or SHA-256 integrity mismatch")
        with staged.open("rb") as check:
            header = verify_safetensors_header(
                check, file_size=size, expected_keys=expected_keys,
            )
        yield staged, header
