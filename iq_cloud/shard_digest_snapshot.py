"""Produce an untrusted-to-start SHA-256 inventory, only on explicit opt-in.

The source Hugging Face bucket has no independent per-shard digest inventory.
This performs an expensive ~539 GB read, but NO GPU and NO local model install.
It does not modify the bucket and does not store the 539 GB on disk.

The resulting digest manifest is a snapshot, not independent proof that the
original training/inference weights were correct. Review and pin its SHA-256
in the remote loader only after all source files and provenance are audited.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from typing import Callable, Mapping

from .checkpoint_metadata import (
    EXPECTED_INDEX_SHA256, IQ235MetadataError,
)
from .remote_shard_loader import (
    CHUNK, MAX_SHARD, RemoteShardError, verify_safetensors_header,
)

def hash_one_remote_shard(
    fs, uri: str, *,
    expected_keys: set[str],
    max_shard_bytes: int = MAX_SHARD,
    chunk_bytes: int = CHUNK,
) -> tuple[str, int]:
    """Hash a remote source object directly, with no on-disk staging."""
    if not uri.startswith("hf://buckets/") or "/../" in uri:
        raise RemoteShardError("unsafe remote digest source")
    size = fs.info(uri).get("size")
    if type(size) is not int or not 8 < size <= max_shard_bytes:
        raise RemoteShardError("unexpected remote shard size")
    digest = sha256()
    total = 0
    with fs.open(uri, "rb") as source:
        # Inspect the actual SafeTensors structure via bounded header reads.
        verify_safetensors_header(source, file_size=size,
                                  expected_keys=expected_keys)
        source.seek(0)
        while total < size:
            data = source.read(min(size - total, chunk_bytes))
            if not data:
                raise RemoteShardError("remote digest scan encountered truncated shard")
            digest.update(data)
            total += len(data)
        if source.read(1):
            raise RemoteShardError("remote shard grew during checksum scan")
    return digest.hexdigest(), total


def snapshot_bucket_shard_hashes(
    fs, *, base: str, source_index: Mapping[str, object],
    on_shard: Callable[[int, int, int], None] | None = None,
    expected_sha256: str = EXPECTED_INDEX_SHA256,
) -> dict[str, object]:
    """Create exact filename->SHA256 manifest from streaming reads.

    This is NOT the approved digest manifest. Workers need explicit approval
    to scan 539 GB; never run this in routine GitHub CI or preflight.
    """
    index = source_index.get("weight_map")
    if not isinstance(index, dict) or not index:
        raise RemoteShardError("missing source checkpoint weight map")
    grouped: dict[str, set[str]] = defaultdict(set)
    for tensor, shard in index.items():
        if (not isinstance(tensor, str) or not tensor
                or not isinstance(shard, str) or not shard.startswith("model-")
                or not shard.endswith(".safetensors") or "/" in shard or "\\" in shard):
            raise RemoteShardError("invalid source index key or shard path")
        grouped[shard].add(tensor)
    digests = {}
    scanned = 0
    for i, shard in enumerate(sorted(grouped), 1):
        hexdigest, size = hash_one_remote_shard(
            fs, base.rstrip("/")+"/"+shard, expected_keys=grouped[shard],
        )
        scanned += size
        digests[shard] = hexdigest
        if on_shard:
            on_shard(i, len(grouped), scanned)
    return {
        "schema_version": 1,
        "source_index_sha256": expected_sha256,
        "sha256_algorithm": "sha256",
        "model_source_status": "compiled_not_inference_validated",
        "shards_sha256": digests,
        "status": "unreviewed_source_digest_snapshot",
        "full_tensor_finiteness_checked": False,
        "lexical_alignment_verified": False,
        "inference_ready": False,
    }
