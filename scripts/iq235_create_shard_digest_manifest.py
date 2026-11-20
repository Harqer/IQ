"""Non-GPU snapshot of remote 539GB shards, with explicit opt-in only.

Default behavior reads NO weight bytes. --scan requires a deliberate worker
environment setting and a NEW local manifest output destination. Do not run
from your laptop: use an authorized remote worker after cost review.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from iq_cloud.checkpoint_metadata import (
    BUCKET, RUN, EXPECTED_INDEX_SHA256, EXPECTED_SHARDS, EXPECTED_BYTES,
    check_pinned_metadata_sha256,
)
from iq_cloud.shard_digest_snapshot import snapshot_bucket_shard_hashes
from iq_cloud.remote_shard_loader import validate_digest_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true",
                        help="Explicitly read all ~539GB remote shard payloads")
    parser.add_argument("--output", type=Path,
                        help="Write a NEW JSON digest snapshot to remote-worker disk")
    args = parser.parse_args()
    if not args.scan:
        print(json.dumps({
            "stage": "requires_explicit_remote_scan_approval",
            "source_bucket": BUCKET,
            "source_run": RUN,
            "indexed_shards": EXPECTED_SHARDS,
            "indexed_weight_bytes": EXPECTED_BYTES,
            "weight_bytes_read": 0,
            "source_weights_downloaded": False,
            "trusted_complete_shard_manifest_available": False,
            "ready_for_pretrained_inference": False,
        }, indent=2))
        return
    if (os.getenv("IQ235_ALLOW_539GB_REMOTE_READ") != "I_AUTHORIZE_539GB"
            or args.output is None or args.output.exists()
            or args.output.is_symlink()):
        parser.error(
            "539GB remote scan requires IQ235_ALLOW_539GB_REMOTE_READ=I_AUTHORIZE_539GB "
            "and an unused --output path on the remote worker"
        )
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(block_size=4 * 1024 * 1024)
    base = f"hf://buckets/{BUCKET}/{RUN}"
    def read(name, bound):
        with fs.open(f"{base}/{name}", "rb") as inp:
            payload = inp.read(bound+1)
        if len(payload) > bound:
            raise RuntimeError("checkpoint source metadata exceeds size cap")
        return payload
    config = read("iq_config.json", 100000)
    index_bytes = read("model.safetensors.index.json", 8000000)
    check_pinned_metadata_sha256(config, index_bytes)
    index = json.loads(index_bytes)
    hashes = snapshot_bucket_shard_hashes(
        fs, base=base, source_index=index,
        on_shard=lambda done,total,read: print(
            f"verified header / hash-read shard={done}/{total}, bytes={read}",
            flush=True,
        ) if done % 250 == 0 or done == total else None,
    )
    validate_digest_manifest(
        hashes, set(index["weight_map"].values()),
        source_index_digest=EXPECTED_INDEX_SHA256,
    )
    output = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as out:
        out.write(output)
    print(json.dumps({
        "snapshot_sha256": sha256(output).hexdigest(),
        "shard_count": len(hashes["shards_sha256"]),
        "reviewed_or_trusted": False,
        "trained_lexical_weights": False,
        "ready_for_pretrained_inference": False,
        "original_bucket_modified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
