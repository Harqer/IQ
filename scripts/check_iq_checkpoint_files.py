"""Filesystem-only IQ checkpoint gate for the next *full inference* job.

This never loads weights or executes any model or CUDA kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    root = args.checkpoint.resolve()
    config = root / "iq_config.json"
    index_path = root / "model.safetensors.index.json"
    manifest_path = root / "transplant_manifest.json"
    for path in (config, index_path, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            p.error(f"Compiled IQ checkpoint is missing {path.name}: {root}")
    try:
        config_data = json.loads(config.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        p.error(f"Unreadable compiled IQ metadata: {exc}")
    if not isinstance(config_data, dict) or not isinstance(manifest, dict):
        p.error("IQ config and transplant manifest must be JSON objects")
    if manifest.get("method") not in {
        "mamba3_foundation_plus_glm53_capability_weight_transfer",
        "direct_cross_architecture_weight_transplant",
    }:
        p.error("Unknown lexical foundation; do not start paid inference")
    if not isinstance(index, dict):
        p.error("Checkpoint index must be a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        p.error("Checkpoint index must declare a nonempty weight_map")
    if not all(isinstance(name, str) and name for name in weight_map):
        p.error("Checkpoint index has malformed parameter names")
    names = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in names):
        p.error("Malformed checkpoint shard names")
    shards = set(names)
    shard_bytes = 0
    for name in sorted(shards):
        relative = Path(name)
        if (
            relative.name != name
            or relative.is_absolute()
            or relative.suffix != ".safetensors"
            or ".." in relative.parts
            or len(relative.parts) != 1
            or "\\" in name
        ):
            p.error(f"Invalid checkpoint shard path: {name}")
        shard = root / name
        if shard.is_symlink() or not shard.is_file() or shard.stat().st_size == 0:
            p.error(f"Missing, empty or symlink safetensors shard: {shard}")
        shard_bytes += shard.stat().st_size
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict) or (
        "total_size" in metadata
        and (
            type(metadata["total_size"]) is not int
            or metadata["total_size"] != shard_bytes
        )
    ):
        p.error("Checkpoint index total_size does not match physical shard bytes")

    # The Mamba-3 foundation supplies the Llama 3.1 vocabulary. The original
    # HF inference job mounted tokenizer assets separately from IQ shards.
    # Never copy or mutate the immutable checkpoint volume.
    tokenizer_dir = Path(os.getenv("IQ_TOKENIZER_DIR") or root)
    if not any(
        (tokenizer_dir / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model")
    ):
        p.error(f"Tokenizer assets unavailable at {tokenizer_dir}")
    print(
        f"IQ checkpoint files available: {len(weight_map)} weights in "
        f"{len(shards)} nonempty shards; complete loader will check tensor coverage"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
