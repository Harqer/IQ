"""Filesystem-only IQ checkpoint gate for the next *full inference* job.

This never loads weights or executes any model or CUDA kernels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    root = args.checkpoint.resolve()
    config = root / "iq_config.json"
    index_path = root / "model.safetensors.index.json"
    for path in (config, index_path):
        if not path.is_file() or path.stat().st_size == 0:
            p.error(f"Compiled IQ checkpoint is missing {path.name}: {root}")
    try:
        json.loads(config.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        p.error(f"Unreadable compiled IQ metadata: {exc}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        p.error("Checkpoint index must declare a nonempty weight_map")
    if not all(isinstance(name, str) and name for name in weight_map):
        p.error("Checkpoint index has malformed parameter names")
    names = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in names):
        p.error("Malformed checkpoint shard names")
    shards = set(names)
    for name in sorted(shards):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            p.error(f"Invalid checkpoint shard path: {name}")
        shard = root / name
        if not shard.is_file() or shard.stat().st_size == 0:
            p.error(f"Missing or empty safetensors shard: {shard}")

    # Actual tokenizer semantics are checked by AutoTokenizer on inference.
    if not any(
        (root / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    ):
        p.error("Compiled IQ checkpoint has no tokenizer.json/tokenizer.model")
    print(
        f"IQ checkpoint files available: {len(weight_map)} weights in "
        f"{len(shards)} nonempty shards; complete loader will check tensor coverage"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
