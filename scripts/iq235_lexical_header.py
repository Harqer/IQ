"""Inspect only the SafeTensors header of IQ235's original lexical shard.

No checkpoints are downloaded locally. For remote access, the filesystem can
buffer a small adjacent block; no tensor data is decoded or returned.
"""
from __future__ import annotations

import json
from pathlib import Path

from iq_cloud.lexical_header import inspect_remote_global_lexical_shard


def main():
    original = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/iq235_checkpoint_source.json").read_text()
    )
    print(json.dumps(inspect_remote_global_lexical_shard(original), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
