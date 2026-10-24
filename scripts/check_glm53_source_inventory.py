#!/usr/bin/env python3
"""Read-only GLM-5.3 donor index audit before 64-expert compilation.

82/82 local shards can still represent a subset of the original donor.
Only small config/index metadata is downloaded; no model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

REPO = "zai-org/GLM-5.3-BF16"
REVISION = "abf42e6c0e53bc29a875511301e7b7fcd220315b"


def compare_inventory(local_index: dict, official_index: dict, available_files: set[str]) -> dict:
    ours = local_index.get("weight_map")
    full = official_index.get("weight_map")
    if not isinstance(ours, dict) or not isinstance(full, dict) or not ours or not full:
        raise ValueError("donor indices must contain nonempty weight_map objects")
    if not all(isinstance(k, str) and isinstance(v, str) for data in (ours, full)
               for k, v in data.items()):
        raise ValueError("invalid donor tensor key / shard path")
    missing_tensors = sorted(set(full) - set(ours))
    extra_tensors = sorted(set(ours) - set(full))
    altered_mapping = sorted(
        key for key in set(ours) & set(full) if ours[key] != full[key]
    )
    local_shards = set(ours.values())
    present = local_shards & available_files
    missing_shards = sorted(local_shards - present)
    full_shards = set(full.values())
    missing_official_shards = sorted(full_shards - available_files)
    return {
        "status": (
            "complete_pinned_index_and_shards"
            if not (missing_tensors or extra_tensors or altered_mapping
                    or missing_official_shards)
            else "incomplete_or_modified_donor_inventory"
        ),
        "local_index_tensors": len(ours),
        "official_index_tensors": len(full),
        "local_index_shards": len(local_shards),
        "present_local_index_shards": len(present),
        "official_shards": len(full_shards),
        "present_official_shards": len(full_shards) - len(missing_official_shards),
        "missing_official_tensors": len(missing_tensors),
        "extra_local_tensors": len(extra_tensors),
        "changed_shard_assignments": len(altered_mapping),
        "missing_local_shards": missing_shards[:10],
        "first_missing_official_shards": missing_official_shards[:10],
        "first_missing_official_tensors": missing_tensors[:10],
        "note": "Metadata consistency does not verify donor shard contents. Never assume 82/82 is all of GLM.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-donor", type=Path, required=True)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--repo-id", default=REPO)
    args = p.parse_args()
    try:
        if args.repo_id != REPO or args.revision != REVISION:
            raise ValueError("canonical transfer requires pinned GLM BF16 revision")
        from_official = hf_hub_download(
            repo_id=args.repo_id,
            filename="model.safetensors.index.json",
            revision=args.revision,
        )
        root = args.local_donor
        local = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        official = json.loads(Path(from_official).read_text(encoding="utf-8"))
        files = {p.name for p in root.glob("*.safetensors") if p.is_file()}
        result = compare_inventory(local, official, files)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "complete_pinned_index_and_shards" else 2
    except (OSError, ValueError) as exc:
        print(f"Donor inventory preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
