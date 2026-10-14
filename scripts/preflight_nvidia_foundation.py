#!/usr/bin/env python3
"""Inspect NVIDIA's real base checkpoint metadata without downloading model weights.

This command is a mandatory preflight, not an implementation of the
NVIDIA-to-IQ weight conversion. No donor weights are modified or published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iq_transfer.nvidia_foundation import (
    NVIDIA_BASE_REPO,
    plan_foundation,
    serialize_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Location for the pinned NVIDIA donor compatibility manifest.",
    )
    parser.add_argument(
        "--revision", default="main",
        help="A public HF branch or commit; the resulting manifest always pins the resolved SHA.",
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        parser.error("Install huggingface_hub to read the actual NVIDIA model")
        raise AssertionError from exc

    info = HfApi().model_info(NVIDIA_BASE_REPO, revision=args.revision)
    revision = info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not resolve a pinned checkpoint SHA")
    config_path = Path(
        hf_hub_download(
            repo_id=NVIDIA_BASE_REPO, filename="config.json", revision=revision
        )
    )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    result = plan_foundation(config, donor_revision=revision, config_bytes=config_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialize_plan(result), encoding="utf-8")
    print(f"PASS: NVIDIA base checkpoint metadata verified at commit {revision}")
    print(f"Metadata plan: {args.output}")
    print("No weight tensors were downloaded, converted, or uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
