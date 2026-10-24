#!/usr/bin/env python3
"""Project one real GLM donor expert into IQ's saved WARM latent subspace.

Smoke default: stage 1, slot 0. Writes only three projected MoE tensors.
Accepts the canonical local calibration bundle or its private HF bucket URI.
Does not create or overwrite an IQ checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from iq_transfer.glm53_single_expert_projection import (
    GLM53SingleExpertProjectionError,
    project_one_raw_expert,
)


def _restore_stage_from_hf(uri: str, directory: Path, stage: int) -> None:
    if not uri.startswith("hf://"):
        raise GLM53SingleExpertProjectionError(
            "remote calibration must be an explicit hf:// URI"
        )
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    prefix = uri.rstrip("/")
    stem = f"stage-{stage:02d}-latent"
    for name in (
        "glm53_calibration.json",
        ".calibration_integrity.json",
        f"{stem}.json",
        f"{stem}.safetensors",
    ):
        # Copy selected calibration bytes only, never the entire HF bucket.
        with fs.open(f"{prefix}/{name}", "rb") as source:
            with (directory / name).open("xb") as dest:
                shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path.home() / "iq-glm53-raw-smoke-lcz6_oln",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--calibration-dir", type=Path,
        help="canonical saved calibration bundle directory",
    )
    sources.add_argument(
        "--calibration-uri", type=str,
        help="HF bucket URI, e.g. hf://buckets/Visionboxx/macha/calibration",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--block-rows", type=int, default=64)
    args = parser.parse_args()
    try:
        if args.calibration_uri is not None:
            with tempfile.TemporaryDirectory(prefix="iq-single-expert-map-") as tmp:
                path = Path(tmp)
                _restore_stage_from_hf(args.calibration_uri, path, args.stage)
                result = project_one_raw_expert(
                    raw_dir=args.raw_dir, calibration_dir=path,
                    output_dir=args.output, stage=args.stage,
                    slot=args.slot, block_rows=args.block_rows,
                )
        else:
            result = project_one_raw_expert(
                raw_dir=args.raw_dir,
                calibration_dir=(
                    args.calibration_dir
                    if args.calibration_dir is not None
                    else Path.home() / "iq-weight-transfer" / "calibration"
                ),
                output_dir=args.output, stage=args.stage, slot=args.slot,
                block_rows=args.block_rows,
            )
        print(json.dumps({
            "status": "projected_expert_only",
            "donor_expert": result["donor_expert_id"],
            "stage": result["stage"],
            "recipient_slot": result["recipient_slot"],
            "recipient_moe_layer": result["recipient_moe_physical_layer"],
            "target_latent_size": result["target_latent_size"],
            "energy": {
                k: v["subspace_retained_energy"]
                for k, v in result["projection"].items()
            },
            "output": str(args.output.resolve()),
            "not_an_iq_checkpoint": True,
        }, indent=2))
        return 0
    except (OSError, ValueError, KeyError, GLM53SingleExpertProjectionError) as exc:
        parser.exit(2, f"GLM expert projection failed: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
