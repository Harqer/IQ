#!/usr/bin/env python3
"""Project 1..64 selected experts from one GLM layer, with a shared WARM basis.

Requires a stage-scoped raw donor bundle. Avoids repeated large basis loading.
Does not write a runnable IQ checkpoint.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from iq_transfer.glm53_single_expert_projection import (
    GLM53SingleExpertProjectionError,
)
from iq_transfer.glm53_stage_projection import project_raw_stage
from scripts.project_glm53_single_expert import _restore_stage_from_hf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--block-rows", type=int, default=64)
    calib = parser.add_mutually_exclusive_group(required=True)
    calib.add_argument("--calibration-dir", type=Path)
    calib.add_argument("--calibration-uri", type=str)
    args = parser.parse_args()
    try:
        def run(path: Path):
            return project_raw_stage(
                raw_dir=args.raw_dir, calibration_dir=path,
                output_dir=args.output, stage=args.stage,
                experts=args.experts, block_rows=args.block_rows,
            )

        if args.calibration_dir is not None:
            result = run(args.calibration_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="iq-glm53-stage-map-") as tmp:
                temp = Path(tmp)
                _restore_stage_from_hf(args.calibration_uri, temp, args.stage)
                result = run(temp)
        print(json.dumps({
            "status": result["status"],
            "stage": result["stage"],
            "donor_layer": result["source_layer"],
            "exported_experts": result["exported_experts"],
            "selected_ids": [x["donor_expert_id"] for x in result["entries"]],
            "energy": {str(x["slot"]): x["energy"] for x in result["entries"]},
            "output": str(args.output.resolve()),
            "not_an_iq_checkpoint": True,
        }, indent=2))
        return 0
    except (GLM53SingleExpertProjectionError, OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"GLM expert stage projection failed: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
