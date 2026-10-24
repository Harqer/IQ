#!/usr/bin/env python3
"""Audit one projected GLM expert with optional real donor hidden-state captures.

Without --activations, generates labeled synthetic in-/off-subspace stress
inputs. Synthetic results are NOT evidence of pretrained model capability.
No model checkpoints or activation tensors are written.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from iq_transfer.glm53_expert_fidelity import (
    GLM53FidelityError,
    audit_expert,
)
from scripts.project_glm53_single_expert import _restore_stage_from_hf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--projected-dir", type=Path, required=True)
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration-dir", type=Path)
    calibration.add_argument("--calibration-uri", type=str)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument(
        "--activations", type=Path,
        help="actual donor pre-expert hidden states, shape [N, 6144], NumPy .npy",
    )
    parser.add_argument(
        "--report", type=Path,
        help="optional NEW JSON output; existing files are never overwritten",
    )
    args = parser.parse_args()

    def evaluate(calibration_dir: Path):
        return audit_expert(
            raw_dir=args.raw_dir,
            projected_dir=args.projected_dir,
            calibration_dir=calibration_dir,
            stage=args.stage,
            slot=args.slot,
            samples=args.samples,
            seed=args.seed,
            captured_activations=args.activations,
        )

    try:
        if args.calibration_dir is not None:
            report = evaluate(args.calibration_dir)
        else:
            with tempfile.TemporaryDirectory(
                prefix="iq-glm53-fidelity-calibration-"
            ) as tmp:
                path = Path(tmp)
                _restore_stage_from_hf(args.calibration_uri, path, args.stage)
                report = evaluate(path)
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GLM53FidelityError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"GLM expert fidelity audit failed: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
