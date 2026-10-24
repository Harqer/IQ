#!/usr/bin/env python3
"""Explore a shared expert-informed WARM basis using four selected GLM experts.

One candidate, not a calibration update: train on three routed experts and
report fourth as validation. No full donor attention, checkpoints, or SVD.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from iq_transfer.glm53_expert_informed_basis import (
    ExpertBasisProbeError,
    run_expert_informed_probe,
)
from scripts.project_glm53_single_expert import _restore_stage_from_hf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--candidate-columns", type=int, default=16)
    p.add_argument("--validation-slot", type=int, default=3)
    p.add_argument("--seed", type=int, default=77)
    calibration = p.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration-dir", type=Path)
    calibration.add_argument("--calibration-uri", type=str)
    args = p.parse_args()

    def run(path: Path):
        return run_expert_informed_probe(
            raw_dir=args.raw_dir,
            baseline_dir=args.baseline_dir,
            calibration_dir=path,
            output_dir=args.output,
            stage=args.stage,
            candidate_columns=args.candidate_columns,
            validation_slot=args.validation_slot,
            seed=args.seed,
        )

    try:
        if args.calibration_dir is not None:
            report = run(args.calibration_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="glm53-probe-map-") as tmp:
                path = Path(tmp)
                _restore_stage_from_hf(args.calibration_uri, path, args.stage)
                report = run(path)
        print(json.dumps({
            "status": "candidate_weight_energy_probe_only",
            "stage": report["stage"],
            "modified_columns": len(report["modified_basis_columns"]),
            "rotation_radians": report["selected_angle_radians"],
            "training_mean_energy_change_percent_points": (
                100 * report["candidate_results"]["train_mean_energy_delta"]
            ),
            "per_expert": [
                {
                    "donor_id": row["donor_id"],
                    "role": row["role"],
                    "change_percent_points": {
                        kind: 100 * delta
                        for kind, delta in row["delta"].items()
                    },
                    "mean_change_percent_points": 100 * row["mean_delta"],
                }
                for row in report["candidate_results"]["experts"]
            ],
            "candidate_only": True,
            "output": str(args.output.resolve()),
        }, indent=2))
        return 0
    except (ExpertBasisProbeError, OSError, KeyError, ValueError) as exc:
        p.exit(2, f"GLM expert-informed basis probe failed: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
