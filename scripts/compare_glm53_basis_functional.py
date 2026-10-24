#!/usr/bin/env python3
"""Compare the original and expert-informed WARM maps on identical GLM inputs.

Results quantify isolated SwiGLU FFN fidelity, not IQ model quality.
No checkpoint, activation, or WARM map is modified. The --report JSON is new.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from iq_transfer.glm53_basis_functional_compare import (
    ExpertBasisFunctionalError,
    compare_basis_functions,
)
from scripts.project_glm53_single_expert import _restore_stage_from_hf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--activations", type=Path, help="optional real pre-expert GLM [N,6144] .npy")
    p.add_argument("--report", type=Path, required=True,
                   help="NEW output JSON file; existing report will not be overwritten")
    c = p.add_mutually_exclusive_group(required=True)
    c.add_argument("--calibration-dir", type=Path)
    c.add_argument("--calibration-uri", type=str)
    args = p.parse_args()

    def evaluate(path: Path):
        return compare_basis_functions(
            raw_dir=args.raw_dir,
            baseline_dir=args.baseline_dir,
            candidate_dir=args.candidate_dir,
            calibration_dir=path,
            stage=args.stage,
            samples=args.samples,
            seed=args.seed,
            captured_activations=args.activations,
        )

    try:
        if args.calibration_dir is not None:
            report = evaluate(args.calibration_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="glm53-functional-map-") as temp:
                root = Path(temp)
                _restore_stage_from_hf(args.calibration_uri, root, args.stage)
                report = evaluate(root)

        # Print the summary so there is no need to paste 12 large matrices.
        compact = {
            "input_origin": report["input_origin"],
            "validation_donor_id": report["source_expert_ids"][
                report["validation_slot"]
            ],
            "summary": report["result"],
            "per_expert": [
                {
                    "donor_id": entry["donor_expert_id"],
                    "role": entry["role"],
                    "relative_rmse_improvement": {
                        key: value["relative_rmse_improvement"]
                        for key, value in entry["scenarios"].items()
                    },
                }
                for entry in report["experts"]
            ],
            "report": str(args.report.resolve()),
            "canonical_calibration_modified": False,
        }
        with args.report.open("x", encoding="utf-8") as out:
            json.dump(report, out, sort_keys=True, indent=2)
            out.write("\n")
        print(json.dumps(compact, indent=2))
        return 0
    except (ExpertBasisFunctionalError, OSError, KeyError, ValueError) as exc:
        p.exit(2, f"GLM basis functional comparison failed: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
