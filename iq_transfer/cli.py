from __future__ import annotations

import argparse
import json

from .job import run_phi_dense_transfer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m iq_transfer.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    phi = sub.add_parser("phi-dense", help="build a transported dense IQ artifact from a local Phi checkpoint")
    phi.add_argument("--checkpoint", required=True)
    phi.add_argument("--recipient-config", required=True)
    phi.add_argument("--fit-batches", required=True)
    phi.add_argument("--validation-batches", required=True)
    phi.add_argument("--calibration-manifest", required=True)
    phi.add_argument("--output", required=True)
    phi.add_argument("--checkpoint-revision", required=True)
    phi.add_argument("--donor-license", required=True)
    phi.add_argument("--source-uri")
    phi.add_argument("--device", default="cpu")
    phi.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
    )
    phi.add_argument("--dora-rank", type=int, default=64)
    phi.add_argument("--dora-alpha", type=float)
    phi.add_argument("--ridge", type=float, default=1e-3)
    phi.add_argument("--shadow-measurements", type=int, default=256)
    phi.add_argument("--shadow-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "phi-dense":
        result = run_phi_dense_transfer(
            checkpoint=args.checkpoint,
            recipient_config_path=args.recipient_config,
            fit_batches_path=args.fit_batches,
            validation_batches_path=args.validation_batches,
            calibration_manifest_path=args.calibration_manifest,
            output_dir=args.output,
            checkpoint_revision=args.checkpoint_revision,
            donor_license=args.donor_license,
            source_uri=args.source_uri,
            device=args.device,
            dtype=args.dtype,
            dora_rank=args.dora_rank,
            dora_alpha=args.dora_alpha,
            ridge=args.ridge,
            shadow_measurements=args.shadow_measurements,
            shadow_seed=args.shadow_seed,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "plan_fingerprint": result.transport_plan.fingerprint,
                    "donor_fingerprint": result.donor_manifest.fingerprint,
                    "layer_mapping": {str(k): int(v) for k, v in sorted(result.layer_mapping.items())},
                    "dora_paths": list(result.dora_paths),
                },
                sort_keys=True,
            )
        )
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
