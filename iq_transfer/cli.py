from __future__ import annotations

import argparse
import json

from .job import run_phi_dense_transfer
from .glm53_job import validate_glm53_donor
from .mamba3_direct import compile_official_mamba3_mimo_15b_transplant


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m iq_transfer.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    glm = sub.add_parser(
        "glm53-validate",
        help="validate and fingerprint a local GLM-5.3 donor snapshot",
    )
    glm.add_argument("--checkpoint", required=True)
    glm.add_argument("--checkpoint-revision", required=True)
    glm.add_argument("--donor-license", required=True)
    glm.add_argument("--source-uri")
    glm.add_argument("--manifest-output")
    glm.add_argument(
        "--allow-quantized",
        action="store_true",
        help="allow an FP8/quantized checkpoint for inspection only",
    )

    mamba = sub.add_parser(
        "mamba3-direct",
        help="compile the pinned official Mamba-3 MIMO 1.5B weights into IQ Mamba-3 slots without distillation",
    )
    mamba.add_argument("--checkpoint", required=True)
    mamba.add_argument("--recipient-config", required=True)
    mamba.add_argument("--output", required=True)
    mamba.add_argument(
        "--checkpoint-revision",
        default="bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec",
    )
    mamba.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="skip the 3 GB donor SHA-256 pass (not recommended for production)",
    )

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
    if args.command == "glm53-validate":
        artifact = validate_glm53_donor(
            checkpoint=args.checkpoint,
            checkpoint_revision=args.checkpoint_revision,
            donor_license=args.donor_license,
            source_uri=args.source_uri,
            require_bf16=not args.allow_quantized,
            manifest_output=args.manifest_output,
        )
        print(
            json.dumps(
                {
                    "checkpoint_dir": str(artifact.checkpoint_dir),
                    "donor_fingerprint": artifact.manifest.fingerprint,
                    "checkpoint_hash": artifact.manifest.checkpoint_hash,
                    "num_layers": artifact.manifest.num_layers,
                    "hidden_size": artifact.manifest.hidden_size,
                    "vocab_size": artifact.manifest.vocab_size,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "mamba3-direct":
        result = compile_official_mamba3_mimo_15b_transplant(
            checkpoint=args.checkpoint,
            recipient_config_path=args.recipient_config,
            output_dir=args.output,
            checkpoint_revision=args.checkpoint_revision,
            verify_checkpoint_hash=not args.skip_checkpoint_hash,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "donor_sha256": result.donor_sha256,
                    "recipient_fingerprint": result.recipient_fingerprint,
                    "transplanted_layers": len(result.placements),
                    "identity_layers": list(result.identity_physical_layers),
                },
                sort_keys=True,
            )
        )
        return 0
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
