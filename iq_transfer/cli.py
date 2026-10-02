from __future__ import annotations

import argparse
import json

import torch

from .job import run_phi_dense_transfer
from .glm53_job import validate_glm53_donor
from .mamba3_direct import compile_official_mamba3_mimo_15b_transplant
from .complete_transplant import compile_complete_iq_checkpoint
from .gpt_oss20b import GPT_OSS_20B_REVISION
from .bias_filter import (
    detect_magnitude_outlier_dimensions,
    fit_leace_bias_filter,
    paired_counterfactual_batch,
)
from .capture import load_capture_records


def _capture_tap_matrix(path: str, tap: str, pooling: str) -> torch.Tensor:
    records = load_capture_records(path)
    if tap not in records:
        raise RuntimeError(f"capture artifact has no tap {tap!r}: {path}")
    rows: list[torch.Tensor] = []
    for value in records[tap]:
        if value.ndim == 2:
            row = value
        elif value.ndim == 3:
            if pooling == "last":
                row = value[:, -1, :]
            elif pooling == "mean":
                row = value.mean(dim=1)
            else:
                raise RuntimeError(f"unsupported pooling mode: {pooling}")
        else:
            raise RuntimeError(
                f"bias-fit expects [batch,hidden] or [batch,seq,hidden] captures; "
                f"got {tuple(value.shape)}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"capture tap {tap!r} is empty: {path}")
    return torch.cat(rows, dim=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m iq_transfer.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    bias = sub.add_parser(
        "bias-fit",
        help="fit a shrinkage-LEACE candidate from paired activation captures",
    )
    bias.add_argument("--fit-a", required=True)
    bias.add_argument("--fit-b", required=True)
    bias.add_argument("--validation-a", required=True)
    bias.add_argument("--validation-b", required=True)
    bias.add_argument("--tap", required=True)
    bias.add_argument("--target-concept", required=True)
    bias.add_argument("--space-name", required=True)
    bias.add_argument("--output", required=True)
    bias.add_argument("--pooling", choices=("last", "mean"), default="last")
    bias.add_argument("--protect-magnitude-outliers", action="store_true")
    bias.add_argument("--outlier-mad-threshold", type=float, default=8.0)
    bias.add_argument("--protected-modality", action="append", default=[])
    bias.add_argument("--required-capability-metric", action="append")
    bias.add_argument("--svd-tol", type=float, default=0.01)
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

    complete = sub.add_parser(
        "complete-iq",
        help="compile a complete IQ checkpoint from gpt-oss-20b plus official Mamba-3 MIMO weights",
    )
    complete.add_argument("--gpt-oss-original", required=True)
    complete.add_argument("--mamba3-checkpoint", required=True)
    complete.add_argument("--output", required=True)
    complete.add_argument("--gpt-oss-revision", default=GPT_OSS_20B_REVISION)
    complete.add_argument(
        "--mamba3-revision",
        default="bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec",
    )
    complete.add_argument(
        "--skip-checkpoint-hashes",
        action="store_true",
        help="skip donor SHA-256 verification (not recommended)",
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
    if args.command == "bias-fit":
        fit_a = _capture_tap_matrix(args.fit_a, args.tap, args.pooling)
        fit_b = _capture_tap_matrix(args.fit_b, args.tap, args.pooling)
        validation_a = _capture_tap_matrix(
            args.validation_a, args.tap, args.pooling
        )
        validation_b = _capture_tap_matrix(
            args.validation_b, args.tap, args.pooling
        )

        fit_x, fit_z = paired_counterfactual_batch(fit_a, fit_b)
        validation_x, validation_z = paired_counterfactual_batch(
            validation_a, validation_b
        )
        protected = None
        if args.protect_magnitude_outliers:
            protected = detect_magnitude_outlier_dimensions(
                torch.cat((fit_a, fit_b), dim=0),
                mad_threshold=args.outlier_mad_threshold,
            )
        required_metrics = tuple(
            args.required_capability_metric or ("coding", "nlp")
        )
        artifact, metrics = fit_leace_bias_filter(
            fit_x,
            fit_z,
            validation_x=validation_x,
            validation_z=validation_z,
            target_concept=args.target_concept,
            space_name=args.space_name,
            protected_dimensions=protected,
            protected_modalities=tuple(args.protected_modality),
            required_capability_metrics=required_metrics,
            svd_tol=args.svd_tol,
        )
        artifact.write(args.output)
        print(
            json.dumps(
                {
                    "output": args.output,
                    "artifact_fingerprint": artifact.fingerprint,
                    "target_concept": artifact.target_concept,
                    "space_name": artifact.space_name,
                    "protected_dimensions": artifact.protected_dimension_count,
                    "protected_modalities": list(artifact.protected_modalities),
                    "required_capability_metrics": list(
                        artifact.required_capability_metrics
                    ),
                    "leakage_before": metrics.leakage_before,
                    "leakage_after": metrics.leakage_after,
                    "leakage_reduction": metrics.leakage_reduction,
                    "relative_mse": metrics.relative_mse,
                    "mean_cosine": metrics.mean_cosine,
                    "approved": False,
                },
                sort_keys=True,
            )
        )
        return 0
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
            output_dir=args.output,
            checkpoint_revision=args.checkpoint_revision,
            verify_checkpoint_hash=not args.skip_checkpoint_hash,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "donor_sha256": result.donor_sha256,
                    "target_fingerprint": result.target_fingerprint,
                    "transplanted_layers": len(result.placements),
                    "identity_mamba_ordinals": list(result.identity_mamba_ordinals),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "complete-iq":
        output = compile_complete_iq_checkpoint(
            gpt_oss_original_dir=args.gpt_oss_original,
            mamba3_checkpoint_dir=args.mamba3_checkpoint,
            output_dir=args.output,
            gpt_oss_revision=args.gpt_oss_revision,
            mamba3_revision=args.mamba3_revision,
            verify_hashes=not args.skip_checkpoint_hashes,
        )
        print(json.dumps({"output_dir": str(output)}, sort_keys=True))
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
