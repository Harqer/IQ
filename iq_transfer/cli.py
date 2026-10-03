from __future__ import annotations

import argparse
import json

from .job import run_phi_dense_transfer
from .glm53_job import validate_glm53_donor, validate_glm53_streaming_donor
from .glm53 import GLM53Inspector
from .glm53_calibration import bootstrap_glm53_calibration
from .glm53_compile import compile_glm53_iq_checkpoint
from .glm53_shards import plan_glm53_bootstrap_shards, plan_glm53_compile_shards
from .capture_runner import load_activation_bundle
from .checkpoint import SafetensorsSource
from .complete_transplant import canonical_glm53_config
from .mamba3_direct import compile_official_mamba3_mimo_15b_transplant
from .complete_transplant import compile_complete_iq_checkpoint
from .gpt_oss20b import GPT_OSS_20B_REVISION


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

    glm_bootstrap = sub.add_parser(
        "glm53-bootstrap-calibration",
        help="build donor-only GLM-5.3 -> IQ bootstrap maps from captured GLM activations",
    )
    glm_bootstrap.add_argument("--checkpoint", required=True)
    glm_bootstrap.add_argument("--source-activations", required=True)
    glm_bootstrap.add_argument("--output", required=True)
    glm_bootstrap.add_argument("--checkpoint-revision", required=True)
    glm_bootstrap.add_argument("--donor-license", required=True)
    glm_bootstrap.add_argument("--source-uri")
    glm_bootstrap.add_argument(
        "--streaming-source",
        action="store_true",
        help="validate from config/index metadata and require only planned local shards",
    )
    glm_bootstrap.add_argument(
        "--warm-device",
        default="cpu",
        help="device used for WARM Gram/eigendecomposition, e.g. cuda",
    )

    glm_compile = sub.add_parser(
        "glm53-compile",
        help="compile a complete IQ checkpoint from GLM-5.3-BF16 calibration plus official Mamba-3 MIMO weights",
    )
    glm_compile.add_argument("--checkpoint", required=True)
    glm_compile.add_argument("--calibration", required=True)
    glm_compile.add_argument("--mamba3-checkpoint", required=True)
    glm_compile.add_argument("--output", required=True)
    glm_compile.add_argument("--checkpoint-revision", required=True)
    glm_compile.add_argument("--mamba3-revision", required=True)
    glm_compile.add_argument("--donor-license", required=True)
    glm_compile.add_argument(
        "--streaming-source",
        action="store_true",
        help="validate from config/index metadata and require only planned local shards",
    )
    glm_compile.add_argument(
        "--skip-checkpoint-hashes",
        action="store_true",
        help="skip donor hash verification (not recommended)",
    )

    glm_plan_bootstrap = sub.add_parser(
        "glm53-plan-bootstrap-shards",
        help="list the GLM-5.3 safetensors shards required for WARM/bootstrap",
    )
    glm_plan_bootstrap.add_argument("--config", required=True)
    glm_plan_bootstrap.add_argument("--index", required=True)

    glm_plan_compile = sub.add_parser(
        "glm53-plan-compile-shards",
        help="list the GLM-5.3 shards required for calibrated IQ compile",
    )
    glm_plan_compile.add_argument("--config", required=True)
    glm_plan_compile.add_argument("--index", required=True)
    glm_plan_compile.add_argument("--calibration", required=True)

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
    if args.command == "glm53-bootstrap-calibration":
        if args.streaming_source:
            artifact = validate_glm53_streaming_donor(
                checkpoint=args.checkpoint,
                checkpoint_revision=args.checkpoint_revision,
                donor_license=args.donor_license,
                source_uri=args.source_uri,
            )
        else:
            artifact = validate_glm53_donor(
                checkpoint=args.checkpoint,
                checkpoint_revision=args.checkpoint_revision,
                donor_license=args.donor_license,
                source_uri=args.source_uri,
                require_bf16=True,
            )
        config_data = json.loads(
            (artifact.checkpoint_dir / "config.json").read_text(encoding="utf-8")
        )
        inspector = GLM53Inspector.from_config_mapping(config_data)
        source = load_activation_bundle(args.source_activations)
        solution = bootstrap_glm53_calibration(
            source,
            source_weights=SafetensorsSource(artifact.checkpoint_dir),
            source_hidden_size=inspector.config.hidden_size,
            target_config=canonical_glm53_config(),
            source_layers=inspector.config.num_hidden_layers,
            source_first_dense_layers=inspector.layout.first_k_dense_replace,
            source_num_experts=inspector.layout.n_routed_experts,
            source_indexer_types=inspector.layout.indexer_types,
            warm_device=args.warm_device,
        )
        manifest = solution.write(args.output)
        print(
            json.dumps(
                {
                    "calibration_manifest": str(manifest),
                    "donor_fingerprint": artifact.manifest.fingerprint,
                    "target_fingerprint": solution.target_config_fingerprint,
                    "source_layer_map": {
                        str(k): int(v)
                        for k, v in solution.source_layer_map.items()
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "glm53-plan-bootstrap-shards":
        for shard in plan_glm53_bootstrap_shards(
            config_path=args.config,
            index_path=args.index,
        ):
            print(shard)
        return 0
    if args.command == "glm53-plan-compile-shards":
        for shard in plan_glm53_compile_shards(
            config_path=args.config,
            index_path=args.index,
            calibration_dir=args.calibration,
        ):
            print(shard)
        return 0
    if args.command == "glm53-compile":
        result = compile_glm53_iq_checkpoint(
            glm53_checkpoint=args.checkpoint,
            calibration_dir=args.calibration,
            mamba3_checkpoint=args.mamba3_checkpoint,
            output_dir=args.output,
            glm53_revision=args.checkpoint_revision,
            mamba3_revision=args.mamba3_revision,
            donor_license=args.donor_license,
            verify_hashes=not args.skip_checkpoint_hashes,
            streaming_source=args.streaming_source,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "donor_fingerprint": result.donor_fingerprint,
                    "target_fingerprint": result.target_fingerprint,
                    "source_layer_map": {
                        str(k): int(v) for k, v in result.source_layer_map.items()
                    },
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
