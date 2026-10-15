#!/usr/bin/env python3
"""Train one real IQ CSA attention branch from paired official GLM-5.3 captures.

The teacher reference is produced by scripts.evaluate_glm53_attention_teacher
using authentic BF16 GLM attention weights and its actual source activations.
IQ input captures must correspond to exactly the same token sequences.
Only a single compressed-attention module is trained/saved, NOT a complete IQ
checkpoint or a claimed full pretrained weight conversion.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path

import torch

from iq_model import CompressedSparseContextAttention
from iq_transfer.complete_transplant import nvidia_parallel_78_config
from iq_transfer.glm53_attention_alignment import (
    PairedAttentionBatch, fit_attention_from_paired_activations,
)
from iq_transfer.transport import load_coordinate_map
from iq_transfer.glm53_token_alignment import align_teacher_by_char_spans


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_paired(
    *, teacher_dir: Path, iq_capture: Path, layer: int,
    allow_shared_token_ids: bool = False,
) -> tuple[PairedAttentionBatch, dict]:
    from safetensors.torch import load_file
    manifest_path = teacher_dir / "manifest.json"
    source = teacher_dir / "reference.safetensors"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        meta.get("format") != "iq_source_native_glm53_attention_reference_v1"
        or meta.get("layer") != layer
        or meta.get("donor_operator") != "transformers==5.15.0 GlmMoeDsaAttention"
        or meta.get("output_sha256") != checksum(source)
    ):
        raise ValueError("teacher attention source provenance/hash does not match")
    raw_teacher = load_file(str(source), device="cpu")
    raw_iq = load_file(str(iq_capture), device="cpu")
    required_teacher = {"source_attention_output", "dsa_indices", "token_ids"}
    required_iq = {"iq_attention_input", "token_ids"}
    span_extras = {"token_offsets", "text_sha256"}
    if set(raw_teacher) not in (required_teacher, required_teacher | span_extras):
        raise ValueError("teacher capture has missing/extra source attention fields")
    if set(raw_iq) not in (required_iq, required_iq | span_extras):
        raise ValueError("IQ capture has missing/extra student attention fields")
    if span_extras.issubset(raw_teacher) and span_extras.issubset(raw_iq):
        aligned = align_teacher_by_char_spans(
            raw_teacher["source_attention_output"],
            raw_teacher["token_offsets"],
            raw_iq["token_offsets"],
            source_text_hashes=raw_teacher["text_sha256"],
            target_text_hashes=raw_iq["text_sha256"],
        )
        data = PairedAttentionBatch(
            iq_attention_input=raw_iq["iq_attention_input"],
            glm_attention_output=aligned.mapped_teacher_output,
            token_ids=raw_iq["token_ids"],
            dsa_indices=None,
            supervision_mask=aligned.supervision_mask,
            sample_text_hashes=raw_iq["text_sha256"],
            alignment_mode="causal_character_spans",
        )
        alignment_metadata = {
            "mode": aligned.method,
            "causal_supervision_coverage": aligned.coverage_ratio,
            "DSA_index_recall_comparable": False,
        }
    else:
        if not allow_shared_token_ids:
            raise ValueError(
                "GLM and Nemotron tokenizers are different: supply token_offsets "
                "and matching text_sha256 from both fast tokenizers, or explicitly "
                "opt in to --shared-tokenizer for a verified common tokenizer"
            )
        if not torch.equal(raw_teacher["token_ids"], raw_iq["token_ids"]):
            raise ValueError("GLM and IQ activations are not from identical token sequences")
        data = PairedAttentionBatch(
            iq_attention_input=raw_iq["iq_attention_input"],
            glm_attention_output=raw_teacher["source_attention_output"],
            token_ids=raw_teacher["token_ids"],
            dsa_indices=raw_teacher["dsa_indices"],
        )
        alignment_metadata = {
            "mode": "user_verified_shared_tokenizer",
            "causal_supervision_coverage": 1.0,
            "DSA_index_recall_comparable": True,
        }
    return data, {
        "token_alignment": alignment_metadata,
        "teacher_manifest_sha256": checksum(manifest_path),
        "teacher_reference_sha256": checksum(source),
        "iq_capture_sha256": checksum(iq_capture),
        "donor_config_sha256": meta["donor_config_sha256"],
        "index_source_block": meta["index_source_block"],
        "indexer_kind": meta["indexer_kind"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--train-teacher-dir", type=Path, required=True)
    parser.add_argument("--train-iq-capture", type=Path, required=True)
    parser.add_argument("--validation-teacher-dir", type=Path, required=True)
    parser.add_argument("--validation-iq-capture", type=Path, required=True)
    parser.add_argument("--coordinate-map", type=Path, required=True,
                        help="prefix of real fitted CoordinateMap .json/.safetensors")
    parser.add_argument("--initial-student", type=Path,
                        help="existing IQ CSA safetensors state dict")
    parser.add_argument("--shared-tokenizer", action="store_true",
                        help="explicitly assert the two captures use one identical tokenizer")
    parser.add_argument("--min-span-coverage", type=float, default=0.5,
                        help="minimum valid fraction of causally completed source spans")
    parser.add_argument("--start-random", action="store_true",
                        help="explicitly initialize a new CSA branch; does NOT transfer weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.layer < 78:
        parser.error("layer must be in [0,77]")
    if (args.initial_student is None) == (not args.start_random):
        parser.error("choose exactly one: --initial-student or --start-random")
    if args.output.exists():
        parser.error("refuse to overwrite existing attention checkpoint directory")

    train, train_provenance = load_paired(
        teacher_dir=args.train_teacher_dir,
        iq_capture=args.train_iq_capture, layer=args.layer,
        allow_shared_token_ids=args.shared_tokenizer,
    )
    validation, validation_provenance = load_paired(
        teacher_dir=args.validation_teacher_dir,
        iq_capture=args.validation_iq_capture, layer=args.layer,
        allow_shared_token_ids=args.shared_tokenizer,
    )
    if (
        train_provenance["donor_config_sha256"]
        != validation_provenance["donor_config_sha256"]
        or train_provenance["indexer_kind"] != validation_provenance["indexer_kind"]
        or train_provenance["index_source_block"] != validation_provenance["index_source_block"]
    ):
        parser.error("train/validation donor checkpoint or indexer provenance differs")
    if not 0 < args.min_span_coverage <= 1:
        parser.error("--min-span-coverage must be in (0,1]")
    for provenance in (train_provenance, validation_provenance):
        if provenance["token_alignment"]["causal_supervision_coverage"] < args.min_span_coverage:
            parser.error(
                "cross-tokenizer causal span coverage below required threshold"
            )
    if train.alignment_mode != validation.alignment_mode:
        parser.error("train and validation cannot mix tokenizer alignment modes")
    config = nvidia_parallel_78_config()
    assert config.compressed_context is not None
    torch.manual_seed(args.seed)
    student = CompressedSparseContextAttention(
        config.compressed_context
    ).to(args.device).float()
    if args.initial_student is not None:
        from safetensors.torch import load_file
        initial = load_file(str(args.initial_student), device="cpu")
        student.load_state_dict(initial, strict=True)
    map_data = load_coordinate_map(args.coordinate_map)
    result = fit_attention_from_paired_activations(
        student, train, validation, map_data,
        steps=args.steps, learning_rate=args.lr,
    )
    from safetensors.torch import save_file
    args.output.mkdir(parents=True, exist_ok=False)
    state = args.output / "student-csa.safetensors"
    save_file({
        k: v.detach().cpu().contiguous()
        for k, v in sorted(student.state_dict().items())
    }, str(state))
    config_string = json.dumps(asdict(config.compressed_context), sort_keys=True)
    result.update({
        "format": "iq_glm53_stagewise_csa_distillation_v1",
        "source_stage": args.layer,
        "student_architecture": "CompressedSparseContextAttention",
        "student_config_sha256": sha256(config_string.encode()).hexdigest(),
        "student_config": asdict(config.compressed_context),
        "student_initialization": (
            {"path": str(args.initial_student), "sha256": checksum(args.initial_student)}
            if args.initial_student is not None else {"seed": args.seed, "kind": "random"}
        ),
        "coordinate_map_sha256": checksum(args.coordinate_map.with_suffix(".safetensors")),
        "coordinate_map_metadata_sha256": checksum(args.coordinate_map.with_suffix(".json")),
        "train_provenance": train_provenance,
        "validation_provenance": validation_provenance,
        "state_sha256": checksum(state),
        "state": "trained_CSA_stage_requires_causal_and_quality_acceptance",
        "full_pretrained_IQ_checkpoint": False,
    })
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": args.layer, "holdout_improved": result["holdout_improved"],
        "initial": result["initial_holdout"],
        "final": result["best_holdout"],
        "output": str(args.output),
        "full_pretrained_IQ_checkpoint": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
