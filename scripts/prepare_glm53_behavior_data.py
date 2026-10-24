#!/usr/bin/env python3
"""Prepare real GLM-5.3 text responses for IQ using IQ tokenizer IDs only.

Input JSONL: prompt, response, teacher_model, teacher_revision, request_id.
This verifies schema and file integrity, NOT truthfulness of teacher attribution.
Never uses GLM token IDs as IQ training labels.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from safetensors.torch import save_file
from transformers import AutoTokenizer

from iq_training.behavior_transfer import (
    BehaviorTransferError,
    build_iq_example,
    validate_teacher_examples,
)


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not (0 < args.validation_fraction < 0.5):
        parser.error("validation-fraction must be between zero and 0.5")
    if not 4 <= args.max_tokens <= 2048:
        parser.error("max-tokens must be between 4 and 2048")
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite data")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (args.tokenizer_dir / name).is_file():
            parser.error(f"missing tokenizer artifact {name!r}")

    try:
        with args.input.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        examples = validate_teacher_examples(records)
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_dir), local_files_only=True
        )
        if len(tokenizer) != 128256:
            raise BehaviorTransferError("IQ tokenizer must have exactly 128256 IDs")
        encoded = [
            (item.content_digest(), build_iq_example(
                tokenizer, item, max_tokens=args.max_tokens
            ))
            for item in examples
        ]
        ordered = sorted(encoded, key=lambda item: item[0])
        count_validation = max(1, round(len(ordered) * args.validation_fraction))
        if len(ordered) - count_validation < 1:
            raise BehaviorTransferError("need distinct train/validation examples")
        splits = {
            "validation": ordered[:count_validation],
            "train": ordered[count_validation:],
        }
        args.output_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "purpose": "iq_response_supervision",
            "teacher_source_claim": examples[0].teacher_model,
            "teacher_revision_claim": examples[0].teacher_revision,
            "teacher_attribution_verified_by_api": False,
            "teacher_input_sha256": checksum(args.input),
            "tokenizer_json_sha256": checksum(args.tokenizer_dir / "tokenizer.json"),
            "tokenizer_config_sha256": checksum(args.tokenizer_dir / "tokenizer_config.json"),
            "max_tokens": args.max_tokens,
            "split": {},
            "tokenization": "IQ Llama3.1 vocabulary; question tokens masked; response+EOS supervised",
        }
        for split_name, rows in splits.items():
            tensors = {}
            for i, (_, example) in enumerate(rows):
                for key, value in example.items():
                    tensors[f"{i:06d}/{key}"] = value.contiguous()
            file = args.output_dir / f"{split_name}.safetensors"
            save_file(tensors, str(file))
            manifest["split"][split_name] = {
                "count": len(rows),
                "content_digests": [digest for digest, _ in rows],
                "sha256": checksum(file),
            }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Prepared {len(ordered)} claimed GLM teacher records. "
              "No IQ model weights changed; provenance is not API-authenticated.")
        print(f"train={len(splits['train'])} validation={len(splits['validation'])}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
