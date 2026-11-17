#!/usr/bin/env python3
"""Validate GLM-5.3 tokenization without model weights or required local paths.

By default, Hugging Face supplies *only tokenizer artifacts and config.json*
from the pinned official GLM release. --offline enforces cached/local files.
Specify --iq-config only when checking or adapting an existing IQ hybrid JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from iq_model.glm_tokenizer import (
    GLM53_TOKENIZER_REPO,
    GLM53_TOKENIZER_REVISION,
    GLMTokenizerError,
    encode_glm_prompt,
    verify_glm_tokenizer_for_iq,
    with_glm_tokenizer_config,
)
from iq_model.hybrid import IQHybridConfig


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="The answer is",
                   help="Text to tokenize and check (no model forward pass)")
    p.add_argument("--offline", action="store_true",
                   help="Never contact Hugging Face; use cached or local tokenizer artifacts")
    p.add_argument("--tokenizer-dir", type=Path,
                   help="Existing local GLM tokenizer directory; no downloads or Hub requests")
    p.add_argument("--iq-config", type=Path,
                   help="Optional existing IQHybridConfig JSON to check/adapt")
    p.add_argument("--write-config", type=Path,
                   help="Write a new GLM-vocabulary IQ config; requires --iq-config")
    p.add_argument("--expected-mapping-sha256",
                   help="Optional independently pinned full tokenizer mapping fingerprint")
    return p


def load_tokenizer_artifacts(
    *,
    offline: bool,
    tokenizer_dir: Path | None,
):
    """Request only tokenizer files and GLM metadata—never model weight shards."""
    from transformers import AutoTokenizer

    if tokenizer_dir is not None:
        config_path = tokenizer_dir / "config.json"
        if not config_path.is_file():
            raise GLMTokenizerError(
                f"local GLM tokenizer directory needs config.json: {tokenizer_dir}"
            )
        source = str(tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=False, local_files_only=True, use_fast=True,
        )
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(
            repo_id=GLM53_TOKENIZER_REPO,
            filename="config.json",
            revision=GLM53_TOKENIZER_REVISION,
            local_files_only=offline,
        ))
        tokenizer = AutoTokenizer.from_pretrained(
            GLM53_TOKENIZER_REPO,
            revision=GLM53_TOKENIZER_REVISION,
            trust_remote_code=False,
            local_files_only=offline,
            use_fast=True,
        )
    return json.loads(config_path.read_text(encoding="utf-8")), tokenizer


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.write_config is not None and args.iq_config is None:
        p.error("--write-config requires --iq-config (a real existing IQ hybrid JSON)")
    if args.write_config is not None and args.write_config.exists():
        raise GLMTokenizerError(f"output IQ config already exists: {args.write_config}")
    config = (
        IQHybridConfig.from_json(str(args.iq_config))
        if args.iq_config is not None else None
    )
    if config is not None and args.write_config is not None:
        config = with_glm_tokenizer_config(config)

    source_config, tokenizer = load_tokenizer_artifacts(
        offline=args.offline, tokenizer_dir=args.tokenizer_dir,
    )
    model_config = config.model if config is not None else None
    result = verify_glm_tokenizer_for_iq(
        tokenizer, source_config, model_config,
        expected_mapping_sha256=args.expected_mapping_sha256,
    )
    result["prompt_token_ids"] = encode_glm_prompt(
        tokenizer, args.prompt, model_config,
    )
    result["source_mode"] = (
        "local" if args.tokenizer_dir is not None
        else "offline_cache" if args.offline else "huggingface_tokenizer_only"
    )
    result["glm_neural_weights_loaded"] = False
    result["iq_architecture_modified"] = False
    if args.write_config is not None:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        config.write_json(str(args.write_config))
        result["new_iq_config"] = str(args.write_config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GLMTokenizerError, OSError, ImportError, ValueError, RuntimeError) as exc:
        print(f"IQ_GLM_TOKENIZER_CONTRACT_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
