#!/usr/bin/env python3
"""Bind the official GLM-5.3 tokenizer to the EXISTING IQ hybrid configuration.

Loads tokenizer/config metadata only. Does not modify an IQ network, weights,
Mamba recurrence, or any inference engine.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq-config", type=Path, required=True,
                        help="Existing IQHybridConfig JSON (the canonical hybrid schedule)")
    parser.add_argument("--write-config", type=Path,
                        help="Write a separate GLM-vocabulary IQ config without overwriting files")
    parser.add_argument("--expected-mapping-sha256",
                        help="Optional independently recorded full GLM token-to-ID SHA-256")
    parser.add_argument("--prompt", default="The answer is",
                        help="Example text to tokenize and validate against IQ vocabulary")
    args = parser.parse_args()

    config = IQHybridConfig.from_json(str(args.iq_config))
    if args.write_config is not None:
        if args.write_config.exists():
            raise GLMTokenizerError(f"output IQ config already exists: {args.write_config}")
        config = with_glm_tokenizer_config(config)

    # Download only tokenizer metadata and tokenizer files at the pinned revision.
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    config_path = hf_hub_download(
        repo_id=GLM53_TOKENIZER_REPO, filename="config.json",
        revision=GLM53_TOKENIZER_REVISION,
    )
    source_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        GLM53_TOKENIZER_REPO,
        revision=GLM53_TOKENIZER_REVISION,
        trust_remote_code=False,
        use_fast=True,
    )
    result = verify_glm_tokenizer_for_iq(
        tokenizer, source_config, config.model,
        expected_mapping_sha256=args.expected_mapping_sha256,
    )
    input_ids = encode_glm_prompt(tokenizer, args.prompt, config.model)
    result["prompt_token_ids"] = input_ids
    result["iq_full_schedule_unchanged"] = True
    result["glm_neural_weights_loaded"] = False
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
