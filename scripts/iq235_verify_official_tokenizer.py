"""Fetch and validate only the official, pinned GLM-5.3 tokenizer assets.

No IQ/GLM weight files, safetensors shards, CUDA, Modal, or bucket writes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from iq_cloud.glm53_lexical import (
    GLM_REPO, GLM_REVISION, GLM_VOCAB, GLM_PINNED_MAPPING_SHA256,
    verify_glm53_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-mapping-sha256",
        default=os.environ.get(
            "IQ_GLM53_TOKENIZER_SHA256", GLM_PINNED_MAPPING_SHA256
        ),
        help="Optional previously verified independent mapping fingerprint",
    )
    parser.add_argument("--offline", action="store_true",
                        help="Use cached tokenizer/config only; prohibit network")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    offline = args.offline or os.getenv("HF_HUB_OFFLINE") == "1"
    config_path = Path(hf_hub_download(
        repo_id=GLM_REPO, revision=GLM_REVISION, filename="config.json",
        local_files_only=offline,
    ))
    source = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        GLM_REPO, revision=GLM_REVISION, local_files_only=offline,
        trust_remote_code=False, use_fast=True,
    )
    result = verify_glm53_tokenizer(
        tokenizer, source,
        expected_mapping_sha256=args.expected_mapping_sha256,
    )
    ids = tokenizer.encode("def square(x): return x * x", add_special_tokens=True)
    if not ids or any(type(token) is not int or not 0 <= token < GLM_VOCAB
                      for token in ids):
        raise RuntimeError("official GLM-5.3 tokenizer produced invalid token IDs")
    result.update({
        "sample_token_ids": ids,
        "chat_template_available": bool(getattr(tokenizer, "chat_template", None)),
        "model_weight_files_downloaded": False,
        "iq_bucket_modified": False,
        "full_pretrained_inference_ready": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
