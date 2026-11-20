"""Build GLM-5.3/Llama-3.1 candidate lexical map with tokenizer files only.

Produces a JSON recipe, not a weight checkpoint. Never downloads model weights.
The public Llama tokenizer copy must be independently validated against the
original Mamba-3 source before any weight initialization or model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iq_cloud.glm53_lexical import (
    GLM_REPO, GLM_REVISION, GLM_PINNED_MAPPING_SHA256,
    verify_glm53_tokenizer,
)
from iq_cloud.lexical_warmstart import (
    LLAMA_PUBLIC_COPY, LLAMA_COPY_REVISION, make_candidate_manifest,
    validate_candidate_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    source_commit = LLAMA_COPY_REVISION
    source = AutoTokenizer.from_pretrained(
        LLAMA_PUBLIC_COPY, revision=source_commit,
        use_fast=True, trust_remote_code=False,
    )
    target = AutoTokenizer.from_pretrained(
        GLM_REPO, revision=GLM_REVISION,
        use_fast=True, trust_remote_code=False,
    )
    config_path = hf_hub_download(
        GLM_REPO, "config.json", revision=GLM_REVISION,
    )
    glm_config = json.loads(Path(config_path).read_text())
    verify_glm53_tokenizer(
        target, glm_config,
        expected_mapping_sha256=GLM_PINNED_MAPPING_SHA256,
    )
    manifest = make_candidate_manifest(
        source, target, source_revision=source_commit,
        target_revision=GLM_REVISION,
        source_resolved_revision=source_commit,
    )
    counts = validate_candidate_manifest(manifest)
    summary = {
        **counts,
        "candidate_diagnostics": manifest["candidate_diagnostics"],
        "source_commit": source_commit,
        "source_mapping_sha256": manifest["source_tokenizer_mapping_sha256"],
        "target_mapping_sha256": manifest["target_tokenizer_mapping_sha256"],
        "source_official_equivalence_verified": False,
        "pretrained_inference_ready": False,
        "weights_downloaded": False,
    }
    if args.out:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError("refusing to overwrite lexical candidate manifest")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
