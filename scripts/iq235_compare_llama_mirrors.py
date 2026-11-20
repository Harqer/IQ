"""Independent public Llama tokenizer cross-check, tokenizer files only.

Agreement between mirrors strengthens confidence in the candidate mapping.
It does not establish equivalence with the access-controlled official repo
or numerical compatibility with pretrained IQ weights.
"""
from __future__ import annotations

import json

from iq_cloud.lexical_warmstart import (
    LLAMA_COPY_MAPPING_SHA256, LLAMA_COPY_REVISION, LLAMA_PUBLIC_COPY,
    _mapping_hash, validate_source_copy_provenance,
)
from iq_cloud.glm53_lexical import SOURCE_IQ_VOCAB

MIRROR_REPO = "NousResearch/Meta-Llama-3.1-8B"
TEXTS = (
    "Hello, world!",
    "Hello  world!\nA second line.",
    "def square(x: int) -> int:\n    return x * x",
    "🔬 quantum computing and 𠜎",
    "中文、日本語, 한국어; العربية; हिन्दी",
    " \t leading and trailing  ",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
    "Price: $42.95\n",
    "naïve café — résumé",
)

def main() -> None:
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    sha = HfApi().model_info(MIRROR_REPO).sha
    if not isinstance(sha, str) or len(sha) != 40:
        raise RuntimeError("failed to resolve independent tokenizer mirror commit")
    primary = AutoTokenizer.from_pretrained(
        LLAMA_PUBLIC_COPY, revision=LLAMA_COPY_REVISION,
        use_fast=True, trust_remote_code=False,
    )
    independent = AutoTokenizer.from_pretrained(
        MIRROR_REPO, revision=sha, use_fast=True, trust_remote_code=False,
    )
    primary_digest = _mapping_hash(primary, SOURCE_IQ_VOCAB)
    validate_source_copy_provenance(primary_digest, LLAMA_COPY_REVISION)
    other_digest = _mapping_hash(independent, SOURCE_IQ_VOCAB)
    if primary_digest != other_digest:
        raise RuntimeError("independent Llama tokenizer token-ID mapping differs")
    for text in TEXTS:
        for add_special_tokens in (False, True):
            a = primary.encode(text, add_special_tokens=add_special_tokens)
            b = independent.encode(text, add_special_tokens=add_special_tokens)
            if a != b:
                raise RuntimeError("independent Llama tokenizer encoding disagrees")
    primary_special = set(primary.all_special_ids)
    independent_special = set(independent.all_special_ids)
    if primary_special != independent_special:
        only_primary = sorted(primary_special - independent_special)
        only_independent = sorted(independent_special - primary_special)
        raise RuntimeError(
            "independent Llama special classifications disagree: "
            f"only_primary={only_primary[:30]}, "
            f"only_mirror={only_independent[:30]}, "
            f"counts=({len(primary_special)}, {len(independent_special)})"
        )
    print(json.dumps({
        "candidate_source_tokenizer": LLAMA_PUBLIC_COPY,
        "candidate_source_commit": LLAMA_COPY_REVISION,
        "independent_mirror_repo": MIRROR_REPO,
        "independent_mirror_commit": sha,
        "mapping_sha256": primary_digest,
        "mapping_agreement": True,
        "sampled_encode_agreement": True,
        "sampled_text_count": len(TEXTS),
        "official_meta_llama_equivalence_verified": False,
        "iq_source_weight_alignment_verified": False,
        "checkpoint_weights_downloaded": False,
    }, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()
