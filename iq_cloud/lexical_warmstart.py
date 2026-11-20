"""Weight-free, conservative lexical correspondences for GLM-5.3 -> Mamba IQ235.

Candidate row matches are *not* trained lexical alignment, even when identical
text roundtrips across tokenizers. No neural weights are loaded or modified.
The source tokenizer is a public third-party copy until independently verified
against the original Mamba-3's official Llama-3.1-8B tokenizer.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

from .glm53_lexical import (
    GLM_PINNED_MAPPING_SHA256, GLM_VOCAB, SOURCE_IQ_VOCAB,
    mapping_sha256,
)

LLAMA_PUBLIC_COPY = "unsloth/Meta-Llama-3.1-8B"
LLAMA_OFFICIAL = "meta-llama/Llama-3.1-8B"
LLAMA_COPY_REVISION = "e9a141a2091ea561b96483212645a2a05e6f99fc"
LLAMA_COPY_MAPPING_SHA256 = "f66003c7a898537275cd9d3ab290d6e247b9c5063352c2cc010213febfbe068f"
MAX_TOKEN_CHARS = 512


class LexicalCandidateError(ValueError):
    """Lexical inputs or a candidate manifest are incompatible."""


def _mapping_hash(tokenizer: Any, limit: int) -> str:
    vocabulary = tokenizer.get_vocab()
    if (not isinstance(vocabulary, dict) or not vocabulary
            or any(type(k) is not str or type(v) is not int
                   or v < 0 or v >= limit for k, v in vocabulary.items())
            or len(set(vocabulary.values())) != len(vocabulary)):
        raise LexicalCandidateError("tokenizer mapping has invalid IDs/entries")
    return sha256(json.dumps(
        vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def validate_source_copy_provenance(digest: str, commit: str) -> None:
    """A stable public tokenizer copy is not proof of official equivalence."""
    if commit != LLAMA_COPY_REVISION or digest != LLAMA_COPY_MAPPING_SHA256:
        raise LexicalCandidateError("source tokenizer copy revision/mapping has drifted")


def _one_id_roundtrips(tokenizer: Any, token_id: int) -> str | None:
    """Only byte/text-exact, printable single-token encodings qualify."""
    if token_id in set(tokenizer.all_special_ids):
        return None
    text = tokenizer.decode(
        [token_id], skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if (not isinstance(text, str) or not 0 < len(text) <= MAX_TOKEN_CHARS
            or "\ufffd" in text
            or any(not (c.isprintable() or c in "\t\n") for c in text)):
        return None
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded != [token_id]:
        return None
    return text


def candidate_rows(
    source_tokenizer: Any,
    target_tokenizer: Any,
    *,
    target_ids: Iterable[int] | None = None,
    diagnostics: dict[str, int] | None = None,
    source_limit: int = SOURCE_IQ_VOCAB,
    target_limit: int = GLM_VOCAB,
) -> dict[int, int]:
    """Return target-row -> source-row matches, ignoring ambiguous tokens.

    This is a conservative warm-start proposal, not a substitute for training.
    Both directions must be exactly a *single* token in their own tokenizer.
    """
    if type(source_limit) is not int or source_limit < 1:
        raise LexicalCandidateError("invalid source vocabulary capacity")
    if type(target_limit) is not int or target_limit < 1:
        raise LexicalCandidateError("invalid GLM vocabulary capacity")
    source_vocabulary = source_tokenizer.get_vocab()
    target_vocabulary = target_tokenizer.get_vocab()
    if not source_vocabulary or not target_vocabulary:
        raise LexicalCandidateError("empty source or target token mapping")
    if target_ids is None:
        ids = sorted(target_vocabulary.values())
    else:
        ids = sorted(target_ids)
    defined_target_ids = set(target_vocabulary.values())
    if len(ids) != len(set(ids)) or any(
        type(i) is not int or i < 0 or i >= target_limit
        or i not in defined_target_ids for i in ids
    ):
        raise LexicalCandidateError("invalid target token IDs")
    source_special = set(source_tokenizer.all_special_ids)
    target_special = set(target_tokenizer.all_special_ids)
    matches: dict[int, int] = {}
    reasons: Counter[str] = Counter()
    for target_id in ids:
        if target_id in target_special:
            reasons["target_special"] += 1
            continue
        text = _one_id_roundtrips(target_tokenizer, target_id)
        if text is None:
            reasons["target_non_roundtrip"] += 1
            continue
        possible = source_tokenizer.encode(text, add_special_tokens=False)
        if (len(possible) != 1 or type(possible[0]) is not int
                or not 0 <= possible[0] < source_limit
                or possible[0] in source_special):
            reasons["source_not_single_defined_token"] += 1
            continue
        source_id = possible[0]
        if _one_id_roundtrips(source_tokenizer, source_id) != text:
            reasons["source_non_roundtrip"] += 1
            continue
        reasons["candidate_matched"] += 1
        matches[target_id] = source_id
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({name: reasons[name] for name in (
            "candidate_matched", "target_special", "target_non_roundtrip",
            "source_not_single_defined_token", "source_non_roundtrip",
        )})
        if sum(diagnostics.values()) != len(ids):
            raise LexicalCandidateError("candidate diagnostic counts incomplete")
    return matches


def make_candidate_manifest(
    source_tokenizer: Any, target_tokenizer: Any,
    *, source_revision: str, target_revision: str,
    source_resolved_revision: str,
) -> dict[str, Any]:
    """Construct deterministic, NO-WEIGHTS candidate mapping + provenance."""
    if not (source_revision and target_revision and source_resolved_revision):
        raise LexicalCandidateError("tokenizer commit provenance must be nonempty")
    target_fingerprint = mapping_sha256(target_tokenizer.get_vocab())
    if target_fingerprint != GLM_PINNED_MAPPING_SHA256:
        raise LexicalCandidateError("unexpected GLM-5.3 mapping fingerprint")
    if len(target_tokenizer) > GLM_VOCAB or len(source_tokenizer) > SOURCE_IQ_VOCAB:
        raise LexicalCandidateError("tokenizer exceeds recipient's vocabulary")
    source_hash = _mapping_hash(source_tokenizer, SOURCE_IQ_VOCAB)
    validate_source_copy_provenance(source_hash, source_resolved_revision)
    if source_revision != LLAMA_COPY_REVISION:
        raise LexicalCandidateError("source revision must use pinned public copy")
    diagnostics: dict[str, int] = {}
    correspondences = candidate_rows(
        source_tokenizer, target_tokenizer, diagnostics=diagnostics,
    )
    defined_target_count = len(target_tokenizer.get_vocab())
    return {
        "schema_version": 1,
        "stage": "lexical_candidate_mapping_only",
        "source_model": "state-spaces/mamba3-mimo-1.5b",
        "source_tokenizer_official_repo": LLAMA_OFFICIAL,
        "source_tokenizer_copy_repo": LLAMA_PUBLIC_COPY,
        "source_tokenizer_requested_revision": source_revision,
        "source_tokenizer_resolved_commit": source_resolved_revision,
        "source_tokenizer_mapping_sha256": source_hash,
        "source_tokenizer_official_equivalence_proven": False,
        "target_tokenizer_repo": "zai-org/GLM-5.3-BF16",
        "target_tokenizer_revision": target_revision,
        "target_tokenizer_mapping_sha256": target_fingerprint,
        "target_model_vocab_capacity": GLM_VOCAB,
        "target_defined_token_count": defined_target_count,
        "unmapped_target_rows_including_reserved": GLM_VOCAB - len(correspondences),
        "candidate_matched_defined_rows": len(correspondences),
        "candidate_diagnostics": diagnostics,
        "candidate_matches": [[t, s] for t, s in sorted(correspondences.items())],
        "candidate_rows_not_trained_alignment": True,
        "full_pretrained_inference_ready": False,
        "source_weights_opened": False,
        "weights_initialized_or_modified": False,
        "needs_training_and_held_out_validation": True,
    }


def validate_candidate_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Fail closed if correspondences or reported counts are inconsistent."""
    if (manifest.get("schema_version") != 1
            or manifest.get("stage") != "lexical_candidate_mapping_only"
            or manifest.get("source_tokenizer_official_equivalence_proven") is not False
            or manifest.get("target_tokenizer_mapping_sha256") != GLM_PINNED_MAPPING_SHA256
            or manifest.get("target_model_vocab_capacity") != GLM_VOCAB
            or manifest.get("full_pretrained_inference_ready") is not False
            or manifest.get("weights_initialized_or_modified") is not False):
        raise LexicalCandidateError("manifest makes unverified provenance/readiness claim")
    rows = manifest.get("candidate_matches")
    if not isinstance(rows, list):
        raise LexicalCandidateError("candidate matches must be a list")
    prev = -1
    for row in rows:
        if (not isinstance(row, list) or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not int
                or not prev < row[0] < GLM_VOCAB
                or not 0 <= row[1] < SOURCE_IQ_VOCAB):
            raise LexicalCandidateError("candidate IDs unsorted, duplicate or out of bounds")
        prev = row[0]
    if (manifest.get("candidate_matched_defined_rows") != len(rows)
            or manifest.get("unmapped_target_rows_including_reserved") != GLM_VOCAB-len(rows)
            or type(manifest.get("target_defined_token_count")) is not int
            or not len(rows) <= manifest["target_defined_token_count"] <= GLM_VOCAB):
        raise LexicalCandidateError("candidate row coverage counts inconsistent")
    diagnosis = manifest.get("candidate_diagnostics")
    if diagnosis is not None:
        expected_names = {"candidate_matched", "target_special",
                          "target_non_roundtrip",
                          "source_not_single_defined_token", "source_non_roundtrip"}
        if (not isinstance(diagnosis, dict)
                or set(diagnosis) != expected_names
                or any(type(v) is not int or v < 0 for v in diagnosis.values())
                or diagnosis["candidate_matched"] != len(rows)
                or sum(diagnosis.values()) != manifest["target_defined_token_count"]):
            raise LexicalCandidateError("candidate diagnostics inconsistent")
    return {
        "candidate_matched_rows": len(rows),
        "unmatched_or_reserved_rows": GLM_VOCAB - len(rows),
    }
