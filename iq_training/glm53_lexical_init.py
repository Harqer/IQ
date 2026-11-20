"""Sparse, non-destructive IQ-space GLM lexical warm-start candidates.

This is an offline numerical preparation primitive, NOT an inference-ready
embedding/head overlay. The source IQ checkpoint was widened from Mamba-3's
2048-wide embeddings to 4096; its output head was scaled by 1/2. Copy IQ-space
source rows *as-is*, never copy raw GLM 6144-wide neural weights.

This module does not fetch any checkpoints, write shards, allocate a new
154880-row matrix, or claim donor-tokenizer equivalence or trained alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Any

import torch

from iq_cloud.glm53_lexical import SOURCE_IQ_VOCAB, IQ_WIDTH
from iq_cloud.lexical_warmstart import (
    LexicalCandidateError, validate_candidate_manifest,
    LLAMA_COPY_MAPPING_SHA256, LLAMA_COPY_REVISION,
)


@dataclass(frozen=True)
class SparseLexicalRow:
    target_token_id: int
    iq_input_candidate: torch.Tensor
    iq_output_head_candidate: torch.Tensor | None
    method: str
    original_source_token_ids: tuple[int, ...]


def iter_sparse_lexical_candidates(
    original_embedding: torch.Tensor,
    original_lm_head: torch.Tensor,
    manifest: Mapping[str, Any],
    *,
    test_only_smaller_geometry: bool = False,
) -> Iterator[SparseLexicalRow]:
    """Return detached sparse candidate rows without modifying source tensors.

    The production path requires exactly [128256,4096] source IQ-space matrices.
    Small geometry is permitted only for synthetic CPU regression tests.
    This does not create/commit a new trainable model or verify teacher logits.
    """
    validate_candidate_manifest(manifest)
    if (manifest.get("source_tokenizer_mapping_sha256") != LLAMA_COPY_MAPPING_SHA256
            or manifest.get("source_tokenizer_resolved_commit") != LLAMA_COPY_REVISION
            or manifest.get("source_weights_opened") is not False
            or manifest.get("candidate_rows_not_trained_alignment") is not True
            or manifest.get("needs_training_and_held_out_validation") is not True):
        raise LexicalCandidateError("unverified lexical candidate provenance")
    if (not isinstance(original_embedding, torch.Tensor)
            or not isinstance(original_lm_head, torch.Tensor)
            or original_embedding.ndim != 2
            or original_embedding.shape != original_lm_head.shape
            or not original_embedding.is_floating_point()
            or not original_lm_head.is_floating_point()):
        raise LexicalCandidateError("original IQ lexical tensors invalid or missing")
    if not test_only_smaller_geometry and tuple(original_embedding.shape) != (
        SOURCE_IQ_VOCAB, IQ_WIDTH
    ):
        raise LexicalCandidateError("original 128256x4096 IQ lexical shapes required")
    if (original_embedding.device != original_lm_head.device
            or original_embedding.dtype != original_lm_head.dtype):
        raise LexicalCandidateError("original embedding/head device or dtype differs")
    if not bool(torch.isfinite(original_embedding).all().item()
                and torch.isfinite(original_lm_head).all().item()):
        raise LexicalCandidateError("nonfinite original lexical tensor values")

    with torch.no_grad():
        for target_id, source_id in manifest["candidate_matches"]:
            if not 0 <= source_id < original_embedding.shape[0]:
                raise LexicalCandidateError("source candidate row outside provided IQ matrix")
            yield SparseLexicalRow(
                target_token_id=target_id,
                iq_input_candidate=original_embedding[source_id].detach().clone(),
                iq_output_head_candidate=original_lm_head[source_id].detach().clone(),
                method="exact_text_single_source_token_candidate",
                original_source_token_ids=(source_id,),
            )
        for target_id, source_ids in manifest.get("input_only_composition_candidates", []):
            if any(i >= original_embedding.shape[0] for i in source_ids):
                raise LexicalCandidateError("composition source row outside IQ matrix")
            # Mean is only a reproducible training initialization, NOT equivalent
            # to a multi-token recurrent/causal Mamba forward pass.
            values = original_embedding[source_ids].float().mean(dim=0)
            if not bool(torch.isfinite(values).all().item()):
                raise LexicalCandidateError("nonfinite composed candidate row")
            yield SparseLexicalRow(
                target_token_id=target_id,
                iq_input_candidate=values.to(dtype=original_embedding.dtype).clone(),
                iq_output_head_candidate=None,
                method="input_only_multitoken_mean_candidate",
                original_source_token_ids=tuple(source_ids),
            )
