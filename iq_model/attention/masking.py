from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedCausalAttention:
    position_ids: torch.Tensor
    valid_tokens: torch.Tensor
    sdpa_mask: torch.Tensor | None
    use_fast_causal: bool


def _validate_token_matrix(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, int],
    *,
    integer: bool = False,
) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if integer and value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be integer typed")


def _validate_document_ids(
    document_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    for row in range(document_ids.shape[0]):
        seen: set[int] = set()
        current: int | None = None
        for col in range(document_ids.shape[1]):
            if not bool(valid_tokens[row, col]):
                continue
            doc = int(document_ids[row, col])
            if current is None:
                current = doc
                seen.add(doc)
            elif doc != current:
                if doc in seen:
                    raise ValueError(
                        "document_ids cannot reappear in non-contiguous segments"
                    )
                seen.add(doc)
                current = doc


def _positions_from_documents(
    document_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    positions = torch.zeros_like(document_ids, dtype=torch.long)
    for row in range(document_ids.shape[0]):
        current: int | None = None
        offset = 0
        for col in range(document_ids.shape[1]):
            if not bool(valid_tokens[row, col]):
                positions[row, col] = 0
                continue
            doc = int(document_ids[row, col])
            if current is None or doc != current:
                current = doc
                offset = 0
            positions[row, col] = offset
            offset += 1
    return positions


def prepare_causal_attention(
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    document_ids: torch.Tensor | None,
) -> PreparedCausalAttention:
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    token_shape = (batch_size, sequence_length)

    if attention_mask is None:
        valid_tokens = torch.ones(token_shape, dtype=torch.bool, device=device)
    else:
        _validate_token_matrix("attention_mask", attention_mask, token_shape)
        valid_tokens = attention_mask.to(device=device, dtype=torch.bool)

    docs: torch.Tensor | None = None
    if document_ids is not None:
        _validate_token_matrix(
            "document_ids",
            document_ids,
            token_shape,
            integer=True,
        )
        docs = document_ids.to(device=device)
        _validate_document_ids(docs, valid_tokens)

    if position_ids is None:
        if docs is None:
            positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(
                batch_size, sequence_length
            )
        else:
            positions = _positions_from_documents(docs, valid_tokens).to(device)
    else:
        _validate_token_matrix(
            "position_ids",
            position_ids,
            token_shape,
            integer=True,
        )
        positions = position_ids.to(device)

    use_fast_causal = docs is None and bool(valid_tokens.all())
    sdpa_mask = None
    if not use_fast_causal:
        causal = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=device,
        ).tril()
        allowed = causal.unsqueeze(0).unsqueeze(0)
        allowed = (
            allowed
            & valid_tokens[:, None, None, :]
            & valid_tokens[:, None, :, None]
        )
        if docs is not None:
            same_document = docs[:, None, :, None] == docs[:, None, None, :]
            allowed = allowed & same_document
        sdpa_mask = allowed

    return PreparedCausalAttention(
        position_ids=positions,
        valid_tokens=valid_tokens,
        sdpa_mask=sdpa_mask,
        use_fast_causal=use_fast_causal,
    )
