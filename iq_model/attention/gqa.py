from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import IQModelConfig
from ..position import RotaryEmbedding, apply_rotary


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
    # A document id may occupy only one contiguous segment per row. Reappearing
    # ids would silently reconnect disjoint packed documents.
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


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: IQModelConfig) -> None:
        super().__init__()
        self.config = config
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.dropout = float(config.attention_dropout)

    def _shape_q(self, tensor: torch.Tensor) -> torch.Tensor:
        b, t, _ = tensor.shape
        return tensor.view(
            b, t, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)

    def _shape_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        b, t, _ = tensor.shape
        return tensor.view(
            b, t, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"attention input must have shape [batch, sequence, {self.config.hidden_size}]"
            )
        b, t, _ = x.shape
        token_shape = (b, t)

        if attention_mask is None:
            valid_tokens = torch.ones(
                token_shape,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            _validate_token_matrix("attention_mask", attention_mask, token_shape)
            valid_tokens = attention_mask.to(device=x.device, dtype=torch.bool)

        if document_ids is not None:
            _validate_token_matrix(
                "document_ids",
                document_ids,
                token_shape,
                integer=True,
            )
            document_ids = document_ids.to(device=x.device)
            _validate_document_ids(document_ids, valid_tokens)

        if position_ids is None:
            if document_ids is None:
                position_ids = (
                    torch.arange(t, device=x.device)
                    .unsqueeze(0)
                    .expand(b, t)
                )
            else:
                position_ids = _positions_from_documents(
                    document_ids,
                    valid_tokens,
                ).to(x.device)
        else:
            _validate_token_matrix(
                "position_ids",
                position_ids,
                token_shape,
                integer=True,
            )
            position_ids = position_ids.to(x.device)

        q = self._shape_q(self.q_proj(x))
        k = self._shape_kv(self.k_proj(x))
        v = self._shape_kv(self.v_proj(x))
        cos, sin = self.rotary.cos_sin(
            position_ids,
            dtype=q.dtype,
            device=q.device,
        )
        q, k = apply_rotary(q, k, cos, sin, self.config.rotary_dim)

        if self.config.kv_repeat != 1:
            k = k.repeat_interleave(self.config.kv_repeat, dim=1)
            v = v.repeat_interleave(self.config.kv_repeat, dim=1)

        use_fast_causal = document_ids is None and bool(valid_tokens.all())
        sdpa_mask = None
        if not use_fast_causal:
            causal = torch.ones(
                (t, t),
                dtype=torch.bool,
                device=x.device,
            ).tril()
            allowed = causal.unsqueeze(0).unsqueeze(0)
            allowed = (
                allowed
                & valid_tokens[:, None, None, :]
                & valid_tokens[:, None, :, None]
            )
            if document_ids is not None:
                same_document = (
                    document_ids[:, None, :, None]
                    == document_ids[:, None, None, :]
                )
                allowed = allowed & same_document
            sdpa_mask = allowed

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_fast_causal,
        )
        attn = attn * valid_tokens[:, None, :, None].to(attn.dtype)
        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn)
