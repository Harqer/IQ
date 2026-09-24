from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import IQModelConfig
from ..norm import HeadRMSNorm
from ..position import RotaryEmbedding, apply_rotary
from .masking import prepare_causal_attention


class DenseContextAttention(nn.Module):
    """Dense Transformer context anchor for the hybrid IQ backbone.

    This is distinct from GroupedQueryAttention, which is retained as the
    Phi-transfer control/teacher module. Hybrid dense anchors add per-head Q/K
    RMS normalization before RoPE.
    """

    def __init__(self, config: IQModelConfig) -> None:
        super().__init__()
        self.config = config
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
        self.q_norm = HeadRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = HeadRMSNorm(config.head_dim, config.rms_norm_eps)
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
                f"context attention input must have shape [batch, sequence, {self.config.hidden_size}]"
            )
        b, t, _ = x.shape
        prepared = prepare_causal_attention(
            batch_size=b,
            sequence_length=t,
            device=x.device,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )

        q = self.q_norm(self._shape_q(self.q_proj(x)))
        k = self.k_norm(self._shape_kv(self.k_proj(x)))
        v = self._shape_kv(self.v_proj(x))
        cos, sin = self.rotary.cos_sin(
            prepared.position_ids,
            dtype=q.dtype,
            device=q.device,
        )
        q, k = apply_rotary(q, k, cos, sin, self.config.rotary_dim)

        if self.config.kv_repeat != 1:
            k = k.repeat_interleave(self.config.kv_repeat, dim=1)
            v = v.repeat_interleave(self.config.kv_repeat, dim=1)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=prepared.sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=prepared.use_fast_causal,
        )
        attn = attn * prepared.valid_tokens[:, None, :, None].to(attn.dtype)
        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn)
