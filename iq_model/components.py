from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, groups, seq_len, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * groups, seq_len, head_dim)


class PhiCompatibleGQA(nn.Module):
    """Explicit eager grouped-query attention matching Phi-3/Phi-4-mini weights.

    Parameter names and tensor shapes intentionally match the donor's `qkv_proj` and
    `o_proj`. Keeping the reference mixer explicit prevents backend-specific Hugging
    Face kernels from changing the transfer target underneath the experiment.
    """

    def __init__(self, config: Phi3Config, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.attention_dropout = float(config.attention_dropout)

        q_rows = self.num_attention_heads * self.head_dim
        kv_rows = self.num_key_value_heads * self.head_dim
        self.q_rows = q_rows
        self.kv_rows = kv_rows

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            q_rows + 2 * kv_rows,
            bias=False,
        )
        self.o_proj = nn.Linear(q_rows, self.hidden_size, bias=False)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids  # position information is already represented by cos/sin.

        batch, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., : self.q_rows]
        k = qkv[..., self.q_rows : self.q_rows + self.kv_rows]
        v = qkv[..., self.q_rows + self.kv_rows :]

        q = q.view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)

        mixed = torch.matmul(weights, v)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, self.q_rows)
        return self.o_proj(mixed), weights


class PhiCompatibleSwiGLU(nn.Module):
    """Exact Phi-style gated SiLU FFN with transfer-compatible parameter names."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class IQBlock(nn.Module):
    """Pluggable pre-norm decoder block used by the IQ backbone."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        mixer: nn.Module,
        feed_forward: nn.Module,
        resid_pdrop: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_norm = Phi3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_mixer_norm = Phi3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mixer = mixer
        self.feed_forward = feed_forward
        self.resid_mixer_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        mixed, _ = self.mixer(
            hidden_states=self.input_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = residual + self.resid_mixer_dropout(mixed)

        residual = hidden_states
        hidden_states = self.feed_forward(self.post_mixer_norm(hidden_states))
        return residual + self.resid_ffn_dropout(hidden_states)


def build_phi_compatible_block(config: Phi3Config, *, layer_idx: int) -> IQBlock:
    return IQBlock(
        hidden_size=config.hidden_size,
        rms_norm_eps=config.rms_norm_eps,
        mixer=PhiCompatibleGQA(config=config, layer_idx=layer_idx),
        feed_forward=PhiCompatibleSwiGLU(config.hidden_size, config.intermediate_size),
        resid_pdrop=config.resid_pdrop,
    )
