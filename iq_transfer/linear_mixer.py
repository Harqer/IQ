from __future__ import annotations

"""Reference IQ linear-time matrix mixer for MOHAWK experiments.

This is deliberately a research reference, not an optimized kernel.  During Stage 1
we materialize the T x T mixing matrix so it can be compared directly with the
teacher attention matrix, exactly as MOHAWK does.  A production implementation can
replace that path with a recurrent/prefix-scan formulation after transfer quality is
established.
"""

import torch
from torch import nn
import torch.nn.functional as F


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, h, t, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, h, groups, t, d)
        .reshape(b, h * groups, t, d)
    )


class IQLinearAttentionMixer(nn.Module):
    """Causal positive-feature linear attention with Phi-compatible GQA shapes.

    The feature dimension equals the Phi head dimension for the first experiment.
    This lets us copy Phi-4-mini Q/K/V/O weights exactly and isolate the architectural
    change to the sequence-mixing rule: softmax attention -> normalized positive
    kernel attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim or hidden_size // num_attention_heads)
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("first MOHAWK experiment requires heads * head_dim == hidden_size")
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.eps = float(eps)

        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, self.hidden_size, bias=False)

    @torch.no_grad()
    def initialize_from_phi4(
        self,
        fused_qkv_weight: torch.Tensor,
        output_weight: torch.Tensor,
    ) -> None:
        """Copy the directly compatible Phi-4-mini attention projections.

        Phi-4-mini stores Q/K/V in one fused projection.  Copying these matrices is
        information preserving; MOHAWK then only has to learn the new mixer rule.
        """
        q_rows = self.num_attention_heads * self.head_dim
        kv_rows = self.num_key_value_heads * self.head_dim
        expected = (q_rows + 2 * kv_rows, self.hidden_size)
        if tuple(fused_qkv_weight.shape) != expected:
            raise ValueError(
                f"unexpected fused QKV shape: got={tuple(fused_qkv_weight.shape)} expected={expected}"
            )
        if tuple(output_weight.shape) != (self.hidden_size, q_rows):
            raise ValueError(
                f"unexpected output projection shape: got={tuple(output_weight.shape)} "
                f"expected={(self.hidden_size, q_rows)}"
            )

        self.q_proj.weight.copy_(fused_qkv_weight[:q_rows])
        self.k_proj.weight.copy_(fused_qkv_weight[q_rows : q_rows + kv_rows])
        self.v_proj.weight.copy_(fused_qkv_weight[q_rows + kv_rows : q_rows + 2 * kv_rows])
        self.o_proj.weight.copy_(output_weight)

    def _project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            # Keep the dependency local so importing iq_transfer does not require
            # transformers unless this experimental mixer is actually used.
            from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        return q, k, v

    def mixing_matrix(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Materialize the Stage-1 mixer matrix [B, H, T, T]."""
        q, k, _ = self._project(hidden_states, position_embeddings)

        # Positive feature map used by kernelized linear attention.  Compute the
        # explicit matrix only for MOHAWK orientation/evaluation.
        qf = F.elu(q.float()) + 1.0
        kf = F.elu(k.float()) + 1.0
        scores = torch.matmul(qf, kf.transpose(-1, -2))

        t = scores.shape[-1]
        causal = torch.ones((t, t), device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, 0.0)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return (scores / denom).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        matrix = self.mixing_matrix(hidden_states, position_embeddings=position_embeddings)
        _, _, v = self._project(hidden_states, position_embeddings)
        mixed = torch.matmul(matrix, v)
        mixed = mixed.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], hidden_states.shape[1], self.hidden_size
        )
        return self.o_proj(mixed)
