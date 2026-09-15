from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING

import math
import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config

if TYPE_CHECKING:
    from .config import IQArchitectureConfig


class MixerDependencyError(RuntimeError):
    pass


class UnsupportedMixerComposition(RuntimeError):
    pass


@dataclass(frozen=True)
class MixerContext:
    """Inputs shared across heterogeneous sequence mixers.

    Dense Phi attention consumes the additive causal mask and precomputed RoPE.
    FLA sparse/recurrent layers consume the 2-D padding mask. Mamba-3 owns its
    state-space dynamics and currently requires packed/unpadded sequences in the
    IQ reference path; padded batches are rejected rather than silently corrupted.
    """

    padding_mask: torch.Tensor | None
    causal_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None


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
    """Explicit eager GQA matching the Phi-3/Phi-4-mini parameter layout."""

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

        self.q_rows = self.num_attention_heads * self.head_dim
        self.kv_rows = self.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_rows + 2 * self.kv_rows,
            bias=False,
        )
        self.o_proj = nn.Linear(self.q_rows, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, *, context: MixerContext) -> torch.Tensor:
        if context.causal_mask is None or context.position_embeddings is None:
            raise ValueError("Phi GQA requires causal_mask and position_embeddings")

        batch, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., : self.q_rows]
        k = qkv[..., self.q_rows : self.q_rows + self.kv_rows]
        v = qkv[..., self.q_rows + self.kv_rows :]

        q = q.view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = context.position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        scores = scores + context.causal_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        mixed = torch.matmul(weights, v)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, self.q_rows)
        return self.o_proj(mixed)


def _require_mamba3() -> None:
    if find_spec("mamba_ssm") is None:
        raise MixerDependencyError(
            "Mamba-3 mixers require the pinned state-spaces/mamba dependency; "
            "install requirements-mamba-v2.txt"
        )


class Mamba3MIMOMixer(nn.Module):
    """Adapter around the official state-spaces Mamba-3 MIMO implementation."""

    def __init__(
        self,
        config: Phi3Config,
        architecture: IQArchitectureConfig,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        _require_mamba3()
        from mamba_ssm import Mamba3

        inner_size = int(config.hidden_size * architecture.mamba3_expand)
        if inner_size % architecture.mamba3_head_dim != 0:
            raise ValueError(
                "Mamba-3 expanded width must be divisible by mamba3_head_dim: "
                f"inner={inner_size}, head_dim={architecture.mamba3_head_dim}"
            )

        self.layer = Mamba3(
            d_model=config.hidden_size,
            d_state=architecture.mamba3_state_size,
            expand=architecture.mamba3_expand,
            headdim=architecture.mamba3_head_dim,
            rope_fraction=architecture.mamba3_rope_fraction,
            is_outproj_norm=architecture.mamba3_outproj_norm,
            is_mimo=True,
            mimo_rank=architecture.mamba3_mimo_rank,
            chunk_size=architecture.mamba3_chunk_size,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor, *, context: MixerContext) -> torch.Tensor:
        if context.padding_mask is not None and not bool(torch.all(context.padding_mask != 0)):
            raise ValueError(
                "IQ's Mamba-3 reference path currently requires packed/unpadded batches. "
                "Do not pass zero-padded tokens through Mamba state; pack sequences first."
            )
        return self.layer(hidden_states)


def _require_fla() -> None:
    if find_spec("fla") is None:
        raise MixerDependencyError(
            "hybrid attention controls require flash-linear-attention; install the "
            "pinned requirements-mamba-v2.txt environment"
        )


class FLAGatedDeltaNetMixer(nn.Module):
    """Hybrid-v1 A/B control around FLA's Gated DeltaNet implementation."""

    def __init__(
        self,
        config: Phi3Config,
        architecture: IQArchitectureConfig,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        _require_fla()
        from fla.layers.gated_deltanet import GatedDeltaNet

        key_width = int(round(config.hidden_size * architecture.gdn_key_width_ratio))
        if key_width % architecture.gdn_head_dim != 0:
            raise ValueError(
                "Gated DeltaNet key width must be divisible by gdn_head_dim: "
                f"key_width={key_width}, head_dim={architecture.gdn_head_dim}"
            )
        num_heads = key_width // architecture.gdn_head_dim
        self.layer = GatedDeltaNet(
            hidden_size=config.hidden_size,
            expand_v=architecture.gdn_expand_v,
            head_dim=architecture.gdn_head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=architecture.gdn_use_short_conv,
            conv_size=architecture.gdn_conv_size,
            layer_idx=layer_idx,
            norm_eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor, *, context: MixerContext) -> torch.Tensor:
        output = self.layer(
            hidden_states,
            attention_mask=context.padding_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        return output[0] if isinstance(output, tuple) else output


class FLANativeSparseAttentionMixer(nn.Module):
    """Adapter around FLA Native Sparse Attention using its validated RoPE path."""

    def __init__(
        self,
        config: Phi3Config,
        architecture: IQArchitectureConfig,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        _require_fla()
        from fla.layers.nsa import NativeSparseAttention

        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_theta = float(rope_params.get("rope_theta", 10000.0))
        head_dim = config.hidden_size // config.num_attention_heads
        self.layer = NativeSparseAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=False,
            block_size=architecture.nsa_block_size,
            block_counts=architecture.nsa_block_count,
            window_size=architecture.nsa_window_size,
            rope_theta=rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor, *, context: MixerContext) -> torch.Tensor:
        output = self.layer(
            hidden_states,
            attention_mask=context.padding_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        return output[0] if isinstance(output, tuple) else output


class FLAPaTHAttentionMixer(nn.Module):
    """Standalone PaTH candidate; never silently injected into Mamba or NSA."""

    def __init__(self, config: Phi3Config, *, layer_idx: int) -> None:
        super().__init__()
        _require_fla()
        from fla.layers.path_attn import PaTHAttention

        self.layer = PaTHAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_idx=layer_idx,
            use_low_rank_w=True,
            use_w_shortconv=True,
        )

    def forward(self, hidden_states: torch.Tensor, *, context: MixerContext) -> torch.Tensor:
        output = self.layer(
            hidden_states,
            attention_mask=context.padding_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        return output[0] if isinstance(output, tuple) else output


class LatentNSAMixer(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        raise UnsupportedMixerComposition(
            "latent_nsa remains gated: implement the actual latent-NSA cache/projection "
            "structure before enabling it. Ordinary NSA must not be relabeled."
        )


def build_mixer(
    kind: str,
    phi_config: Phi3Config,
    architecture: IQArchitectureConfig,
    *,
    layer_idx: int,
) -> nn.Module:
    if kind == "phi_gqa":
        return PhiCompatibleGQA(phi_config, layer_idx=layer_idx)
    if kind == "mamba3_mimo":
        return Mamba3MIMOMixer(phi_config, architecture, layer_idx=layer_idx)
    if kind == "gated_deltanet":
        return FLAGatedDeltaNetMixer(phi_config, architecture, layer_idx=layer_idx)
    if kind == "nsa":
        return FLANativeSparseAttentionMixer(phi_config, architecture, layer_idx=layer_idx)
    if kind == "path_attention":
        return FLAPaTHAttentionMixer(phi_config, layer_idx=layer_idx)
    if kind == "latent_nsa":
        return LatentNSAMixer(phi_config, architecture, layer_idx=layer_idx)
    raise ValueError(f"unknown mixer kind: {kind}")
