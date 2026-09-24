from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm


class CompressedAttentionError(ValueError):
    pass


class CompressedAttentionType(str, Enum):
    CSA = "csa"
    HCA = "hca"


@dataclass(frozen=True)
class CompressedAttentionConfig:
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    rope_head_dim: int
    sliding_window: int
    o_groups: int
    o_lora_rank: int
    csa_compress_rate: int = 4
    hca_compress_rate: int = 128
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 2048
    rope_theta: float = 10000.0
    compress_rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        ints = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "rope_head_dim": self.rope_head_dim,
            "sliding_window": self.sliding_window,
            "o_groups": self.o_groups,
            "o_lora_rank": self.o_lora_rank,
            "csa_compress_rate": self.csa_compress_rate,
            "hca_compress_rate": self.hca_compress_rate,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise CompressedAttentionError(
                f"positive compressed-attention fields required: {', '.join(bad)}"
            )
        if self.rope_head_dim % 2:
            raise CompressedAttentionError("rope_head_dim must be even")
        if self.rope_head_dim > self.head_dim:
            raise CompressedAttentionError(
                "rope_head_dim cannot exceed head_dim"
            )
        if self.num_attention_heads % self.o_groups:
            raise CompressedAttentionError(
                "num_attention_heads must be divisible by o_groups"
            )
        if (self.num_attention_heads * self.head_dim) % self.o_groups:
            raise CompressedAttentionError(
                "stacked attention width must be divisible by o_groups"
            )
        if self.rope_theta <= 0 or self.compress_rope_theta <= 0:
            raise CompressedAttentionError("RoPE theta values must be positive")
        if self.rms_norm_eps <= 0:
            raise CompressedAttentionError("rms_norm_eps must be positive")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise CompressedAttentionError(
                "attention_dropout must be in [0, 1)"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        ).to(x.dtype)


class InterleavedPartialRotaryEmbedding(nn.Module):
    """DeepSeek-V4-style interleaved RoPE over the trailing rotary slice."""

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("rotary_dim must be a positive even integer")
        if max_position_embeddings <= 0 or base <= 0:
            raise ValueError(
                "max_position_embeddings and base must be positive"
            )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=torch.float32,
                )
                / rotary_dim
            )
        )
        self.rotary_dim = int(rotary_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(
                "position_ids must have shape [batch, sequence]"
            )
        if position_ids.numel() and int(position_ids.min()) < 0:
            raise ValueError("position_ids must be non-negative")
        if (
            position_ids.numel()
            and int(position_ids.max()) >= self.max_position_embeddings
        ):
            raise ValueError(
                "position_ids exceed configured maximum"
            )
        freqs = (
            position_ids.to(
                device=device,
                dtype=torch.float32,
            ).unsqueeze(-1)
            * self.inv_freq.to(device=device)
        )
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        paired = x.reshape(*shape[:-1], -1, 2)
        first = paired[..., 0]
        second = paired[..., 1]
        return torch.stack(
            (-second, first),
            dim=-1,
        ).reshape(shape)

    def apply(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "rotary input must have shape [batch, heads, sequence, head_dim]"
            )
        if self.rotary_dim > x.shape[-1]:
            raise ValueError(
                "rotary_dim exceeds attention head dimension"
            )
        cos, sin = self.cos_sin(
            position_ids,
            dtype=x.dtype,
            device=x.device,
        )
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
        nope = x[..., :-self.rotary_dim]
        rope = x[..., -self.rotary_dim:]
        sign = -1.0 if inverse else 1.0
        rotated = (
            rope.float() * cos.float()
            + sign
            * self._rotate_pairs(rope).float()
            * sin.float()
        ).to(x.dtype)
        return torch.cat((nope, rotated), dim=-1)


class GroupedLinear(nn.Linear):
    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        bias: bool = False,
    ) -> None:
        if (
            in_features_per_group <= 0
            or out_features <= 0
            or n_groups <= 0
        ):
            raise ValueError(
                "grouped-linear dimensions must be positive"
            )
        if out_features % n_groups:
            raise ValueError(
                "out_features must be divisible by n_groups"
            )
        super().__init__(
            in_features_per_group,
            out_features,
            bias=bias,
        )
        self.n_groups = int(n_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-2] != self.n_groups:
            raise ValueError(
                f"grouped linear expects penultimate dimension {self.n_groups}"
            )
        hidden_dim = x.shape[-1]
        if hidden_dim != self.in_features:
            raise ValueError(
                f"grouped linear expected per-group width {self.in_features}, got {hidden_dim}"
            )
        input_shape = x.shape[:-2]
        weight = self.weight.view(
            self.n_groups,
            -1,
            hidden_dim,
        ).transpose(1, 2)
        flat = x.reshape(
            -1,
            self.n_groups,
            hidden_dim,
        ).transpose(0, 1)
        output = torch.bmm(flat, weight).transpose(0, 1)
        if self.bias is not None:
            bias = self.bias.view(
                self.n_groups,
                -1,
            )
            output = output + bias.unsqueeze(0)
        return output.reshape(
            *input_shape,
            self.n_groups,
            -1,
        )


@dataclass(frozen=True)
class _Segment:
    batch_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _segments(
    *,
    batch_size: int,
    sequence_length: int,
    attention_mask: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    device: torch.device,
) -> tuple[_Segment, ...]:
    if attention_mask is None:
        valid = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=device,
        )
    else:
        if attention_mask.shape != (batch_size, sequence_length):
            raise CompressedAttentionError(
                "attention_mask shape does not match hidden states"
            )
        valid = attention_mask.to(
            device=device,
            dtype=torch.bool,
        )
    if document_ids is not None:
        if document_ids.shape != valid.shape:
            raise CompressedAttentionError(
                "document_ids shape does not match hidden states"
            )
        if document_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise CompressedAttentionError(
                "document_ids must be integer typed"
            )
        docs = document_ids.to(device)
    else:
        docs = None

    result: list[_Segment] = []
    for batch_index in range(batch_size):
        current_start: int | None = None
        current_doc: int | None = None
        seen: set[int] = set()
        previous_valid = False
        for index in range(sequence_length):
            if not bool(valid[batch_index, index]):
                if current_start is not None:
                    result.append(
                        _Segment(
                            batch_index,
                            current_start,
                            index,
                        )
                    )
                    current_start = None
                    current_doc = None
                previous_valid = False
                continue
            doc = (
                int(docs[batch_index, index])
                if docs is not None
                else batch_index
            )
            if current_start is None:
                if docs is not None and doc in seen:
                    raise CompressedAttentionError(
                        "document_ids cannot reappear in non-contiguous segments"
                    )
                current_start = index
                current_doc = doc
                seen.add(doc)
            elif doc != current_doc:
                result.append(
                    _Segment(
                        batch_index,
                        current_start,
                        index,
                    )
                )
                if doc in seen:
                    raise CompressedAttentionError(
                        "document_ids cannot reappear in non-contiguous segments"
                    )
                current_start = index
                current_doc = doc
                seen.add(doc)
            previous_valid = True
        if current_start is not None:
            result.append(
                _Segment(
                    batch_index,
                    current_start,
                    sequence_length,
                )
            )
    if not result:
        raise CompressedAttentionError(
            "compressed attention input contains no valid tokens"
        )
    return tuple(result)


class _BaseCompressor(nn.Module):
    def __init__(
        self,
        config: CompressedAttentionConfig,
        *,
        compress_rate: int,
        projected_width: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.compress_rate = int(compress_rate)
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(
            config.hidden_size,
            projected_width,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            projected_width,
            bias=False,
        )
        self.position_bias = nn.Parameter(
            torch.empty(
                self.compress_rate,
                projected_width,
            )
        )
        self.kv_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.rotary = InterleavedPartialRotaryEmbedding(
            config.rope_head_dim,
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else 1,
            config.compress_rope_theta,
        )


def _compress_hca(
    kv: torch.Tensor,
    gate: torch.Tensor,
    *,
    position_bias: torch.Tensor,
    kv_norm: nn.Module,
    compress_rate: int,
) -> torch.Tensor:
    batch, sequence, head_dim = kv.shape
    usable = (sequence // compress_rate) * compress_rate
    if usable == 0:
        return kv.new_zeros((batch, 0, head_dim))
    n_windows = usable // compress_rate
    kv = kv[:, :usable].view(
        batch,
        n_windows,
        compress_rate,
        head_dim,
    )
    gate = gate[:, :usable].view(
        batch,
        n_windows,
        compress_rate,
        head_dim,
    )
    gate = gate + position_bias.view(
        1,
        1,
        compress_rate,
        head_dim,
    )
    weights = gate.softmax(
        dim=2,
        dtype=torch.float32,
    ).to(kv.dtype)
    return kv_norm((kv * weights).sum(dim=2))


def _compress_csa_overlap(
    kv: torch.Tensor,
    gate: torch.Tensor,
    *,
    position_bias: torch.Tensor,
    kv_norm: nn.Module,
    compress_rate: int,
    head_dim: int,
) -> torch.Tensor:
    batch, sequence, projected = kv.shape
    if projected != 2 * head_dim:
        raise CompressedAttentionError(
            "CSA projected width must equal 2 * head_dim"
        )
    usable = (sequence // compress_rate) * compress_rate
    if usable == 0:
        return kv.new_zeros((batch, 0, head_dim))
    n_windows = usable // compress_rate
    kv = kv[:, :usable].view(
        batch,
        n_windows,
        compress_rate,
        projected,
    )
    gate = gate[:, :usable].view(
        batch,
        n_windows,
        compress_rate,
        projected,
    )
    gate = gate + position_bias.view(
        1,
        1,
        compress_rate,
        projected,
    )

    new_kv = kv.new_zeros(
        batch,
        n_windows,
        2 * compress_rate,
        head_dim,
    )
    new_gate = gate.new_full(
        (
            batch,
            n_windows,
            2 * compress_rate,
            head_dim,
        ),
        float("-inf"),
    )
    new_kv[:, :, compress_rate:] = kv[..., head_dim:]
    new_gate[:, :, compress_rate:] = gate[..., head_dim:]
    if n_windows > 1:
        new_kv[:, 1:, :compress_rate] = kv[
            :, :-1, :, :head_dim
        ]
        new_gate[:, 1:, :compress_rate] = gate[
            :, :-1, :, :head_dim
        ]

    weights = new_gate.softmax(
        dim=2,
        dtype=torch.float32,
    ).to(new_kv.dtype)
    return kv_norm((new_kv * weights).sum(dim=2))


class HCACompressor(nn.Module):
    def __init__(
        self,
        config: CompressedAttentionConfig,
        *,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.compress_rate = config.hca_compress_rate
        self.kv_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
        )
        self.position_bias = nn.Parameter(
            torch.empty(
                self.compress_rate,
                config.head_dim,
            )
        )
        self.kv_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.rotary = InterleavedPartialRotaryEmbedding(
            config.rope_head_dim,
            max_position_embeddings,
            config.compress_rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = _compress_hca(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            position_bias=self.position_bias,
            kv_norm=self.kv_norm,
            compress_rate=self.compress_rate,
        )
        positions = (
            torch.arange(
                compressed.shape[1],
                device=compressed.device,
                dtype=torch.long,
            )
            * self.compress_rate
        ).unsqueeze(0).expand(
            compressed.shape[0],
            -1,
        )
        if compressed.shape[1]:
            compressed = self.rotary.apply(
                compressed.unsqueeze(1),
                positions,
            ).squeeze(1)
        return compressed, positions


class LightningIndexer(nn.Module):
    def __init__(
        self,
        config: CompressedAttentionConfig,
        *,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.compress_rate = config.csa_compress_rate
        self.kv_proj = nn.Linear(
            config.hidden_size,
            2 * config.index_head_dim,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            2 * config.index_head_dim,
            bias=False,
        )
        self.position_bias = nn.Parameter(
            torch.empty(
                self.compress_rate,
                2 * config.index_head_dim,
            )
        )
        self.kv_norm = RMSNorm(
            config.index_head_dim,
            config.rms_norm_eps,
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
        )
        self.weights_proj = nn.Linear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
        )
        self.rotary = InterleavedPartialRotaryEmbedding(
            min(config.rope_head_dim, config.index_head_dim),
            max_position_embeddings,
            config.compress_rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        compressed = _compress_csa_overlap(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            position_bias=self.position_bias,
            kv_norm=self.kv_norm,
            compress_rate=self.compress_rate,
            head_dim=self.config.index_head_dim,
        )
        compressed_positions = (
            torch.arange(
                compressed.shape[1],
                device=compressed.device,
                dtype=torch.long,
            )
            * self.compress_rate
        ).unsqueeze(0).expand(
            compressed.shape[0],
            -1,
        )
        if compressed.shape[1]:
            compressed = self.rotary.apply(
                compressed.unsqueeze(1),
                compressed_positions,
            ).squeeze(1)

        batch, sequence, _ = hidden_states.shape
        q = self.q_b_proj(q_residual).view(
            batch,
            sequence,
            self.config.index_n_heads,
            self.config.index_head_dim,
        ).transpose(1, 2)
        q = self.rotary.apply(q, position_ids).transpose(1, 2)

        if compressed.shape[1] == 0:
            return torch.empty(
                batch,
                sequence,
                0,
                dtype=torch.long,
                device=hidden_states.device,
            )

        scores = torch.matmul(
            q.float(),
            compressed.transpose(-1, -2).float().unsqueeze(1),
        )
        scores = F.relu(scores) * (
            self.config.index_head_dim ** -0.5
        )
        weights = (
            self.weights_proj(hidden_states).float()
            * (self.config.index_n_heads ** -0.5)
        )
        scores = (
            scores
            * weights.unsqueeze(-1)
        ).sum(dim=2)

        threshold = (
            position_ids + 1
        ) // self.compress_rate
        entry_indices = torch.arange(
            compressed.shape[1],
            device=compressed.device,
        )
        future = (
            entry_indices.view(1, 1, -1)
            >= threshold.unsqueeze(-1)
        )
        scores = scores.masked_fill(
            future,
            float("-inf"),
        )
        top_k = min(
            self.config.index_topk,
            compressed.shape[1],
        )
        indices = scores.topk(
            top_k,
            dim=-1,
        ).indices
        invalid = indices >= threshold.unsqueeze(-1)
        return torch.where(
            invalid,
            torch.full_like(indices, -1),
            indices,
        )


class CSACompressor(nn.Module):
    def __init__(
        self,
        config: CompressedAttentionConfig,
        *,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.compress_rate = config.csa_compress_rate
        self.kv_proj = nn.Linear(
            config.hidden_size,
            2 * config.head_dim,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            2 * config.head_dim,
            bias=False,
        )
        self.position_bias = nn.Parameter(
            torch.empty(
                self.compress_rate,
                2 * config.head_dim,
            )
        )
        self.kv_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.rotary = InterleavedPartialRotaryEmbedding(
            config.rope_head_dim,
            max_position_embeddings,
            config.compress_rope_theta,
        )
        self.indexer = LightningIndexer(
            config,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = _compress_csa_overlap(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            position_bias=self.position_bias,
            kv_norm=self.kv_norm,
            compress_rate=self.compress_rate,
            head_dim=self.config.head_dim,
        )
        positions = (
            torch.arange(
                compressed.shape[1],
                device=compressed.device,
                dtype=torch.long,
            )
            * self.compress_rate
        ).unsqueeze(0).expand(
            compressed.shape[0],
            -1,
        )
        if compressed.shape[1]:
            compressed = self.rotary.apply(
                compressed.unsqueeze(1),
                positions,
            ).squeeze(1)
        indices = self.indexer(
            hidden_states,
            q_residual,
            position_ids,
        )
        return compressed, indices


def _attention_with_sink(
    query: torch.Tensor,
    key_value: torch.Tensor,
    additive_mask: torch.Tensor,
    sinks: torch.Tensor,
    *,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    if query.ndim != 4 or key_value.ndim != 4:
        raise ValueError(
            "query and key_value must be [batch, heads, sequence, head_dim]"
        )
    if key_value.shape[1] != 1:
        raise ValueError(
            "compressed attention requires a single shared K=V head"
        )
    repeated = key_value.expand(
        -1,
        query.shape[1],
        -1,
        -1,
    )
    logits = torch.matmul(
        query,
        repeated.transpose(-1, -2),
    ) * (query.shape[-1] ** -0.5)
    logits = logits + additive_mask
    sink_logits = sinks.view(
        1,
        -1,
        1,
        1,
    ).expand(
        query.shape[0],
        -1,
        query.shape[2],
        1,
    )
    combined = torch.cat(
        (logits.float(), sink_logits.float()),
        dim=-1,
    )
    combined = combined - combined.max(
        dim=-1,
        keepdim=True,
    ).values
    probabilities = F.softmax(
        combined,
        dim=-1,
        dtype=torch.float32,
    )[..., :-1]
    probabilities = F.dropout(
        probabilities,
        p=dropout_p,
        training=training,
    ).to(repeated.dtype)
    return torch.matmul(probabilities, repeated)


class CompressedContextAttention(nn.Module):
    """Stateless full-sequence DeepSeek-V4 CSA/HCA reference.

    Packed documents are evaluated independently, so sliding memory,
    compressor windows, indexer scores, and compressed memory never cross
    document boundaries. Decode caching is intentionally a separate runtime;
    this class is the exact training/reference path.
    """

    def __init__(
        self,
        model_hidden_size: int,
        max_position_embeddings: int,
        config: CompressedAttentionConfig,
        attention_type: CompressedAttentionType,
    ) -> None:
        super().__init__()
        if model_hidden_size != config.hidden_size:
            raise CompressedAttentionError(
                "model hidden_size does not match compressed-attention config"
            )
        self.config = config
        self.attention_type = CompressedAttentionType(
            attention_type
        )
        self.max_position_embeddings = int(
            max_position_embeddings
        )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_a_proj = nn.Linear(
            config.hidden_size,
            config.q_lora_rank,
            bias=False,
        )
        self.q_a_norm = RMSNorm(
            config.q_lora_rank,
            config.rms_norm_eps,
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.q_b_norm = UnweightedRMSNorm(
            config.rms_norm_eps
        )
        self.kv_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.rotary = InterleavedPartialRotaryEmbedding(
            config.rope_head_dim,
            max_position_embeddings,
            config.compress_rope_theta,
        )
        per_group = (
            config.num_attention_heads
            * config.head_dim
            // config.o_groups
        )
        self.o_a_proj = GroupedLinear(
            per_group,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
            bias=False,
        )
        self.o_b_proj = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.sinks = nn.Parameter(
            torch.empty(config.num_attention_heads)
        )
        self.compressor = (
            CSACompressor(
                config,
                max_position_embeddings=max_position_embeddings,
            )
            if self.attention_type is CompressedAttentionType.CSA
            else HCACompressor(
                config,
                max_position_embeddings=max_position_embeddings,
            )
        )

    def _segment_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, hidden = hidden_states.shape
        if batch != 1:
            raise CompressedAttentionError(
                "reference segment_forward expects batch=1"
            )
        local_positions = torch.arange(
            sequence,
            device=hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)

        q_residual = self.q_a_norm(
            self.q_a_proj(hidden_states)
        )
        query = self.q_b_proj(q_residual).view(
            1,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        query = self.q_b_norm(query)
        query = self.rotary.apply(
            query,
            local_positions,
        )

        local_kv = self.kv_norm(
            self.kv_proj(hidden_states)
        ).unsqueeze(1)
        local_kv = self.rotary.apply(
            local_kv,
            local_positions,
        )

        if self.attention_type is CompressedAttentionType.CSA:
            compressed, indices = self.compressor(
                hidden_states,
                q_residual,
                local_positions,
            )
            compressed = compressed.unsqueeze(1)
        else:
            compressed, _ = self.compressor(
                hidden_states,
            )
            compressed = compressed.unsqueeze(1)
            indices = None

        key_value = torch.cat(
            (local_kv, compressed),
            dim=2,
        )
        compressed_len = compressed.shape[2]
        total_kv = sequence + compressed_len

        mask = hidden_states.new_full(
            (
                1,
                1,
                sequence,
                total_kv,
            ),
            float("-inf"),
        )
        qpos = torch.arange(
            sequence,
            device=hidden_states.device,
        )
        kpos = torch.arange(
            sequence,
            device=hidden_states.device,
        )
        local_allowed = (
            kpos.view(1, -1)
            <= qpos.view(-1, 1)
        ) & (
            kpos.view(1, -1)
            >= (
                qpos.view(-1, 1)
                - self.config.sliding_window
                + 1
            )
        )
        mask[
            0,
            0,
            :,
            :sequence,
        ] = torch.where(
            local_allowed,
            torch.zeros_like(
                local_allowed,
                dtype=hidden_states.dtype,
            ),
            torch.full_like(
                local_allowed,
                float("-inf"),
                dtype=hidden_states.dtype,
            ),
        )

        if compressed_len:
            if (
                self.attention_type
                is CompressedAttentionType.HCA
            ):
                threshold = (
                    local_positions + 1
                ) // self.config.hca_compress_rate
                entries = torch.arange(
                    compressed_len,
                    device=hidden_states.device,
                )
                allowed = (
                    entries.view(1, 1, -1)
                    < threshold.unsqueeze(-1)
                )
                mask[
                    :,
                    :,
                    :,
                    sequence:,
                ] = torch.where(
                    allowed.unsqueeze(1),
                    torch.zeros(
                        (),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                    torch.full(
                        (),
                        float("-inf"),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                )
            else:
                assert indices is not None
                selected = torch.zeros(
                    (
                        1,
                        sequence,
                        compressed_len,
                    ),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                valid = indices >= 0
                safe = indices.clamp_min(0)
                selected.scatter_(
                    -1,
                    safe,
                    valid,
                )
                mask[
                    :,
                    :,
                    :,
                    sequence:,
                ] = torch.where(
                    selected.unsqueeze(1),
                    torch.zeros(
                        (),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                    torch.full(
                        (),
                        float("-inf"),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                )

        output = _attention_with_sink(
            query,
            key_value,
            mask,
            self.sinks,
            dropout_p=self.config.attention_dropout,
            training=self.training,
        )
        output = self.rotary.apply(
            output,
            local_positions,
            inverse=True,
        )
        output = output.transpose(
            1,
            2,
        ).contiguous()
        grouped = output.reshape(
            1,
            sequence,
            self.config.o_groups,
            -1,
        )
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise CompressedAttentionError(
                "hidden_states must have shape [batch, sequence, hidden]"
            )
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise CompressedAttentionError(
                "hidden-state width does not match compressed-attention config"
            )
        batch, sequence, hidden = hidden_states.shape
        segments = _segments(
            batch_size=batch,
            sequence_length=sequence,
            attention_mask=attention_mask,
            document_ids=document_ids,
            device=hidden_states.device,
        )
        output = torch.zeros_like(hidden_states)
        for segment in segments:
            states = hidden_states[
                segment.batch_index : segment.batch_index + 1,
                segment.start : segment.end,
                :,
            ]
            result = self._segment_forward(states)
            output[
                segment.batch_index : segment.batch_index + 1,
                segment.start : segment.end,
                :,
            ] = result
        return output
