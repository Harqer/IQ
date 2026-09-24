from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm
from ..position import (
    InterleavedRotaryEmbedding,
    apply_v4_interleaved_rotary_at_end,
)
from .masking import prepare_causal_attention


class CompressedAttentionError(ValueError):
    pass


class CompressedAttentionMode(str, Enum):
    CSA = "csa"
    HCA = "hca"


@dataclass(frozen=True)
class CompressedAttentionConfig:
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    rope_head_dim: int
    max_position_embeddings: int
    sliding_window: int
    o_groups: int
    o_lora_rank: int
    csa_compress_rate: int = 4
    hca_compress_rate: int = 128
    index_n_heads: int = 8
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        ints = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "rope_head_dim": self.rope_head_dim,
            "max_position_embeddings": self.max_position_embeddings,
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
        if self.rope_head_dim > self.head_dim or self.rope_head_dim % 2:
            raise CompressedAttentionError(
                "rope_head_dim must be even and <= head_dim"
            )
        if self.rope_head_dim > self.index_head_dim:
            raise CompressedAttentionError(
                "rope_head_dim must be <= index_head_dim"
            )
        if self.num_attention_heads % self.o_groups:
            raise CompressedAttentionError(
                "num_attention_heads must be divisible by o_groups"
            )
        if self.compress_rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise CompressedAttentionError(
                "compress_rope_theta and rms_norm_eps must be positive"
            )
        if not (0.0 <= self.attention_dropout < 1.0):
            raise CompressedAttentionError(
                "attention_dropout must be in [0, 1)"
            )


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.float()
            * torch.rsqrt(
                x.float().square().mean(dim=-1, keepdim=True) + self.eps
            )
        ).to(x.dtype)


class GroupedLinear(nn.Module):
    """Block-diagonal grouped projection used before the final output mix."""

    def __init__(
        self,
        in_features_per_group: int,
        out_features_per_group: int,
        num_groups: int,
    ) -> None:
        super().__init__()
        if min(in_features_per_group, out_features_per_group, num_groups) <= 0:
            raise ValueError("grouped-linear dimensions must be positive")
        self.in_features_per_group = int(in_features_per_group)
        self.out_features_per_group = int(out_features_per_group)
        self.num_groups = int(num_groups)
        self.weight = nn.Parameter(
            torch.empty(
                num_groups,
                out_features_per_group,
                in_features_per_group,
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (
            self.num_groups,
            self.in_features_per_group,
        ):
            raise ValueError(
                "GroupedLinear input must end in "
                f"[{self.num_groups}, {self.in_features_per_group}]"
            )
        return torch.einsum("...gi,gri->...gr", x, self.weight)


@dataclass
class CompressionEntries:
    values: torch.Tensor
    valid: torch.Tensor
    segment_ids: torch.Tensor
    positions: torch.Tensor
    ready_positions: torch.Tensor

    @property
    def counts(self) -> torch.Tensor:
        return self.valid.sum(dim=-1)


@dataclass
class CompressedContextOutput:
    hidden_states: torch.Tensor
    compressed_entry_counts: torch.Tensor
    selected_indices: torch.Tensor | None


def _segment_ids(
    valid_tokens: torch.Tensor,
    document_ids: torch.Tensor | None,
) -> torch.Tensor:
    batch, seq_len = valid_tokens.shape
    result = torch.full(
        (batch, seq_len),
        -1,
        dtype=torch.long,
        device=valid_tokens.device,
    )
    for b in range(batch):
        next_segment = 0
        previous_doc: int | None = None
        previous_valid = False
        current_segment = -1
        for t in range(seq_len):
            if not bool(valid_tokens[b, t]):
                previous_valid = False
                previous_doc = None
                continue
            doc = int(document_ids[b, t]) if document_ids is not None else 0
            if not previous_valid or doc != previous_doc:
                current_segment = next_segment
                next_segment += 1
            result[b, t] = current_segment
            previous_valid = True
            previous_doc = doc
    return result


def _ranges_for_row(segment_ids: torch.Tensor) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    start: int | None = None
    segment = -1
    for t in range(segment_ids.numel() + 1):
        current = int(segment_ids[t]) if t < segment_ids.numel() else -1
        if start is None:
            if current >= 0:
                start = t
                segment = current
            continue
        if current != segment:
            ranges.append((start, t, segment))
            start = t if current >= 0 else None
            segment = current
    return ranges


def _validate_segment_positions(
    position_ids: torch.Tensor,
    segment_ids: torch.Tensor,
) -> None:
    for b in range(position_ids.shape[0]):
        for start, end, _ in _ranges_for_row(segment_ids[b]):
            positions = position_ids[b, start:end]
            if positions.numel() > 1 and not bool(
                torch.equal(
                    positions[1:] - positions[:-1],
                    torch.ones_like(positions[1:]),
                )
            ):
                raise CompressedAttentionError(
                    "compressed attention requires consecutive position_ids "
                    "within each packed-document segment"
                )


def _pack_entry_rows(
    row_values: list[list[torch.Tensor]],
    row_segments: list[list[int]],
    row_positions: list[list[int]],
    row_ready: list[list[int]],
    *,
    head_dim: int,
    reference: torch.Tensor,
) -> CompressionEntries:
    batch = len(row_values)
    max_entries = max((len(row) for row in row_values), default=0)
    values_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    segment_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    ready_rows: list[torch.Tensor] = []

    for b in range(batch):
        count = len(row_values[b])
        if count:
            values = torch.stack(row_values[b], dim=0)
        else:
            values = reference.new_zeros((0, head_dim))
        pad = max_entries - count
        values_rows.append(F.pad(values, (0, 0, 0, pad)))
        valid_rows.append(
            torch.cat(
                [
                    torch.ones(count, dtype=torch.bool, device=reference.device),
                    torch.zeros(pad, dtype=torch.bool, device=reference.device),
                ]
            )
        )
        segment_rows.append(
            torch.tensor(
                row_segments[b] + [-1] * pad,
                dtype=torch.long,
                device=reference.device,
            )
        )
        position_rows.append(
            torch.tensor(
                row_positions[b] + [0] * pad,
                dtype=torch.long,
                device=reference.device,
            )
        )
        ready_rows.append(
            torch.tensor(
                row_ready[b] + [-1] * pad,
                dtype=torch.long,
                device=reference.device,
            )
        )

    if max_entries == 0:
        return CompressionEntries(
            values=reference.new_zeros((batch, 0, head_dim)),
            valid=torch.zeros((batch, 0), dtype=torch.bool, device=reference.device),
            segment_ids=torch.full((batch, 0), -1, dtype=torch.long, device=reference.device),
            positions=torch.zeros((batch, 0), dtype=torch.long, device=reference.device),
            ready_positions=torch.full((batch, 0), -1, dtype=torch.long, device=reference.device),
        )
    return CompressionEntries(
        values=torch.stack(values_rows, dim=0),
        valid=torch.stack(valid_rows, dim=0),
        segment_ids=torch.stack(segment_rows, dim=0),
        positions=torch.stack(position_rows, dim=0),
        ready_positions=torch.stack(ready_rows, dim=0),
    )


def _compress_hca_projected(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    norm: nn.Module,
    segment_ids: torch.Tensor,
    position_ids: torch.Tensor,
    compress_rate: int,
) -> CompressionEntries:
    batch, _, head_dim = kv.shape
    row_values: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    row_segments: list[list[int]] = [[] for _ in range(batch)]
    row_positions: list[list[int]] = [[] for _ in range(batch)]
    row_ready: list[list[int]] = [[] for _ in range(batch)]

    for b in range(batch):
        for start, end, segment in _ranges_for_row(segment_ids[b]):
            usable = ((end - start) // compress_rate) * compress_rate
            for offset in range(0, usable, compress_rate):
                left = start + offset
                right = left + compress_rate
                window_kv = kv[b, left:right]
                window_gate = gate[b, left:right] + position_bias
                weights = torch.softmax(window_gate.float(), dim=0).to(window_kv.dtype)
                compressed = norm((window_kv * weights).sum(dim=0))
                row_values[b].append(compressed)
                row_segments[b].append(segment)
                row_positions[b].append(int(position_ids[b, left]))
                row_ready[b].append(int(position_ids[b, right - 1]))

    return _pack_entry_rows(
        row_values,
        row_segments,
        row_positions,
        row_ready,
        head_dim=head_dim,
        reference=kv,
    )


def _compress_csa_projected(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    norm: nn.Module,
    segment_ids: torch.Tensor,
    position_ids: torch.Tensor,
    compress_rate: int,
    head_dim: int,
) -> CompressionEntries:
    batch = kv.shape[0]
    row_values: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    row_segments: list[list[int]] = [[] for _ in range(batch)]
    row_positions: list[list[int]] = [[] for _ in range(batch)]
    row_ready: list[list[int]] = [[] for _ in range(batch)]

    for b in range(batch):
        for start, end, segment in _ranges_for_row(segment_ids[b]):
            n_windows = (end - start) // compress_rate
            if n_windows == 0:
                continue
            chunks_kv = kv[b, start : start + n_windows * compress_rate].view(
                n_windows,
                compress_rate,
                2 * head_dim,
            )
            chunks_gate = (
                gate[b, start : start + n_windows * compress_rate].view(
                    n_windows,
                    compress_rate,
                    2 * head_dim,
                )
                + position_bias.unsqueeze(0)
            )
            for window in range(n_windows):
                current_cb = chunks_kv[window, :, head_dim:]
                current_gate_cb = chunks_gate[window, :, head_dim:]
                if window == 0:
                    combined_kv = current_cb
                    combined_gate = current_gate_cb
                else:
                    previous_ca = chunks_kv[window - 1, :, :head_dim]
                    previous_gate_ca = chunks_gate[window - 1, :, :head_dim]
                    combined_kv = torch.cat((previous_ca, current_cb), dim=0)
                    combined_gate = torch.cat(
                        (previous_gate_ca, current_gate_cb),
                        dim=0,
                    )
                weights = torch.softmax(
                    combined_gate.float(),
                    dim=0,
                ).to(combined_kv.dtype)
                compressed = norm((combined_kv * weights).sum(dim=0))
                left = start + window * compress_rate
                right = left + compress_rate
                row_values[b].append(compressed)
                row_segments[b].append(segment)
                row_positions[b].append(int(position_ids[b, left]))
                row_ready[b].append(int(position_ids[b, right - 1]))

    return _pack_entry_rows(
        row_values,
        row_segments,
        row_positions,
        row_ready,
        head_dim=head_dim,
        reference=kv,
    )


def _rotate_entries(
    entries: CompressionEntries,
    rotary: InterleavedRotaryEmbedding,
    rope_head_dim: int,
) -> CompressionEntries:
    if entries.values.shape[1] == 0:
        return entries
    cos, sin = rotary.cos_sin(
        entries.positions,
        dtype=entries.values.dtype,
        device=entries.values.device,
    )
    rotated = apply_v4_interleaved_rotary_at_end(
        entries.values.unsqueeze(1),
        cos,
        sin,
        rope_head_dim,
    ).squeeze(1)
    return CompressionEntries(
        values=rotated,
        valid=entries.valid,
        segment_ids=entries.segment_ids,
        positions=entries.positions,
        ready_positions=entries.ready_positions,
    )


class LightningIndexer(nn.Module):
    def __init__(self, config: CompressedAttentionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.index_head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * d, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * d, bias=False)
        self.position_bias = nn.Parameter(
            torch.zeros(config.csa_compress_rate, 2 * d)
        )
        self.kv_norm = RMSNorm(d, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * d,
            bias=False,
        )
        self.weights_proj = nn.Linear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
        )
        self.rotary = InterleavedRotaryEmbedding(
            config.rope_head_dim,
            config.max_position_embeddings,
            config.compress_rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        d = self.config.index_head_dim
        compressed = _compress_csa_projected(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            self.position_bias,
            self.kv_norm,
            segment_ids,
            position_ids,
            self.config.csa_compress_rate,
            d,
        )
        compressed = _rotate_entries(
            compressed,
            self.rotary,
            self.config.rope_head_dim,
        )
        result = torch.full(
            (batch, seq_len, self.config.index_topk),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        compressed_len = compressed.values.shape[1]
        if compressed_len == 0:
            return result

        q = self.q_b_proj(q_residual).view(
            batch,
            seq_len,
            self.config.index_n_heads,
            d,
        ).transpose(1, 2)
        cos, sin = self.rotary.cos_sin(
            position_ids,
            dtype=q.dtype,
            device=q.device,
        )
        q = apply_v4_interleaved_rotary_at_end(
            q,
            cos,
            sin,
            self.config.rope_head_dim,
        )

        scores = torch.einsum(
            "bhsd,btd->bsht",
            q.float(),
            compressed.values.float(),
        )
        scores = F.relu(scores) * (d**-0.5)
        query_weights = (
            self.weights_proj(hidden_states).float()
            * (self.config.index_n_heads**-0.5)
        )
        scores = (scores * query_weights.unsqueeze(-1)).sum(dim=2)

        visible = (
            compressed.valid[:, None, :]
            & (compressed.segment_ids[:, None, :] == segment_ids[:, :, None])
            & (
                compressed.ready_positions[:, None, :]
                <= position_ids[:, :, None]
            )
            & (segment_ids[:, :, None] >= 0)
        )
        masked_scores = scores.masked_fill(~visible, float("-inf"))
        k = min(self.config.index_topk, compressed_len)
        if k == 0:
            return result
        top_scores, top_indices = torch.topk(
            masked_scores,
            k=k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        valid_selection = torch.isfinite(top_scores)
        top_indices = torch.where(
            valid_selection,
            top_indices,
            torch.full_like(top_indices, -1),
        )
        result[..., :k] = top_indices
        return result


class CSACompressor(nn.Module):
    def __init__(self, config: CompressedAttentionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * d, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * d, bias=False)
        self.position_bias = nn.Parameter(
            torch.zeros(config.csa_compress_rate, 2 * d)
        )
        self.kv_norm = RMSNorm(d, config.rms_norm_eps)
        self.rotary = InterleavedRotaryEmbedding(
            config.rope_head_dim,
            config.max_position_embeddings,
            config.compress_rope_theta,
        )
        self.indexer = LightningIndexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[CompressionEntries, torch.Tensor]:
        entries = _compress_csa_projected(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            self.position_bias,
            self.kv_norm,
            segment_ids,
            position_ids,
            self.config.csa_compress_rate,
            self.config.head_dim,
        )
        entries = _rotate_entries(
            entries,
            self.rotary,
            self.config.rope_head_dim,
        )
        selected = self.indexer(
            hidden_states,
            q_residual,
            position_ids,
            segment_ids,
        )
        return entries, selected


class HCACompressor(nn.Module):
    def __init__(self, config: CompressedAttentionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.position_bias = nn.Parameter(
            torch.zeros(config.hca_compress_rate, d)
        )
        self.kv_norm = RMSNorm(d, config.rms_norm_eps)
        self.rotary = InterleavedRotaryEmbedding(
            config.rope_head_dim,
            config.max_position_embeddings,
            config.compress_rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> CompressionEntries:
        entries = _compress_hca_projected(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            self.position_bias,
            self.kv_norm,
            segment_ids,
            position_ids,
            self.config.hca_compress_rate,
        )
        return _rotate_entries(
            entries,
            self.rotary,
            self.config.rope_head_dim,
        )


class CompressedContextAttention(nn.Module):
    """Full-sequence PyTorch reference for DeepSeek-V4-style CSA/HCA.

    The reference intentionally performs only O(window + selected-compressed)
    value attention per query. Streaming cache state is implemented separately;
    this module is the training/correctness oracle for that optimized path.
    """

    def __init__(
        self,
        config: CompressedAttentionConfig,
        mode: CompressedAttentionMode,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = CompressedAttentionMode(mode)
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
        self.q_b_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.kv_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        heads_per_group = config.num_attention_heads // config.o_groups
        self.o_a_proj = GroupedLinear(
            heads_per_group * config.head_dim,
            config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.sinks = nn.Parameter(
            torch.zeros(config.num_attention_heads)
        )
        self.rotary = InterleavedRotaryEmbedding(
            config.rope_head_dim,
            config.max_position_embeddings,
            config.compress_rope_theta,
        )
        self.compressor: CSACompressor | HCACompressor
        if self.mode is CompressedAttentionMode.CSA:
            self.compressor = CSACompressor(config)
        else:
            self.compressor = HCACompressor(config)

    def _local_indices(
        self,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
        b: int,
        t: int,
    ) -> torch.Tensor:
        segment = int(segment_ids[b, t])
        if segment < 0:
            return torch.empty(
                0,
                dtype=torch.long,
                device=segment_ids.device,
            )
        q_pos = position_ids[b, t]
        allowed = (
            (segment_ids[b] == segment)
            & (position_ids[b] <= q_pos)
            & (
                position_ids[b]
                >= q_pos - (self.config.sliding_window - 1)
            )
        )
        return torch.where(allowed)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> CompressedContextOutput:
        if hidden_states.ndim != 3:
            raise ValueError(
                "compressed context input must have shape [batch, sequence, hidden]"
            )
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "compressed context input width does not match config.hidden_size"
            )
        batch, seq_len, _ = hidden_states.shape
        prepared = prepare_causal_attention(
            batch_size=batch,
            sequence_length=seq_len,
            device=hidden_states.device,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        docs = (
            document_ids.to(hidden_states.device)
            if document_ids is not None
            else None
        )
        segment_ids = _segment_ids(prepared.valid_tokens, docs)
        _validate_segment_positions(
            prepared.position_ids,
            segment_ids,
        )

        q_residual = self.q_a_norm(
            self.q_a_proj(hidden_states)
        )
        q = self.q_b_proj(q_residual).view(
            batch,
            seq_len,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        q = self.q_b_norm(q)
        cos, sin = self.rotary.cos_sin(
            prepared.position_ids,
            dtype=q.dtype,
            device=q.device,
        )
        q = apply_v4_interleaved_rotary_at_end(
            q,
            cos,
            sin,
            self.config.rope_head_dim,
        )

        local_kv = self.kv_norm(
            self.kv_proj(hidden_states)
        ).unsqueeze(1)
        local_kv = apply_v4_interleaved_rotary_at_end(
            local_kv,
            cos,
            sin,
            self.config.rope_head_dim,
        )

        selected_indices: torch.Tensor | None = None
        if self.mode is CompressedAttentionMode.CSA:
            assert isinstance(self.compressor, CSACompressor)
            compressed, selected_indices = self.compressor(
                hidden_states,
                q_residual,
                prepared.position_ids,
                segment_ids,
            )
        else:
            assert isinstance(self.compressor, HCACompressor)
            compressed = self.compressor(
                hidden_states,
                prepared.position_ids,
                segment_ids,
            )

        output_rows: list[torch.Tensor] = []
        scale = self.config.head_dim**-0.5
        for b in range(batch):
            query_outputs: list[torch.Tensor] = []
            for t in range(seq_len):
                if int(segment_ids[b, t]) < 0:
                    query_outputs.append(q[b, :, t, :] * 0.0)
                    continue

                local_indices = self._local_indices(
                    segment_ids,
                    prepared.position_ids,
                    b,
                    t,
                )
                local_values = local_kv[
                    b,
                    0,
                    local_indices,
                    :,
                ]

                if self.mode is CompressedAttentionMode.CSA:
                    assert selected_indices is not None
                    selected = selected_indices[b, t]
                    selected = selected[selected >= 0]
                    compressed_values = compressed.values[b, selected]
                else:
                    visible = (
                        compressed.valid[b]
                        & (
                            compressed.segment_ids[b]
                            == segment_ids[b, t]
                        )
                        & (
                            compressed.ready_positions[b]
                            <= prepared.position_ids[b, t]
                        )
                    )
                    compressed_values = compressed.values[
                        b,
                        torch.where(visible)[0],
                    ]

                values = torch.cat(
                    (local_values, compressed_values),
                    dim=0,
                )
                logits = torch.matmul(
                    q[b, :, t, :].float(),
                    values.transpose(0, 1).float(),
                ) * scale
                sink_logits = self.sinks.float().unsqueeze(-1)
                combined_logits = torch.cat(
                    (logits, sink_logits),
                    dim=-1,
                )
                probabilities = torch.softmax(
                    combined_logits,
                    dim=-1,
                    dtype=torch.float32,
                )[..., :-1]
                probabilities = F.dropout(
                    probabilities,
                    p=self.config.attention_dropout,
                    training=self.training,
                ).to(values.dtype)
                query_outputs.append(
                    torch.matmul(probabilities, values)
                )
            output_rows.append(
                torch.stack(query_outputs, dim=1)
            )
        attn_output = torch.stack(output_rows, dim=0)

        attn_output = apply_v4_interleaved_rotary_at_end(
            attn_output,
            cos,
            sin,
            self.config.rope_head_dim,
            inverse=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        grouped = attn_output.reshape(
            batch,
            seq_len,
            self.config.o_groups,
            -1,
        )
        grouped = self.o_a_proj(grouped).reshape(
            batch,
            seq_len,
            -1,
        )
        hidden = self.o_b_proj(grouped)
        hidden = hidden * prepared.valid_tokens.unsqueeze(-1).to(
            hidden.dtype
        )
        return CompressedContextOutput(
            hidden_states=hidden,
            compressed_entry_counts=compressed.counts,
            selected_indices=selected_indices,
        )


class CompressedContextLayer(nn.Module):
    """Pre-norm residual CSA/HCA layer for the standard-residual hybrid path."""

    def __init__(
        self,
        config: CompressedAttentionConfig,
        mode: CompressedAttentionMode,
        *,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= residual_dropout < 1.0):
            raise ValueError("residual_dropout must be in [0, 1)")
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attention = CompressedContextAttention(
            config,
            mode,
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> CompressedContextOutput:
        output = self.attention(
            self.norm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        output.hidden_states = hidden_states + self.residual_dropout(
            output.hidden_states
        )
        return output
