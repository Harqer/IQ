from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import HeadRMSNorm, RMSNorm, UnweightedRMSNorm
from ..position import (
    InterleavedRotaryEmbedding,
    apply_inverse_partial_rotary_at_end,
    apply_partial_rotary_at_end,
)
from .masking import prepare_causal_attention


class CompressedContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexerSegmentScores:
    batch_index: int
    token_indices: torch.Tensor
    scores: torch.Tensor
    valid_mask: torch.Tensor
    selected_indices: torch.Tensor


@dataclass(frozen=True)
class CompressedContextConfig:
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    partial_rotary_dim: int
    max_position_embeddings: int
    sliding_window: int = 128
    csa_compress_rate: int = 4
    hca_compress_rate: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 160000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    projection_bias: bool = False
    normalize_query_output: bool = True
    normalize_candidate_nonrotary_only: bool = False
    direct_token_indexer: bool = False

    def __post_init__(self) -> None:
        ints = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "partial_rotary_dim": self.partial_rotary_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_window": self.sliding_window,
            "csa_compress_rate": self.csa_compress_rate,
            "hca_compress_rate": self.hca_compress_rate,
            "o_groups": self.o_groups,
            "o_lora_rank": self.o_lora_rank,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise ValueError(
                f"positive compressed-context dimensions required: {', '.join(bad)}"
            )
        if self.partial_rotary_dim % 2:
            raise ValueError("partial_rotary_dim must be even")
        if self.partial_rotary_dim > self.head_dim:
            raise ValueError("partial_rotary_dim cannot exceed head_dim")
        if self.partial_rotary_dim > self.index_head_dim:
            raise ValueError("partial_rotary_dim cannot exceed index_head_dim")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if self.compress_rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("compress_rope_theta and rms_norm_eps must be positive")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError("attention_dropout must be in [0, 1)")


@dataclass
class CompressedAttentionCache:
    """One causal segment of one attention layer (inference only).

    Buffers grow geometrically: appending a token never recopies the entire
    historical index or compressed KV buffer. Do not share between documents.
    """

    length: int
    hidden_tail: torch.Tensor
    positions_tail: torch.Tensor
    local_kv: torch.Tensor
    compressed: torch.Tensor
    compressed_count: int
    index_keys: torch.Tensor
    index_count: int


def _append_cache_row(buffer: torch.Tensor, count: int, row: torch.Tensor) -> torch.Tensor:
    if row.ndim != 1 or row.shape[0] != buffer.shape[1]:
        raise ValueError("attention cache row has incorrect dimensions")
    if count == buffer.shape[0]:
        larger = buffer.new_empty((max(16, 2 * buffer.shape[0]), buffer.shape[1]))
        larger[:count].copy_(buffer[:count])
        buffer = larger
    buffer[count].copy_(row)
    return buffer


class GroupedLowRankOutput(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        groups: int,
        rank: int,
        hidden_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % groups:
            raise ValueError("num_heads must be divisible by groups")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.groups = groups
        self.rank = rank
        self.heads_per_group = num_heads // groups
        in_per_group = self.heads_per_group * head_dim
        self.weight = nn.Parameter(torch.empty(groups, rank, in_per_group))
        self.out_proj = nn.Linear(groups * rank, hidden_size, bias=bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError(
                "grouped output expects [tokens, num_heads, head_dim]"
            )
        grouped = x.reshape(
            x.shape[0],
            self.groups,
            self.heads_per_group * self.head_dim,
        )
        reduced = torch.einsum("tgi,gri->tgr", grouped, self.weight)
        return self.out_proj(reduced.flatten(1))


def _segments(
    valid: torch.Tensor,
    document_ids: torch.Tensor | None,
) -> list[list[int]]:
    result: list[list[int]] = []
    current: list[int] = []
    previous_col: int | None = None
    previous_doc: int | None = None
    for col in range(valid.shape[0]):
        if not bool(valid[col]):
            if current:
                result.append(current)
                current = []
            previous_col = None
            previous_doc = None
            continue
        doc = int(document_ids[col]) if document_ids is not None else 0
        boundary = (
            previous_col is not None
            and (col != previous_col + 1 or doc != previous_doc)
        )
        if boundary and current:
            result.append(current)
            current = []
        current.append(col)
        previous_col = col
        previous_doc = doc
    if current:
        result.append(current)
    return result


class _V4CompressedContextAttention(nn.Module):
    """Stateless PyTorch reference for DeepSeek-V4-style CSA/HCA.

    The reference operates on full training segments and preserves packed-document
    isolation. It intentionally uses explicit per-query candidate construction;
    optimized kernels may replace it only after forward/backward parity.
    """

    def __init__(
        self,
        config: CompressedContextConfig,
        *,
        mode: Literal["csa", "hca"],
    ) -> None:
        super().__init__()
        if mode not in {"csa", "hca"}:
            raise ValueError("mode must be 'csa' or 'hca'")
        self.config = config
        self.mode = mode
        self.scale = config.head_dim**-0.5

        self.q_a_proj = nn.Linear(
            config.hidden_size,
            config.q_lora_rank,
            bias=config.projection_bias,
        )
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
            bias=config.projection_bias,
        )
        self.q_b_norm = UnweightedRMSNorm(config.rms_norm_eps)

        self.kv_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=config.projection_bias,
        )
        candidate_norm_dim = (
            config.head_dim - config.partial_rotary_dim
            if config.normalize_candidate_nonrotary_only
            else config.head_dim
        )
        if candidate_norm_dim <= 0:
            raise ValueError(
                "non-rotary candidate normalization requires head_dim > partial_rotary_dim"
            )
        self.kv_norm = HeadRMSNorm(candidate_norm_dim, config.rms_norm_eps)

        compressor_width = config.head_dim * (2 if mode == "csa" else 1)
        self.compressor_kv_proj = nn.Linear(
            config.hidden_size,
            compressor_width,
            bias=config.projection_bias,
        )
        self.compressor_gate_proj = nn.Linear(
            config.hidden_size,
            compressor_width,
            bias=False,
        )
        compress_rate = (
            config.csa_compress_rate if mode == "csa" else config.hca_compress_rate
        )
        self.compressor_position_bias = nn.Parameter(
            torch.zeros(compress_rate, compressor_width)
        )
        self.compressor_kv_norm = HeadRMSNorm(
            candidate_norm_dim,
            config.rms_norm_eps,
        )

        self.index_kv_proj: nn.Linear | None = None
        self.index_gate_proj: nn.Linear | None = None
        self.index_position_bias: nn.Parameter | None = None
        self.index_kv_norm: nn.Module | None = None
        self.index_q_proj: nn.Linear | None = None
        self.index_weight_proj: nn.Linear | None = None
        if mode == "csa":
            if config.direct_token_indexer:
                self.index_kv_proj = nn.Linear(
                    config.hidden_size,
                    config.index_head_dim,
                    bias=config.projection_bias,
                )
                self.index_kv_norm = nn.LayerNorm(
                    config.index_head_dim,
                    eps=config.rms_norm_eps,
                    elementwise_affine=True,
                )
            else:
                self.index_kv_proj = nn.Linear(
                    config.hidden_size,
                    2 * config.index_head_dim,
                    bias=config.projection_bias,
                )
                self.index_gate_proj = nn.Linear(
                    config.hidden_size,
                    2 * config.index_head_dim,
                    bias=config.projection_bias,
                )
                self.index_position_bias = nn.Parameter(
                    torch.zeros(
                        config.csa_compress_rate,
                        2 * config.index_head_dim,
                    )
                )
                self.index_kv_norm = HeadRMSNorm(
                    config.index_head_dim,
                    config.rms_norm_eps,
                )
            self.index_q_proj = nn.Linear(
                config.q_lora_rank,
                config.index_n_heads * config.index_head_dim,
                bias=config.projection_bias,
            )
            self.index_weight_proj = nn.Linear(
                config.hidden_size,
                config.index_n_heads,
                bias=config.projection_bias,
            )

        self.rotary = InterleavedRotaryEmbedding(
            config.partial_rotary_dim,
            config.max_position_embeddings,
            config.compress_rope_theta,
        )
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))
        self.output = GroupedLowRankOutput(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            groups=config.o_groups,
            rank=config.o_lora_rank,
            hidden_size=config.hidden_size,
            bias=config.projection_bias,
        )

    def _normalize_main_candidate(
        self,
        value: torch.Tensor,
        norm: HeadRMSNorm,
    ) -> torch.Tensor:
        if not self.config.normalize_candidate_nonrotary_only:
            return norm(value)
        split = self.config.head_dim - self.config.partial_rotary_dim
        nonrot = norm(value[..., :split])
        rotary = value[..., split:]
        return torch.cat((nonrot, rotary), dim=-1)

    @property
    def compress_rate(self) -> int:
        return (
            self.config.csa_compress_rate
            if self.mode == "csa"
            else self.config.hca_compress_rate
        )

    def _rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("RoPE input must have shape [tokens, heads, head_dim]")
        cos, sin = self.rotary.cos_sin(
            positions.unsqueeze(0),
            dtype=x.dtype,
            device=x.device,
        )
        tensor = x.transpose(0, 1).unsqueeze(0)
        if inverse:
            rotated = apply_inverse_partial_rotary_at_end(
                tensor,
                cos,
                sin,
                self.config.partial_rotary_dim,
            )
        else:
            rotated = apply_partial_rotary_at_end(
                tensor,
                cos,
                sin,
                self.config.partial_rotary_dim,
            )
        return rotated.squeeze(0).transpose(0, 1)

    def _compress_hca(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        rate = self.config.hca_compress_rate
        usable = (hidden.shape[0] // rate) * rate
        if usable == 0:
            return hidden.new_zeros((0, self.config.head_dim))
        hidden = hidden[:usable]
        windows = usable // rate
        kv = self.compressor_kv_proj(hidden).view(
            windows,
            rate,
            self.config.head_dim,
        )
        gate = self.compressor_gate_proj(hidden).view(
            windows,
            rate,
            self.config.head_dim,
        )
        gate = gate + self.compressor_position_bias
        weights = gate.softmax(dim=1, dtype=torch.float32).to(kv.dtype)
        compressed = self._normalize_main_candidate(
            (kv * weights).sum(dim=1),
            self.compressor_kv_norm,
        )
        representative = positions[:usable:rate]
        return self._rope(
            compressed.unsqueeze(1),
            representative,
        ).squeeze(1)

    def _compress_overlap(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_proj: nn.Linear,
        gate_proj: nn.Linear,
        position_bias: torch.Tensor,
        norm: HeadRMSNorm,
        head_dim: int,
    ) -> torch.Tensor:
        rate = self.config.csa_compress_rate
        usable = (hidden.shape[0] // rate) * rate
        if usable == 0:
            return hidden.new_zeros((0, head_dim))
        hidden = hidden[:usable]
        windows = usable // rate
        kv = kv_proj(hidden).view(windows, rate, 2 * head_dim)
        gate = gate_proj(hidden).view(windows, rate, 2 * head_dim)
        gate = gate + position_bias

        combined_kv = kv.new_zeros((windows, 2 * rate, head_dim))
        combined_gate = gate.new_full(
            (windows, 2 * rate, head_dim),
            float("-inf"),
        )
        combined_kv[:, rate:] = kv[..., head_dim:]
        combined_gate[:, rate:] = gate[..., head_dim:]
        if windows > 1:
            combined_kv[1:, :rate] = kv[:-1, :, :head_dim]
            combined_gate[1:, :rate] = gate[:-1, :, :head_dim]

        weights = combined_gate.softmax(dim=1, dtype=torch.float32).to(kv.dtype)
        compressed = (combined_kv * weights).sum(dim=1)
        if (
            head_dim == self.config.head_dim
            and self.config.normalize_candidate_nonrotary_only
        ):
            compressed = self._normalize_main_candidate(compressed, norm)
        else:
            compressed = norm(compressed)
        representative = positions[:usable:rate]
        return self._rope(
            compressed.unsqueeze(1),
            representative,
        ).squeeze(1)

    def _compress_main(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "hca":
            return self._compress_hca(hidden, positions)
        return self._compress_overlap(
            hidden,
            positions,
            kv_proj=self.compressor_kv_proj,
            gate_proj=self.compressor_gate_proj,
            position_bias=self.compressor_position_bias,
            norm=self.compressor_kv_norm,
            head_dim=self.config.head_dim,
        )

    def _index_scores(
        self,
        hidden: torch.Tensor,
        q_residual: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "csa":
            raise CompressedContextError("index selection is CSA-only")
        assert self.index_kv_proj is not None
        assert self.index_kv_norm is not None
        assert self.index_q_proj is not None
        assert self.index_weight_proj is not None

        length = hidden.shape[0]
        if self.config.direct_token_indexer:
            token_keys = self.index_kv_norm(self.index_kv_proj(hidden))
            compressed = self._rope(
                token_keys.unsqueeze(1),
                positions,
            ).squeeze(1)
        else:
            assert self.index_gate_proj is not None
            assert self.index_position_bias is not None
            assert isinstance(self.index_kv_norm, HeadRMSNorm)
            compressed = self._compress_overlap(
                hidden,
                positions,
                kv_proj=self.index_kv_proj,
                gate_proj=self.index_gate_proj,
                position_bias=self.index_position_bias,
                norm=self.index_kv_norm,
                head_dim=self.config.index_head_dim,
            )
        if compressed.shape[0] == 0:
            empty = hidden.new_zeros((length, 0), dtype=torch.float32)
            return empty, torch.zeros(
                (length, 0),
                dtype=torch.bool,
                device=hidden.device,
            )

        q = self.index_q_proj(q_residual).view(
            length,
            self.config.index_n_heads,
            self.config.index_head_dim,
        )
        q = self._rope(q, positions)
        scores = torch.einsum(
            "thd,sd->ths",
            q.float(),
            compressed.float(),
        )
        scores = F.relu(scores) * (self.config.index_head_dim**-0.5)
        weights = (
            self.index_weight_proj(hidden).float()
            * (self.config.index_n_heads**-0.5)
        )
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=1)

        entry = torch.arange(compressed.shape[0], device=hidden.device)
        if self.config.direct_token_indexer:
            query = torch.arange(length, device=hidden.device)
            valid_mask = entry.unsqueeze(0) <= query.unsqueeze(1)
        else:
            visible = (
                torch.arange(length, device=hidden.device) + 1
            ) // self.compress_rate
            valid_mask = entry.unsqueeze(0) < visible.unsqueeze(1)
        index_scores = index_scores.masked_fill(
            ~valid_mask,
            float("-inf"),
        )
        return index_scores, valid_mask

    def _index_selection(
        self,
        hidden: torch.Tensor,
        q_residual: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        index_scores, valid_mask = self._index_scores(
            hidden,
            q_residual,
            positions,
        )
        if index_scores.shape[-1] == 0:
            return torch.full(
                (hidden.shape[0], 0),
                -1,
                dtype=torch.long,
                device=hidden.device,
            )
        topk = min(self.config.index_topk, index_scores.shape[-1])
        selected = index_scores.topk(topk, dim=-1).indices
        selected_valid = valid_mask.gather(1, selected)
        return torch.where(
            selected_valid,
            selected,
            torch.full_like(selected, -1),
        )

    def indexer_scores(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> tuple[IndexerSegmentScores, ...]:
        if self.mode != "csa":
            raise CompressedContextError(
                "Lightning indexer scores are available only for CSA"
            )
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"indexer input must have shape [batch, sequence, {self.config.hidden_size}]"
            )
        batch, sequence, _ = x.shape
        prepared = prepare_causal_attention(
            batch_size=batch,
            sequence_length=sequence,
            device=x.device,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        docs = (
            document_ids.to(x.device)
            if document_ids is not None
            else None
        )
        result: list[IndexerSegmentScores] = []
        for row in range(batch):
            row_docs = docs[row] if docs is not None else None
            for cols in _segments(prepared.valid_tokens[row], row_docs):
                token_indices = torch.tensor(
                    cols,
                    dtype=torch.long,
                    device=x.device,
                )
                hidden = x[row].index_select(0, token_indices)
                positions = prepared.position_ids[row].index_select(
                    0,
                    token_indices,
                )
                q_residual = self.q_a_norm(self.q_a_proj(hidden))
                scores, valid_mask = self._index_scores(
                    hidden,
                    q_residual,
                    positions,
                )
                selected = self._index_selection(
                    hidden,
                    q_residual,
                    positions,
                )
                result.append(
                    IndexerSegmentScores(
                        batch_index=row,
                        token_indices=token_indices,
                        scores=scores,
                        valid_mask=valid_mask,
                        selected_indices=selected,
                    )
                )
        return tuple(result)

    def _attend_candidates(
        self,
        q: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if q.shape != (self.config.num_attention_heads, self.config.head_dim):
            raise ValueError("query shape mismatch")
        if candidates.ndim != 2 or candidates.shape[-1] != self.config.head_dim:
            raise ValueError("candidate shape mismatch")
        if candidates.shape[0] == 0:
            return torch.zeros_like(q)

        scores = torch.einsum("hd,nd->hn", q.float(), candidates.float())
        scores = scores * self.scale
        sink = self.sinks.float().unsqueeze(-1)
        max_score = torch.maximum(
            scores.max(dim=-1, keepdim=True).values,
            sink,
        )
        numerator = torch.exp(scores - max_score)
        denominator = (
            numerator.sum(dim=-1, keepdim=True)
            + torch.exp(sink - max_score)
        )
        weights = numerator / denominator.clamp_min(
            torch.finfo(numerator.dtype).tiny
        )
        if self.training and self.config.attention_dropout:
            weights = F.dropout(
                weights,
                p=self.config.attention_dropout,
                training=True,
            )
        return torch.einsum(
            "hn,nd->hd",
            weights.to(candidates.dtype),
            candidates,
        )

    def _index_keys(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Preproject keys once for streamed CSA indexing."""
        assert self.mode == "csa"
        assert self.index_kv_proj is not None and self.index_kv_norm is not None
        if self.config.direct_token_indexer:
            values = self.index_kv_norm(self.index_kv_proj(hidden))
            return self._rope(values.unsqueeze(1), positions).squeeze(1)
        assert self.index_gate_proj is not None
        assert self.index_position_bias is not None
        assert isinstance(self.index_kv_norm, HeadRMSNorm)
        return self._compress_overlap(
            hidden,
            positions,
            kv_proj=self.index_kv_proj,
            gate_proj=self.index_gate_proj,
            position_bias=self.index_position_bias,
            norm=self.index_kv_norm,
            head_dim=self.config.index_head_dim,
        )

    def _streaming_index_selection(
        self,
        query_hidden: torch.Tensor,
        query_residual: torch.Tensor,
        query_positions: torch.Tensor,
        query_offsets: torch.Tensor,
        index_keys: torch.Tensor,
        *,
        key_chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Exact full-key top-k, bounded to query_chunk x key_chunk scores.

        Excludes future/incomplete entries before top-k. This intentionally
        preserves the hard-selection, nondifferentiable index semantics.
        """
        assert self.mode == "csa"
        assert self.index_q_proj is not None and self.index_weight_proj is not None
        query_count = query_hidden.shape[0]
        total = index_keys.shape[0]
        if total == 0:
            return torch.empty((query_count, 0), dtype=torch.long, device=query_hidden.device)
        k = min(self.config.index_topk, total)
        q = self.index_q_proj(query_residual).view(
            query_count, self.config.index_n_heads, self.config.index_head_dim
        )
        q = self._rope(q, query_positions).float()
        w = self.index_weight_proj(query_hidden).float() * (
            self.config.index_n_heads ** -0.5
        )
        best_scores = q.new_empty((query_count, 0))
        best_indices = torch.empty((query_count, 0), dtype=torch.long, device=q.device)
        for lo in range(0, total, key_chunk_size):
            hi = min(total, lo + key_chunk_size)
            scores = torch.einsum("qhd,kd->qhk", q, index_keys[lo:hi].float())
            scores = F.relu(scores) * (self.config.index_head_dim ** -0.5)
            scores = (scores * w.unsqueeze(-1)).sum(dim=1)
            positions = torch.arange(lo, hi, device=q.device)
            if self.config.direct_token_indexer:
                valid = positions.unsqueeze(0) <= query_offsets.unsqueeze(1)
            else:
                visible = (query_offsets + 1) // self.config.csa_compress_rate
                valid = positions.unsqueeze(0) < visible.unsqueeze(1)
            scores = scores.masked_fill(~valid, float("-inf"))
            indices = positions.unsqueeze(0).expand(query_count, -1)
            candidates = torch.cat((best_scores, scores), dim=1)
            candidate_indices = torch.cat((best_indices, indices), dim=1)
            keep = min(k, candidates.shape[1])
            best_scores, selected = candidates.topk(keep, dim=1)
            best_indices = candidate_indices.gather(1, selected)
        return best_indices.masked_fill(~torch.isfinite(best_scores), -1)

    def _attend_chunk(
        self,
        q: torch.Tensor,
        local_kv: torch.Tensor,
        compressed: torch.Tensor,
        *,
        start: int,
        selected: torch.Tensor | None,
    ) -> torch.Tensor:
        """Vectorize all query heads and tokens in a bounded query chunk."""
        count = q.shape[0]
        offsets = torch.arange(start, start + count, device=q.device)
        window_start = max(0, start - self.config.sliding_window + 1)
        local = local_kv[window_start : start + count]
        local_offsets = torch.arange(window_start, start + count, device=q.device)
        local_valid = (
            (local_offsets.unsqueeze(0) <= offsets.unsqueeze(1))
            & (local_offsets.unsqueeze(0) + self.config.sliding_window > offsets.unsqueeze(1))
        )
        local_scores = torch.einsum("qhd,kd->qhk", q.float(), local.float()) * self.scale
        local_scores = local_scores.masked_fill(~local_valid.unsqueeze(1), float("-inf"))

        if self.mode == "hca":
            long_kv = compressed
            long_scores = torch.einsum("qhd,kd->qhk", q.float(), long_kv.float()) * self.scale
            visible = (offsets + 1) // self.config.hca_compress_rate
            long_valid = (
                torch.arange(long_kv.shape[0], device=q.device).unsqueeze(0)
                < visible.unsqueeze(1)
            )
        else:
            assert selected is not None
            source = local_kv if self.config.direct_token_indexer else compressed
            if selected.shape[1]:
                long_kv = source.index_select(0, selected.clamp_min(0).reshape(-1)).view(
                    count, selected.shape[1], self.config.head_dim
                )
                long_scores = torch.einsum("qhd,qkd->qhk", q.float(), long_kv.float()) * self.scale
            else:
                long_kv = q.new_empty((count, 0, self.config.head_dim))
                long_scores = q.new_empty((count, q.shape[1], 0), dtype=torch.float32)
            long_valid = selected >= 0
        long_scores = long_scores.masked_fill(~long_valid.unsqueeze(1), float("-inf"))
        sink = self.sinks.float().view(1, -1, 1)
        # Treat the sink as an extra softmax entry (without an associated value).
        max_score = torch.maximum(
            torch.maximum(local_scores.max(dim=-1, keepdim=True).values,
                          long_scores.max(dim=-1, keepdim=True).values
                          if long_scores.shape[-1] else sink),
            sink,
        )
        local_exp = torch.exp(local_scores - max_score)
        long_exp = torch.exp(long_scores - max_score)
        denom = (
            local_exp.sum(dim=-1, keepdim=True)
            + long_exp.sum(dim=-1, keepdim=True)
            + torch.exp(sink - max_score)
        ).clamp_min(torch.finfo(local_exp.dtype).tiny)
        local_weight = local_exp / denom
        long_weight = long_exp / denom
        if self.training and self.config.attention_dropout:
            local_weight = F.dropout(local_weight, p=self.config.attention_dropout, training=True)
            long_weight = F.dropout(long_weight, p=self.config.attention_dropout, training=True)
        local_out = torch.einsum("qhk,kd->qhd", local_weight.to(local.dtype), local)
        if self.mode == "hca":
            long_out = torch.einsum("qhk,kd->qhd", long_weight.to(long_kv.dtype), long_kv)
        else:
            long_out = torch.einsum("qhk,qkd->qhd", long_weight.to(long_kv.dtype), long_kv)
        return local_out + long_out

    def _forward_segment(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Bound peak attention/index scores; remove per-token Python loop."""
        length = hidden.shape[0]
        q_residual = self.q_a_norm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_residual).view(
            length, self.config.num_attention_heads, self.config.head_dim
        )
        if self.config.normalize_query_output:
            q = self.q_b_norm(q)
        q = self._rope(q, positions)
        local_kv = self._normalize_main_candidate(self.kv_proj(hidden), self.kv_norm)
        local_kv = self._rope(local_kv.unsqueeze(1), positions).squeeze(1)
        compressed = self._compress_main(hidden, positions)
        index_keys = self._index_keys(hidden, positions) if self.mode == "csa" else None

        pieces: list[torch.Tensor] = []
        # 32 queries x 1024 index keys bounds the largest selection workspace.
        for start in range(0, length, 32):
            end = min(start + 32, length)
            selected = (
                self._streaming_index_selection(
                    hidden[start:end], q_residual[start:end], positions[start:end],
                    torch.arange(start, end, device=hidden.device), index_keys,
                )
                if index_keys is not None else None
            )
            pieces.append(self._attend_chunk(
                q[start:end], local_kv, compressed, start=start, selected=selected,
            ))
        attended = torch.cat(pieces, dim=0)
        attended = self._rope(attended, positions, inverse=True)
        return self.output(attended)

    def new_cache(self, *, device: torch.device, dtype: torch.dtype) -> CompressedAttentionCache:
        """Allocate an empty inference cache for one sequence/document and layer."""
        if self.training:
            raise ValueError("set eval() before using compressed-attention inference cache")
        return CompressedAttentionCache(
            length=0,
            hidden_tail=torch.empty((0, self.config.hidden_size), device=device, dtype=dtype),
            positions_tail=torch.empty((0,), device=device, dtype=torch.long),
            local_kv=torch.empty((0, self.config.head_dim), device=device, dtype=dtype),
            compressed=torch.empty((16, self.config.head_dim), device=device, dtype=dtype),
            compressed_count=0,
            index_keys=torch.empty((16, self.config.index_head_dim), device=device, dtype=dtype),
            index_count=0,
        )

    @torch.no_grad()
    def forward_step(
        self,
        x: torch.Tensor,
        cache: CompressedAttentionCache,
        *,
        position: int | None = None,
    ) -> torch.Tensor:
        """Decode one token while incrementally updating compressed/index KVs.

        Returns [1,1,hidden]. Not yet wired into IQHybridForCausalLM.decode().
        Each layer and independent document must have its own cache.
        """
        if self.training or x.shape != (1, 1, self.config.hidden_size):
            raise ValueError("forward_step requires eval() and [1,1,hidden] input")
        if x.device != cache.hidden_tail.device or x.dtype != cache.hidden_tail.dtype:
            raise ValueError("cache device/dtype mismatch")
        token = x[0]
        offset = cache.length
        pos = offset if position is None else int(position)
        if pos < 0 or pos >= self.config.max_position_embeddings:
            raise ValueError("position is outside compressed-attention RoPE range")
        pos_t = torch.tensor([pos], device=x.device, dtype=torch.long)
        q_residual = self.q_a_norm(self.q_a_proj(token))
        q = self.q_b_proj(q_residual).view(
            1, self.config.num_attention_heads, self.config.head_dim
        )
        if self.config.normalize_query_output:
            q = self.q_b_norm(q)
        q = self._rope(q, pos_t)
        local_new = self._normalize_main_candidate(self.kv_proj(token), self.kv_norm)
        local_new = self._rope(local_new.unsqueeze(1), pos_t).squeeze(1)
        cache.local_kv = torch.cat((cache.local_kv, local_new), dim=0)[
            -self.config.sliding_window:
        ]
        tail_capacity = 2 * self.compress_rate
        cache.hidden_tail = torch.cat((cache.hidden_tail, token), dim=0)[-tail_capacity:]
        cache.positions_tail = torch.cat((cache.positions_tail, pos_t), dim=0)[-tail_capacity:]
        cache.length += 1
        if self.mode == "csa" and self.config.direct_token_indexer:
            new_idx = self._index_keys(token, pos_t)[0]
            cache.index_keys = _append_cache_row(cache.index_keys, cache.index_count, new_idx)
            cache.index_count += 1
        if cache.length % self.compress_rate == 0:
            new_compressed = self._compress_main(
                cache.hidden_tail, cache.positions_tail
            )[-1]
            cache.compressed = _append_cache_row(
                cache.compressed, cache.compressed_count, new_compressed
            )
            cache.compressed_count += 1
            if self.mode == "csa" and not self.config.direct_token_indexer:
                new_idx = self._index_keys(cache.hidden_tail, cache.positions_tail)[-1]
                cache.index_keys = _append_cache_row(cache.index_keys, cache.index_count, new_idx)
                cache.index_count += 1

        history = cache.compressed[:cache.compressed_count]
        if self.mode == "csa":
            selected = self._streaming_index_selection(
                token, q_residual, pos_t,
                torch.tensor([offset], device=x.device),
                cache.index_keys[:cache.index_count],
            )[0]
            source = cache.index_keys  # selection is only an index, not attention value
            if self.config.direct_token_indexer:
                # CSA direct index selection attends to historical local KVs.
                # Without retaining those values the cache cannot decode correctly.
                if not hasattr(cache, "direct_kv"):
                    cache.direct_kv = cache.local_kv.new_empty((16, self.config.head_dim))
                    cache.direct_count = 0
                cache.direct_kv = _append_cache_row(
                    cache.direct_kv, cache.direct_count, local_new[0]
                )
                cache.direct_count += 1
                source = cache.direct_kv[:cache.direct_count]
            else:
                source = history
            picked = source.index_select(0, selected[selected >= 0])
        else:
            picked = history
        candidates = torch.cat((cache.local_kv, picked), dim=0)
        attended = self._attend_candidates(q[0], candidates).unsqueeze(0)
        attended = self._rope(attended, pos_t, inverse=True)
        return self.output(attended).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"compressed attention input must have shape [batch, sequence, {self.config.hidden_size}]"
            )
        batch, sequence, _ = x.shape
        prepared = prepare_causal_attention(
            batch_size=batch,
            sequence_length=sequence,
            device=x.device,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        docs = (
            document_ids.to(x.device)
            if document_ids is not None
            else None
        )
        output = torch.zeros_like(x)
        for row in range(batch):
            row_docs = docs[row] if docs is not None else None
            for cols in _segments(prepared.valid_tokens[row], row_docs):
                index = torch.tensor(cols, dtype=torch.long, device=x.device)
                segment = x[row].index_select(0, index)
                positions = prepared.position_ids[row].index_select(0, index)
                segment_output = self._forward_segment(segment, positions)
                output[row].index_copy_(0, index, segment_output)
        if not bool(torch.isfinite(output).all()):
            raise CompressedContextError(
                f"{self.mode.upper()} produced non-finite output"
            )
        return output


class CompressedSparseContextAttention(_V4CompressedContextAttention):
    def __init__(self, config: CompressedContextConfig) -> None:
        super().__init__(config, mode="csa")


class HeavilyCompressedContextAttention(_V4CompressedContextAttention):
    def __init__(self, config: CompressedContextConfig) -> None:
        super().__init__(config, mode="hca")
