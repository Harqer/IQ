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



class DeepseekV4HCACache:
    """DeepSeek-V4 HCA cache state, adapted to IQ's shared-KV attention.

    The buffer, compressed-entry and entry-count contracts follow the upstream
    DeepseekV4HCACache. IQ owns its local KV tensor rather than subclassing
    Transformers' DynamicSlidingWindowLayer.
    """

    def __init__(self, config: CompressedContextConfig) -> None:
        self.compress_rate = config.hca_compress_rate
        self.sliding_window = config.sliding_window
        self.keys: torch.Tensor | None = None
        self.cumulative_length = 0
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        if key_states.shape != value_states.shape:
            raise ValueError("shared-KV update requires matching K/V shapes")
        full = key_states if self.keys is None else torch.cat([self.keys, key_states], dim=1)
        self.keys = full[:, -self.sliding_window + 1 :] if self.sliding_window > 1 else full[:, :0]
        self.cumulative_length += key_states.shape[1]
        return full

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        previous = self.compressed_kv[name]
        if previous is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1]:
            self.compressed_kv[name] = torch.cat([previous, compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        assert self.compressed_kv[name] is not None
        return self.compressed_kv[name]

    def reset(self) -> None:
        self.keys = None
        self.cumulative_length = 0
        for name in self.compressed_kv:
            self.buffer_kv[name] = None
            self.buffer_gate[name] = None
            self.compressed_kv[name] = None
            self.entry_count[name] = 0


class DeepseekV4CSACache(DeepseekV4HCACache):
    """CSA extends HCA's state with an indexer and Ca overlap per stream."""

    def __init__(self, config: CompressedContextConfig) -> None:
        super().__init__(config)
        self.compress_rate = config.csa_compress_rate
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate

    def reset(self) -> None:
        super().reset()
        for name in self.overlap_kv:
            self.overlap_kv[name] = self.overlap_gate[name] = None


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
        self.kv_norm = HeadRMSNorm(config.head_dim, config.rms_norm_eps)

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
            config.head_dim,
            config.rms_norm_eps,
        )

        self.index_kv_proj: nn.Linear | None = None
        self.index_gate_proj: nn.Linear | None = None
        self.index_position_bias: nn.Parameter | None = None
        self.index_kv_norm: HeadRMSNorm | None = None
        self.index_q_proj: nn.Linear | None = None
        self.index_weight_proj: nn.Linear | None = None
        if mode == "csa":
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
        compressed = self.compressor_kv_norm((kv * weights).sum(dim=1))
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
        compressed = norm((combined_kv * weights).sum(dim=1))
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
        assert self.index_gate_proj is not None
        assert self.index_position_bias is not None
        assert self.index_kv_norm is not None
        assert self.index_q_proj is not None
        assert self.index_weight_proj is not None

        compressed = self._compress_overlap(
            hidden,
            positions,
            kv_proj=self.index_kv_proj,
            gate_proj=self.index_gate_proj,
            position_bias=self.index_position_bias,
            norm=self.index_kv_norm,
            head_dim=self.config.index_head_dim,
        )
        length = hidden.shape[0]
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

        visible = (torch.arange(length, device=hidden.device) + 1) // self.compress_rate
        entry = torch.arange(compressed.shape[0], device=hidden.device)
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

    def _forward_segment(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        length = hidden.shape[0]
        q_residual = self.q_a_norm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_residual).view(
            length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        q = self.q_b_norm(q)
        q = self._rope(q, positions)

        local_kv = self.kv_norm(self.kv_proj(hidden))
        local_kv = self._rope(
            local_kv.unsqueeze(1),
            positions,
        ).squeeze(1)
        compressed = self._compress_main(hidden, positions)
        selected = (
            self._index_selection(hidden, q_residual, positions)
            if self.mode == "csa"
            else None
        )

        outputs: list[torch.Tensor] = []
        for token in range(length):
            start = max(0, token - self.config.sliding_window + 1)
            local = local_kv[start : token + 1]

            if self.mode == "hca":
                visible_count = (token + 1) // self.config.hca_compress_rate
                long_range = compressed[:visible_count]
            else:
                assert selected is not None
                indices = selected[token]
                valid = indices[indices >= 0]
                long_range = (
                    compressed.index_select(0, valid)
                    if valid.numel()
                    else compressed[:0]
                )

            candidates = (
                torch.cat([local, long_range], dim=0)
                if long_range.shape[0]
                else local
            )
            outputs.append(self._attend_candidates(q[token], candidates))

        attended = torch.stack(outputs, dim=0)
        attended = self._rope(attended, positions, inverse=True)
        return self.output(attended)


    def _cache_compress(
        self,
        cache: DeepseekV4HCACache,
        name: str,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if name == "compressor":
            kv_proj, gate_proj = self.compressor_kv_proj, self.compressor_gate_proj
            bias, norm, width = (
                self.compressor_position_bias,
                self.compressor_kv_norm,
                self.config.head_dim,
            )
        else:
            assert self.mode == "csa"
            assert self.index_kv_proj is not None
            assert self.index_gate_proj is not None
            assert self.index_position_bias is not None
            assert self.index_kv_norm is not None
            kv_proj, gate_proj = self.index_kv_proj, self.index_gate_proj
            bias, norm, width = self.index_position_bias, self.index_kv_norm, self.config.index_head_dim

        kv = kv_proj(hidden)
        gate = gate_proj(hidden)
        chunk_kv, chunk_gate, first_position = cache.store_compression_weights(name, kv, gate)
        batch = hidden.shape[0]
        ratio = cache.compress_rate
        if chunk_kv.shape[1]:
            n_windows = chunk_kv.shape[1] // ratio
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + bias
            if self.mode == "hca":
                weights = chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
                compressed = norm((chunk_kv * weights).sum(dim=2))
            else:
                assert isinstance(cache, DeepseekV4CSACache)
                new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, width))
                new_gate = chunk_gate.new_full(
                    (batch, n_windows, 2 * ratio, width), float("-inf")
                )
                new_kv[:, :, ratio:] = chunk_kv[..., width:]
                new_gate[:, :, ratio:] = chunk_gate[..., width:]
                if n_windows > 1:
                    new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :width]
                    new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :width]
                prior_kv, prior_gate = cache.update_overlap_state(
                    name, chunk_kv, chunk_gate, width
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv
                    new_gate[:, 0, :ratio] = prior_gate
                weights = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
                compressed = norm((new_kv * weights).sum(dim=2))
            positions = (
                torch.arange(n_windows, device=hidden.device) * ratio + first_position
            ).repeat(batch)
            compressed = self._rope(
                compressed.reshape(batch * n_windows, 1, width), positions
            ).reshape(batch, n_windows, width)
        else:
            compressed = hidden.new_zeros((batch, 0, width))
        return cache.update_compressor_states(name, compressed)

    def _forward_cached(
        self,
        x: torch.Tensor,
        cache: DeepseekV4HCACache,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "csa" and not isinstance(cache, DeepseekV4CSACache):
            raise ValueError("CSA needs a DeepseekV4CSACache")
        if self.mode == "hca" and type(cache) is not DeepseekV4HCACache:
            raise ValueError("HCA needs a DeepseekV4HCACache")
        batch, sequence, _ = x.shape
        if self.training:
            raise ValueError("incremental compressed attention requires eval mode")
        if document_ids is not None:
            raise ValueError("cached generation does not support packed document_ids")
        if attention_mask is not None and not bool(attention_mask.to(torch.bool).all()):
            raise ValueError("cached generation requires unpadded new tokens")
        if cache.keys is not None and cache.keys.shape[0] != batch:
            raise ValueError("cache batch size changed without reset")
        expected = torch.arange(
            cache.cumulative_length,
            cache.cumulative_length + sequence,
            device=x.device,
        ).expand(batch, -1)
        if position_ids is not None and not torch.equal(position_ids.to(x.device), expected):
            raise ValueError("cached position_ids must follow the uninterrupted token offset")
        outputs: list[torch.Tensor] = []
        for token in range(sequence):
            hidden = x[:, token : token + 1, :]
            positions = expected[:, token]
            residual = self.q_a_norm(self.q_a_proj(hidden))
            q = self.q_b_norm(
                self.q_b_proj(residual).view(
                    batch, self.config.num_attention_heads, self.config.head_dim
                )
            )
            q = self._rope(q, positions)
            local = self.kv_norm(self.kv_proj(hidden))
            local = self._rope(local, positions).unsqueeze(1)
            local = cache.update(local, local)
            compressed = self._cache_compress(cache, "compressor", hidden)
            index_kv = (
                self._cache_compress(cache, "indexer", hidden)
                if self.mode == "csa"
                else None
            )
            attended: list[torch.Tensor] = []
            for row in range(batch):
                long_range = compressed[row]
                if index_kv is not None:
                    assert self.index_q_proj is not None
                    assert self.index_weight_proj is not None
                    index_query = self.index_q_proj(residual[row]).view(
                        self.config.index_n_heads, self.config.index_head_dim
                    )
                    index_query = self._rope(index_query.unsqueeze(0), positions[row : row + 1]).squeeze(0)
                    scores = torch.einsum(
                        "hd,sd->hs", index_query.float(), index_kv[row].float()
                    )
                    scores = F.relu(scores) * (self.config.index_head_dim**-0.5)
                    head_weights = (
                        self.index_weight_proj(hidden[row]).float().squeeze(0)
                        * (self.config.index_n_heads**-0.5)
                    )
                    scores = (scores * head_weights.unsqueeze(-1)).sum(dim=0)
                    if scores.numel():
                        indices = scores.topk(min(self.config.index_topk, scores.numel())).indices
                        long_range = long_range.index_select(0, indices)
                    else:
                        long_range = long_range[:0]
                candidates = (
                    torch.cat([local[row], long_range], dim=0)
                    if long_range.shape[0]
                    else local[row]
                )
                attended.append(self._attend_candidates(q[row], candidates))
            inverse = self._rope(torch.stack(attended, dim=0), positions, inverse=True)
            outputs.append(self.output(inverse))
        return torch.stack(outputs, dim=1) if outputs else x.new_zeros(x.shape)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_values: DeepseekV4HCACache | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"compressed attention input must have shape [batch, sequence, {self.config.hidden_size}]"
            )
        if past_key_values is not None:
            return self._forward_cached(
                x, past_key_values, position_ids, attention_mask, document_ids
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
