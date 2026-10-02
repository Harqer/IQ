from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm
from .state import Mamba3MIMOConfig, Mamba3MIMOState


class MultimodalError(RuntimeError):
    pass


@dataclass(frozen=True)
class IQMultimodalConfig:
    """Production multimodal contract for IQ.

    Vamba-style separation keeps the large visual stream out of quadratic
    self-attention. TimeViper-style transfer progressively moves visual
    information into text states and compresses visual memory after fusion.
    """
    vision_model_name: str
    fusion_layers: tuple[int, ...]
    transv_layers: tuple[int, ...]
    vision_backend: str = "auto"
    visual_mamba_layers: int = 1
    tokens_per_video_frame: int = 16
    cross_attention_heads: int = 8
    projector_hidden_multiplier: float = 2.0
    transv_shallow_keep_ratio: float = 0.5
    transv_deep_keep_ratio: float = 0.1
    min_visual_tokens: int = 16
    max_frames: int = 16384
    temporal_dilations: tuple[int, ...] = (1, 2, 4)
    use_bidirectional_video: bool = True
    query_projector_tokens: int = 64
    freeze_vision_tower: bool = True
    drop_cls_token: bool = True

    def __post_init__(self) -> None:
        if not self.vision_model_name.strip():
            raise ValueError("vision_model_name must be non-empty")
        if self.vision_backend not in {"auto", "glm5_next"}:
            raise ValueError("vision_backend must be 'auto' or 'glm5_next'")
        if self.visual_mamba_layers <= 0 or self.cross_attention_heads <= 0:
            raise ValueError("visual_mamba_layers and cross_attention_heads must be positive")
        if self.tokens_per_video_frame <= 0:
            raise ValueError("tokens_per_video_frame must be positive")
        if self.projector_hidden_multiplier <= 0:
            raise ValueError("projector_hidden_multiplier must be positive")
        for name, ratio in (
            ("transv_shallow_keep_ratio", self.transv_shallow_keep_ratio),
            ("transv_deep_keep_ratio", self.transv_deep_keep_ratio),
        ):
            if not (0.0 < ratio <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")
        if self.min_visual_tokens <= 0 or self.max_frames <= 0:
            raise ValueError("min_visual_tokens and max_frames must be positive")
        if self.query_projector_tokens <= 0:
            raise ValueError("query_projector_tokens must be positive")
        if not self.temporal_dilations or any(int(x) <= 0 for x in self.temporal_dilations):
            raise ValueError("temporal_dilations must contain positive strides")
        if tuple(sorted(set(self.fusion_layers))) != self.fusion_layers:
            raise ValueError("fusion_layers must be sorted and unique")
        if tuple(sorted(set(self.transv_layers))) != self.transv_layers:
            raise ValueError("transv_layers must be sorted and unique")
        if not set(self.transv_layers).issubset(self.fusion_layers):
            raise ValueError("every TransV layer must also be a fusion layer")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["fusion_layers"] = list(self.fusion_layers)
        data["transv_layers"] = list(self.transv_layers)
        data["temporal_dilations"] = list(self.temporal_dilations)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "IQMultimodalConfig":
        payload = dict(data)
        payload["fusion_layers"] = tuple(int(x) for x in payload.get("fusion_layers", ()))
        payload["transv_layers"] = tuple(int(x) for x in payload.get("transv_layers", ()))
        payload["temporal_dilations"] = tuple(int(x) for x in payload.get("temporal_dilations", (1, 2, 4)))
        return cls(**payload)


@dataclass
class VisualMemory:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    frame_ids: torch.Tensor
    source: str


def _sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = max(1, width // 2)
    scale = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    emb = torch.cat((torch.sin(pos * scale), torch.cos(pos * scale)), dim=1)[:, :width]
    if emb.shape[1] < width:
        emb = F.pad(emb, (0, width - emb.shape[1]))
    return emb.to(dtype=dtype)


class VisionTower(nn.Module):
    def __init__(self, config: IQMultimodalConfig, hidden_size: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.is_glm5 = config.vision_backend == "glm5_next"
        try:
            if self.is_glm5:
                from transformers import AutoConfig, Glm5NextVisionModel
                composite = AutoConfig.from_pretrained(config.vision_model_name)
                vision_config = getattr(composite, "vision_config", None)
                if vision_config is None:
                    raise MultimodalError("GLM-5.3 donor config has no vision_config")
                self.encoder = Glm5NextVisionModel.from_pretrained(
                    config.vision_model_name,
                    config=vision_config,
                    dtype=dtype,
                ).to(device=device)
                vision_width = getattr(vision_config, "out_hidden_size", None)
            else:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained(
                    config.vision_model_name,
                    dtype=dtype,
                ).to(device=device)
                vision_width = getattr(self.encoder.config, "hidden_size", None)
                if vision_width is None:
                    vision_width = getattr(self.encoder.config, "vision_embed_dim", None)
        except ImportError as exc:
            raise MultimodalError("Transformers with the configured vision backend is required") from exc
        if config.freeze_vision_tower:
            self.encoder.requires_grad_(False)
        if not isinstance(vision_width, int) or vision_width <= 0:
            raise MultimodalError("vision encoder does not expose a usable output width")
        mid = max(hidden_size, int(hidden_size * config.projector_hidden_multiplier))
        self.projector = nn.Sequential(
            nn.Linear(vision_width, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, hidden_size, bias=False),
            RMSNorm(hidden_size),
        ).to(device=device, dtype=dtype)

    def _generic_frames(self, frames: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=frames)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3:
            raise MultimodalError("generic vision encoder must return rank-3 last_hidden_state")
        if self.config.drop_cls_token and hidden.shape[1] > 1:
            hidden = hidden[:, 1:]
        return self.projector(hidden)

    def _glm_features(
        self,
        values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        video: bool,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise MultimodalError("GLM-5.3 vision requires grid_thw with shape [items, 3]")
        if video:
            t = grid_thw[:, 0]
            hw = grid_thw[:, 1:]
            flat_hw = torch.repeat_interleave(hw, t, dim=0)
            ones = grid_thw.new_ones(flat_hw.shape[0], 1)
            encoder_grid = torch.cat((ones, flat_hw), dim=1)
        else:
            encoder_grid = grid_thw
        output = self.encoder(values.to(dtype=self.encoder.dtype), grid_thw=encoder_grid)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None or pooled.ndim != 2:
            raise MultimodalError("GLM-5.3 vision tower did not return pooler_output")
        merge = int(self.encoder.spatial_merge_size)
        split_sizes = (grid_thw.prod(-1) // (merge * merge)).tolist()
        if sum(split_sizes) != pooled.shape[0]:
            raise MultimodalError("GLM-5.3 grid metadata does not match visual feature count")
        return self.projector(pooled), [int(x) for x in split_sizes], grid_thw

    @staticmethod
    def _pad_items(
        features: torch.Tensor,
        split_sizes: list[int],
        grid_thw: torch.Tensor,
        *,
        source: str,
    ) -> VisualMemory:
        batch = len(split_sizes)
        width = features.shape[-1]
        max_tokens = max(split_sizes)
        hidden = features.new_zeros((batch, max_tokens, width))
        mask = torch.zeros((batch, max_tokens), dtype=torch.bool, device=features.device)
        frame_ids = torch.zeros((batch, max_tokens), dtype=torch.long, device=features.device)
        cursor = 0
        for row, count in enumerate(split_sizes):
            item = features[cursor : cursor + count]
            hidden[row, :count] = item
            mask[row, :count] = True
            if source == "video":
                frames = max(1, int(grid_thw[row, 0]))
                per_frame = max(1, count // frames)
                ids = torch.arange(frames, device=features.device).repeat_interleave(per_frame)
                if ids.numel() < count:
                    ids = F.pad(ids, (0, count - ids.numel()), value=frames - 1)
                frame_ids[row, :count] = ids[:count]
            cursor += count
        return VisualMemory(hidden, mask, frame_ids, source)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None,
        video_values: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> VisualMemory | None:
        if pixel_values is not None and video_values is not None:
            raise MultimodalError("provide pixel_values or video_values, not both")
        if pixel_values is None and video_values is None:
            if frame_mask is not None or image_grid_thw is not None or video_grid_thw is not None:
                raise MultimodalError("visual masks/grid metadata require visual values")
            return None

        if self.is_glm5:
            if frame_mask is not None:
                raise MultimodalError("GLM-5.3 processed video uses video_grid_thw, not frame_mask")
            if pixel_values is not None:
                features, sizes, grid = self._glm_features(
                    pixel_values, image_grid_thw, video=False
                )
                return self._pad_items(features, sizes, grid, source="image")
            assert video_values is not None
            features, sizes, grid = self._glm_features(
                video_values, video_grid_thw, video=True
            )
            if any(int(x) > self.config.max_frames for x in grid[:, 0].tolist()):
                raise MultimodalError("processed video exceeds configured max_frames")
            return self._pad_items(features, sizes, grid, source="video")

        if image_grid_thw is not None or video_grid_thw is not None:
            raise MultimodalError("grid_thw metadata is only valid for the glm5_next vision backend")
        if pixel_values is not None:
            if pixel_values.ndim != 4:
                raise MultimodalError("pixel_values must have shape [batch, channels, height, width]")
            projected = self._generic_frames(pixel_values)
            mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
            frame_ids = torch.zeros(projected.shape[:2], dtype=torch.long, device=projected.device)
            return VisualMemory(projected, mask, frame_ids, "image")

        assert video_values is not None
        if video_values.ndim != 5:
            raise MultimodalError("video_values must have shape [batch, frames, channels, height, width]")
        batch, frames = video_values.shape[:2]
        if frames > self.config.max_frames:
            raise MultimodalError(f"video has {frames} frames; configured maximum is {self.config.max_frames}")
        valid_frames = (
            torch.ones((batch, frames), dtype=torch.bool, device=video_values.device)
            if frame_mask is None
            else frame_mask.to(device=video_values.device, dtype=torch.bool)
        )
        if valid_frames.shape != (batch, frames):
            raise MultimodalError(f"frame_mask must have shape {(batch, frames)}")
        if bool((valid_frames.sum(dim=1) == 0).any()):
            raise MultimodalError("every video row must contain at least one valid frame")
        flat = video_values.reshape(batch * frames, *video_values.shape[2:])
        projected = self._generic_frames(flat)
        patches = projected.shape[1]
        projected = projected.view(batch, frames, patches, -1)
        temporal = _sinusoidal_positions(frames, projected.shape[-1], projected.device, projected.dtype)
        projected = projected + temporal.view(1, frames, 1, -1)
        tokens = projected.reshape(batch, frames * patches, -1)
        mask = valid_frames.unsqueeze(-1).expand(batch, frames, patches).reshape(batch, frames * patches)
        frame_ids = torch.arange(frames, device=tokens.device).view(1, frames, 1)
        frame_ids = frame_ids.expand(batch, frames, patches).reshape(batch, frames * patches)
        return VisualMemory(tokens, mask, frame_ids, "video")

def _tome_merge(tokens: torch.Tensor, target: int) -> torch.Tensor:
    """Training-free ToMe-style bipartite similarity merging to a target count."""
    if tokens.ndim != 2 or target <= 0:
        raise MultimodalError("ToMe expects [tokens, hidden] and a positive target")
    x = tokens
    while x.shape[0] > target:
        n = x.shape[0]
        a = x[0::2]
        b = x[1::2]
        if b.shape[0] == 0:
            break
        metric_a = F.normalize(a.float(), dim=-1)
        metric_b = F.normalize(b.float(), dim=-1)
        similarity = metric_a @ metric_b.transpose(0, 1)
        best_score, best_dst = similarity.max(dim=-1)
        max_merges = min(n - target, a.shape[0])
        merge_src = torch.topk(best_score, k=max_merges, sorted=False).indices
        merge_mask = torch.zeros(a.shape[0], dtype=torch.bool, device=x.device)
        merge_mask[merge_src] = True
        dst = best_dst[merge_src]

        # Average merged source tokens into their matched destination tokens.
        b_new = b.clone()
        counts = torch.ones((b.shape[0], 1), dtype=x.dtype, device=x.device)
        b_new.index_add_(0, dst, a[merge_src])
        counts.index_add_(
            0,
            dst,
            torch.ones((dst.numel(), 1), dtype=x.dtype, device=x.device),
        )
        b_new = b_new / counts
        kept_a = a[~merge_mask]
        x = torch.cat((kept_a, b_new), dim=0)
    return x[:target]


def _merge_video_frames(memory: VisualMemory, target_per_frame: int) -> VisualMemory:
    if memory.source != "video":
        return memory
    rows: list[torch.Tensor] = []
    row_frames: list[torch.Tensor] = []
    for row in range(memory.hidden_states.shape[0]):
        valid = memory.attention_mask[row]
        src = memory.hidden_states[row, valid]
        frames = memory.frame_ids[row, valid]
        parts: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        for frame in torch.unique_consecutive(frames):
            frame_tokens = src[frames == frame]
            merged = _tome_merge(
                frame_tokens,
                min(target_per_frame, frame_tokens.shape[0]),
            )
            parts.append(merged)
            ids.append(torch.full(
                (merged.shape[0],), int(frame), dtype=torch.long, device=src.device
            ))
        rows.append(torch.cat(parts, dim=0))
        row_frames.append(torch.cat(ids, dim=0))
    max_tokens = max(x.shape[0] for x in rows)
    hidden = memory.hidden_states.new_zeros(
        (len(rows), max_tokens, memory.hidden_states.shape[-1])
    )
    mask = torch.zeros((len(rows), max_tokens), dtype=torch.bool, device=hidden.device)
    frame_ids = torch.zeros((len(rows), max_tokens), dtype=torch.long, device=hidden.device)
    for row, values in enumerate(rows):
        count = values.shape[0]
        hidden[row, :count] = values
        mask[row, :count] = True
        frame_ids[row, :count] = row_frames[row]
    return VisualMemory(hidden, mask, frame_ids, memory.source)


class VisualMambaEncoder(nn.Module):
    """Linear-complexity visual-token processor; no visual self-attention."""
    def __init__(
        self,
        hidden_size: int,
        mamba_config: Mamba3MIMOConfig,
        num_layers: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.norms = nn.ModuleList(
            [RMSNorm(hidden_size).to(device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.layers = nn.ModuleList(
            [
                Mamba3MIMOState(mamba_config, layer_idx=i, dtype=dtype, device=device)
                for i in range(num_layers)
            ]
        )

    def forward(self, memory: VisualMemory) -> VisualMemory:
        x = memory.hidden_states
        batch, tokens, _ = x.shape
        valid = memory.attention_mask
        # Pack valid visual tokens so padded frames never alter recurrent state.
        flat_parts: list[torch.Tensor] = []
        lengths: list[int] = []
        for row in range(batch):
            part = x[row, valid[row]]
            flat_parts.append(part)
            lengths.append(part.shape[0])
        packed = torch.cat(flat_parts, dim=0).unsqueeze(0)
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu = torch.tensor(boundaries, dtype=torch.int32, device=x.device)
        for norm, layer in zip(self.norms, self.layers, strict=True):
            packed = packed + layer(norm(packed), cu_seqlens=cu)
        output = torch.zeros_like(x)
        cursor = 0
        for row, length in enumerate(lengths):
            output[row, valid[row]] = packed[0, cursor : cursor + length]
            cursor += length
        return VisualMemory(output, valid, memory.frame_ids, memory.source)


class QueryConditionedProjector(nn.Module):
    """Q-Mamba-style query compression without introducing another SSM."""

    def __init__(self, hidden_size: int, heads: int, num_queries: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_size, device=device, dtype=dtype))
        nn.init.normal_(self.queries, std=hidden_size ** -0.5)
        self.query_norm = RMSNorm(hidden_size).to(device=device, dtype=dtype)
        self.visual_norm = RMSNorm(hidden_size).to(device=device, dtype=dtype)
        self.attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True, device=device, dtype=dtype)

    def forward(self, memory: VisualMemory, text: torch.Tensor, text_mask: torch.Tensor | None) -> VisualMemory:
        batch = text.shape[0]
        if memory.hidden_states.shape[0] != batch:
            raise MultimodalError("query projection requires one visual memory row per text row")
        if text_mask is None:
            instruction = text.mean(dim=1)
        else:
            mask = text_mask.to(text.dtype).unsqueeze(-1)
            instruction = (text * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1) + instruction.unsqueeze(1)
        projected, _ = self.attn(
            self.query_norm(queries),
            self.visual_norm(memory.hidden_states),
            self.visual_norm(memory.hidden_states),
            key_padding_mask=~memory.attention_mask,
            need_weights=False,
        )
        mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
        frames = torch.zeros(projected.shape[:2], dtype=torch.long, device=projected.device)
        return VisualMemory(projected, mask, frames, memory.source)


class MultiScaleTemporalMamba3(nn.Module):
    """MS-Temba-style multi-scale temporal views using IQ's single Mamba-3 primitive."""

    def __init__(self, hidden_size: int, config: Mamba3MIMOConfig, dilations: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.dilations = dilations
        self.mixers = nn.ModuleList([
            VisualMambaEncoder(hidden_size, config, 1, dtype=dtype, device=device)
            for _ in dilations
        ])
        self.scale_logits = nn.Parameter(torch.zeros(len(dilations), device=device, dtype=dtype))

    def forward(self, memory: VisualMemory) -> VisualMemory:
        if memory.source != "video":
            return memory
        outputs = []
        for dilation, mixer in zip(self.dilations, self.mixers, strict=True):
            sampled_mask = memory.attention_mask & (memory.frame_ids.remainder(dilation) == 0)
            if bool((sampled_mask.sum(dim=1) == 0).any()):
                sampled_mask = memory.attention_mask
            sampled = VisualMemory(memory.hidden_states, sampled_mask, memory.frame_ids, memory.source)
            mixed = mixer(sampled)
            outputs.append(mixed.hidden_states)
        weights = torch.softmax(self.scale_logits.float(), dim=0).to(memory.hidden_states.dtype)
        merged = sum(weight * output for weight, output in zip(weights, outputs, strict=True))
        return VisualMemory(memory.hidden_states + merged, memory.attention_mask, memory.frame_ids, memory.source)


class BidirectionalVideoMamba3(nn.Module):
    """VideoMambaPro-style backward context and residual preservation on Mamba-3."""

    def __init__(self, hidden_size: int, config: Mamba3MIMOConfig, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.forward_mixer = VisualMambaEncoder(hidden_size, config, 1, dtype=dtype, device=device)
        self.backward_mixer = VisualMambaEncoder(hidden_size, config, 1, dtype=dtype, device=device)
        self.gate = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    def forward(self, memory: VisualMemory) -> VisualMemory:
        if memory.source != "video":
            return memory
        forward = self.forward_mixer(memory)
        reversed_memory = VisualMemory(
            memory.hidden_states.flip(1),
            memory.attention_mask.flip(1),
            memory.frame_ids.flip(1),
            memory.source,
        )
        backward = self.backward_mixer(reversed_memory).hidden_states.flip(1)
        correction = 0.5 * (forward.hidden_states + backward)
        hidden = memory.hidden_states + torch.tanh(self.gate) * correction
        return VisualMemory(hidden, memory.attention_mask, memory.frame_ids, memory.source)


class CrossModalFusion(nn.Module):
    """Text queries visual memory; visual tokens never enter quadratic self-attention."""
    def __init__(self, hidden_size: int, heads: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by cross_attention_heads")
        self.norm_q = RMSNorm(hidden_size).to(device=device, dtype=dtype)
        self.norm_kv = RMSNorm(hidden_size).to(device=device, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True, device=device, dtype=dtype
        )
        self.gate = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    def last_instruction_attention(
        self,
        text: torch.Tensor,
        memory: VisualMemory,
        *,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-head pre-softmax attention from the last valid instruction token."""
        q_all = self.norm_q(text)
        kv_all = self.norm_kv(memory.hidden_states)
        batch, _, width = q_all.shape
        heads = self.attn.num_heads
        head_dim = width // heads
        weight = self.attn.in_proj_weight
        bias = self.attn.in_proj_bias
        q_weight = weight[:width]
        k_weight = weight[width : 2 * width]
        q_bias = None if bias is None else bias[:width]
        k_bias = None if bias is None else bias[width : 2 * width]
        scores = memory.hidden_states.new_full(
            memory.attention_mask.shape,
            float("-inf"),
        )
        for row in range(batch):
            valid_text = (
                torch.ones(text.shape[1], dtype=torch.bool, device=text.device)
                if text_mask is None
                else text_mask[row].to(dtype=torch.bool, device=text.device)
            )
            positions = torch.nonzero(valid_text, as_tuple=False).flatten()
            if positions.numel() == 0:
                raise MultimodalError("TransV requires at least one valid instruction token")
            last = q_all[row, positions[-1]]
            q = F.linear(last, q_weight, q_bias).view(heads, head_dim)
            valid_visual = memory.attention_mask[row]
            keys = F.linear(kv_all[row, valid_visual], k_weight, k_bias)
            keys = keys.view(-1, heads, head_dim).transpose(0, 1)
            row_scores = torch.einsum("hd,hvd->hv", q, keys)
            row_scores = row_scores / math.sqrt(head_dim)
            scores[row, valid_visual] = row_scores.float().mean(dim=0).to(scores.dtype)
        return scores

    def forward(
        self,
        text: torch.Tensor,
        memory: VisualMemory,
        *,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.norm_q(text)
        kv = self.norm_kv(memory.hidden_states)
        fused, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=~memory.attention_mask,
            need_weights=False,
        )
        if text_mask is not None:
            fused = fused * text_mask.to(fused.dtype).unsqueeze(-1)
        return text + torch.tanh(self.gate) * fused


class TransVTransfer(nn.Module):
    """TimeViper-style shallow uniform and deep attention-guided compression."""
    def __init__(self, config: IQMultimodalConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        memory: VisualMemory,
        *,
        relevance_scores: torch.Tensor | None,
        deep: bool,
    ) -> VisualMemory:
        batch, _, hidden = memory.hidden_states.shape
        ratio = (
            self.config.transv_deep_keep_ratio
            if deep
            else self.config.transv_shallow_keep_ratio
        )
        valid_counts = memory.attention_mask.sum(dim=1)
        target = torch.clamp(
            torch.ceil(valid_counts.float() * ratio).long(),
            min=self.config.min_visual_tokens,
        )
        target = torch.minimum(target, valid_counts)
        max_target = int(target.max())
        out = memory.hidden_states.new_zeros((batch, max_target, hidden))
        out_mask = torch.zeros((batch, max_target), dtype=torch.bool, device=out.device)
        out_frames = torch.zeros((batch, max_target), dtype=torch.long, device=out.device)

        for row in range(batch):
            src = memory.hidden_states[row, memory.attention_mask[row]]
            src_frames = memory.frame_ids[row, memory.attention_mask[row]]
            count = int(target[row])
            if count == src.shape[0]:
                indices = torch.arange(count, device=src.device)
            elif deep:
                if relevance_scores is None:
                    raise MultimodalError(
                        "deep attention-guided TransV requires last-instruction attention scores"
                    )
                scores = relevance_scores[row, memory.attention_mask[row]]
                # Select by learned cross-attention relevance, then restore
                # temporal/token order for recurrent processing.
                indices = torch.topk(scores, k=count, sorted=False).indices.sort().values
            else:
                # TimeViper shallow TransV uses uniform token dropping.
                indices = torch.linspace(
                    0, src.shape[0] - 1, steps=count, device=src.device
                ).round().long().unique(sorted=True)
                if indices.numel() < count:
                    chosen = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
                    chosen[indices] = True
                    fill = torch.nonzero(~chosen, as_tuple=False).flatten()[: count - indices.numel()]
                    indices = torch.cat((indices, fill)).sort().values

            selected = src.index_select(0, indices[:count])
            selected_frames = src_frames.index_select(0, indices[:count])
            out[row, :count] = selected
            out_mask[row, :count] = True
            out_frames[row, :count] = selected_frames
        return VisualMemory(out, out_mask, out_frames, memory.source)

class IQMultimodalPathway(nn.Module):
    def __init__(
        self,
        config: IQMultimodalConfig,
        hidden_size: int,
        mamba_config: Mamba3MIMOConfig,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision = VisionTower(config, hidden_size, dtype=dtype, device=device)
        self.temporal = MultiScaleTemporalMamba3(
            hidden_size, mamba_config, config.temporal_dilations, dtype=dtype, device=device
        )
        self.video_context = (
            BidirectionalVideoMamba3(hidden_size, mamba_config, dtype=dtype, device=device)
            if config.use_bidirectional_video else nn.Identity()
        )
        self.query_projector = QueryConditionedProjector(
            hidden_size,
            config.cross_attention_heads,
            config.query_projector_tokens,
            dtype=dtype,
            device=device,
        )
        # Vamba advances visual state alongside decoder depth rather than
        # treating Mamba as a one-shot visual front-end.
        self.visual_mixers = nn.ModuleDict(
            {
                str(layer): VisualMambaEncoder(
                    hidden_size,
                    mamba_config,
                    config.visual_mamba_layers,
                    dtype=dtype,
                    device=device,
                )
                for layer in config.fusion_layers
            }
        )
        self.fusion = nn.ModuleDict(
            {
                str(layer): CrossModalFusion(
                    hidden_size, config.cross_attention_heads, dtype=dtype, device=device
                )
                for layer in config.fusion_layers
            }
        )
        self.transv = TransVTransfer(config)

    def encode(
        self,
        *,
        pixel_values: torch.Tensor | None,
        video_values: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> VisualMemory | None:
        memory = self.vision(
            pixel_values=pixel_values,
            video_values=video_values,
            frame_mask=frame_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        if memory is None:
            return None
        memory = _merge_video_frames(
            memory,
            self.config.tokens_per_video_frame,
        )
        memory = self.temporal(memory)
        if isinstance(self.video_context, BidirectionalVideoMamba3):
            memory = self.video_context(memory)
        return memory

    def fuse(
        self,
        layer_index: int,
        text: torch.Tensor,
        memory: VisualMemory | None,
        *,
        text_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, VisualMemory | None]:
        if memory is None or layer_index not in self.config.fusion_layers:
            return text, memory
        memory = self.visual_mixers[str(layer_index)](memory)
        # Query-conditioned compression is applied once, at the first fusion
        # boundary, so it replaces the generic projector role rather than
        # stacking another recurrent backbone.
        if layer_index == self.config.fusion_layers[0]:
            memory = self.query_projector(memory, text, text_mask)
        fusion = self.fusion[str(layer_index)]
        if layer_index in self.config.transv_layers:
            deep = layer_index == self.config.transv_layers[-1]
            relevance_scores = (
                fusion.last_instruction_attention(text, memory, text_mask=text_mask)
                if deep
                else None
            )
            memory = self.transv(
                memory,
                relevance_scores=relevance_scores,
                deep=deep,
            )
        text = fusion(text, memory, text_mask=text_mask)
        return text, memory
