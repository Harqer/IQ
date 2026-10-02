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
    visual_mamba_layers: int = 2
    cross_attention_heads: int = 8
    projector_hidden_multiplier: float = 2.0
    transv_keep_ratio: float = 0.5
    min_visual_tokens: int = 16
    max_frames: int = 16384
    freeze_vision_tower: bool = True
    drop_cls_token: bool = True

    def __post_init__(self) -> None:
        if not self.vision_model_name.strip():
            raise ValueError("vision_model_name must be non-empty")
        if self.vision_backend not in {"auto", "glm5_next"}:
            raise ValueError("vision_backend must be 'auto' or 'glm5_next'")
        if self.visual_mamba_layers <= 0 or self.cross_attention_heads <= 0:
            raise ValueError("visual_mamba_layers and cross_attention_heads must be positive")
        if self.projector_hidden_multiplier <= 0:
            raise ValueError("projector_hidden_multiplier must be positive")
        if not (0.0 < self.transv_keep_ratio <= 1.0):
            raise ValueError("transv_keep_ratio must be in (0, 1]")
        if self.min_visual_tokens <= 0 or self.max_frames <= 0:
            raise ValueError("min_visual_tokens and max_frames must be positive")
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
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "IQMultimodalConfig":
        payload = dict(data)
        payload["fusion_layers"] = tuple(int(x) for x in payload.get("fusion_layers", ()))
        payload["transv_layers"] = tuple(int(x) for x in payload.get("transv_layers", ()))
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
    """Transfer visual evidence into text, then compact visual memory.

    Compression is mask-aware and preserves temporal order. It is only invoked
    after a cross-modal fusion at the same physical depth.
    """
    def __init__(self, config: IQMultimodalConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, memory: VisualMemory) -> VisualMemory:
        batch, tokens, hidden = memory.hidden_states.shape
        valid_counts = memory.attention_mask.sum(dim=1)
        target = torch.clamp(
            torch.ceil(valid_counts.float() * self.config.transv_keep_ratio).long(),
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
                pooled = src
                pooled_frames = src_frames
            else:
                # Adaptive pooling is deterministic, ordered, and includes the
                # entire visual stream rather than dropping late frames.
                pooled = F.adaptive_avg_pool1d(src.transpose(0, 1).unsqueeze(0), count)
                pooled = pooled.squeeze(0).transpose(0, 1)
                frame_float = F.adaptive_avg_pool1d(
                    src_frames.float().view(1, 1, -1), count
                ).view(-1)
                pooled_frames = frame_float.round().long()
            out[row, :count] = pooled
            out_mask[row, :count] = True
            out_frames[row, :count] = pooled_frames
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
        self.visual_mamba = VisualMambaEncoder(
            hidden_size, mamba_config, config.visual_mamba_layers, dtype=dtype, device=device
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
        return self.visual_mamba(memory) if memory is not None else None

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
        text = self.fusion[str(layer_index)](text, memory, text_mask=text_mask)
        if layer_index in self.config.transv_layers:
            memory = self.transv(memory)
        return text, memory
