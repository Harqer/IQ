from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping
import json

import torch
from torch import nn
from torch.nn import functional as F

from .architecture import HybridLayerType, HybridSchedule
from .attention import (
    CompressedContextConfig,
    CompressedSparseContextAttention,
    DenseContextAttention,
    HeavilyCompressedContextAttention,
)
from .config import IQModelConfig
from .mlp import MoEOutput, RoutedMoEConfig, RoutedSwiGLUMoELayer
from .norm import RMSNorm
from .state import Mamba3MIMOConfig, Mamba3MIMOState


class HybridModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class IQHybridConfig:
    model: IQModelConfig
    schedule: HybridSchedule
    mamba3: Mamba3MIMOConfig
    moe: RoutedMoEConfig
    compressed_context: CompressedContextConfig | None = None

    def __post_init__(self) -> None:
        hidden = self.model.hidden_size
        if self.mamba3.d_model != hidden:
            raise HybridModelError(
                f"Mamba-3 d_model={self.mamba3.d_model} does not match hidden_size={hidden}"
            )
        if self.moe.hidden_size != hidden:
            raise HybridModelError(
                f"MoE hidden_size={self.moe.hidden_size} does not match hidden_size={hidden}"
            )
        mamba_layers = self.schedule.count(HybridLayerType.MAMBA3)
        if self.mamba3.num_layers != mamba_layers:
            raise HybridModelError(
                "Mamba-3 num_layers must equal the number of Mamba layers in the schedule: "
                f"config={self.mamba3.num_layers}, schedule={mamba_layers}"
            )
        if self.schedule.count(HybridLayerType.MOE) == 0:
            raise HybridModelError("hybrid schedule requires at least one MoE layer")
        compressed_layers = (
            self.schedule.count(HybridLayerType.CSA)
            + self.schedule.count(HybridLayerType.HCA)
        )
        if compressed_layers:
            if self.compressed_context is None:
                raise HybridModelError(
                    "CSA/HCA schedule requires compressed_context configuration"
                )
            if self.compressed_context.hidden_size != hidden:
                raise HybridModelError(
                    "compressed_context hidden_size does not match model hidden_size"
                )
        elif self.compressed_context is not None:
            raise HybridModelError(
                "compressed_context config is present but the schedule has no CSA/HCA layers"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model": self.model.to_dict(),
            "schedule": self.schedule.to_dict(),
            "mamba3": asdict(self.mamba3),
            "moe": asdict(self.moe),
            "compressed_context": (
                asdict(self.compressed_context)
                if self.compressed_context is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "IQHybridConfig":
        if int(data.get("schema_version", -1)) != 1:
            raise HybridModelError(
                f"unsupported hybrid-model schema: {data.get('schema_version')!r}"
            )
        model = data.get("model")
        schedule = data.get("schedule")
        mamba3 = data.get("mamba3")
        moe = data.get("moe")
        compressed_context = data.get("compressed_context")
        if not isinstance(model, dict):
            raise HybridModelError("hybrid config model must be an object")
        if not isinstance(schedule, dict):
            raise HybridModelError("hybrid config schedule must be an object")
        if not isinstance(mamba3, dict):
            raise HybridModelError("hybrid config mamba3 must be an object")
        if not isinstance(moe, dict):
            raise HybridModelError("hybrid config moe must be an object")
        if compressed_context is not None and not isinstance(compressed_context, dict):
            raise HybridModelError(
                "hybrid config compressed_context must be an object or null"
            )
        try:
            return cls(
                model=IQModelConfig.from_dict(model),
                schedule=HybridSchedule.from_dict(schedule),
                mamba3=Mamba3MIMOConfig(**mamba3),
                moe=RoutedMoEConfig(**moe),
                compressed_context=(
                    CompressedContextConfig(**compressed_context)
                    if isinstance(compressed_context, dict)
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            raise HybridModelError(
                f"invalid hybrid-model configuration: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, path: str) -> "IQHybridConfig":
        from pathlib import Path
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HybridModelError(
                f"invalid hybrid-model config file: {path}"
            ) from exc
        if not isinstance(data, dict):
            raise HybridModelError("hybrid-model config JSON must contain an object")
        return cls.from_dict(data)

    def write_json(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class MambaPackedLayout:
    packed_hidden_states: torch.Tensor
    cu_seqlens: torch.Tensor | None
    flat_indices: torch.Tensor | None
    original_shape: tuple[int, int, int]

    @property
    def packed(self) -> bool:
        return self.flat_indices is not None


@dataclass
class HybridCausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    hidden_states: torch.Tensor | None
    load_balance_loss: torch.Tensor | None
    router_z_loss: torch.Tensor | None
    expert_counts: tuple[torch.Tensor, ...]
    schedule_fingerprint: str


def _valid_tokens(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch, sequence, _ = hidden_states.shape
    if attention_mask is None:
        return torch.ones(
            (batch, sequence),
            dtype=torch.bool,
            device=hidden_states.device,
        )
    if attention_mask.shape != (batch, sequence):
        raise HybridModelError(
            f"attention_mask must have shape {(batch, sequence)}"
        )
    return attention_mask.to(device=hidden_states.device, dtype=torch.bool)


def _validate_document_ids(
    document_ids: torch.Tensor | None,
    valid: torch.Tensor,
) -> torch.Tensor | None:
    if document_ids is None:
        return None
    if document_ids.shape != valid.shape:
        raise HybridModelError(
            f"document_ids must have shape {tuple(valid.shape)}"
        )
    if document_ids.dtype not in (torch.int32, torch.int64):
        raise HybridModelError("document_ids must be integer typed")
    docs = document_ids.to(valid.device)
    for row in range(docs.shape[0]):
        seen: set[int] = set()
        current: int | None = None
        for col in range(docs.shape[1]):
            if not bool(valid[row, col]):
                continue
            doc = int(docs[row, col])
            if current is None:
                current = doc
                seen.add(doc)
            elif doc != current:
                if doc in seen:
                    raise HybridModelError(
                        "document_ids cannot reappear in non-contiguous valid-token segments"
                    )
                current = doc
                seen.add(doc)
    return docs


def pack_mamba_varlen(
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
) -> MambaPackedLayout:
    """Pack padded / multi-document batches for Mamba-3's varlen kernel.

    Fully valid batches without document boundaries keep the native batched
    Mamba path. Otherwise valid tokens are packed row-major into batch=1 and
    cu_seqlens resets recurrence at each row/document boundary.
    """

    if hidden_states.ndim != 3:
        raise HybridModelError(
            "hidden_states must have shape [batch, sequence, hidden]"
        )
    batch, sequence, hidden = hidden_states.shape
    if batch <= 0 or sequence <= 0 or hidden <= 0:
        raise HybridModelError("hidden-state dimensions must be positive")

    valid = _valid_tokens(hidden_states, attention_mask)
    docs = _validate_document_ids(document_ids, valid)

    if docs is None and bool(valid.all()):
        return MambaPackedLayout(
            packed_hidden_states=hidden_states,
            cu_seqlens=None,
            flat_indices=None,
            original_shape=(batch, sequence, hidden),
        )

    flat_indices: list[int] = []
    segment_lengths: list[int] = []
    for row in range(batch):
        row_positions = torch.nonzero(valid[row], as_tuple=False).flatten().tolist()
        if not row_positions:
            continue

        segment_length = 0
        previous_doc: int | None = None
        for col in row_positions:
            doc = int(docs[row, col]) if docs is not None else row
            if segment_length and doc != previous_doc:
                segment_lengths.append(segment_length)
                segment_length = 0
            flat_indices.append(row * sequence + col)
            segment_length += 1
            previous_doc = doc
        if segment_length:
            segment_lengths.append(segment_length)

    if not flat_indices:
        raise HybridModelError("Mamba input contains no valid tokens")

    flat_index_tensor = torch.tensor(
        flat_indices,
        dtype=torch.long,
        device=hidden_states.device,
    )
    flat = hidden_states.reshape(batch * sequence, hidden)
    packed = flat.index_select(0, flat_index_tensor).unsqueeze(0)

    boundaries = [0]
    total = 0
    for length in segment_lengths:
        total += length
        boundaries.append(total)
    if total != len(flat_indices):
        raise HybridModelError("internal Mamba varlen packing mismatch")

    cu_seqlens = torch.tensor(
        boundaries,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    return MambaPackedLayout(
        packed_hidden_states=packed,
        cu_seqlens=cu_seqlens,
        flat_indices=flat_index_tensor,
        original_shape=(batch, sequence, hidden),
    )


def unpack_mamba_varlen(
    packed_output: torch.Tensor,
    layout: MambaPackedLayout,
) -> torch.Tensor:
    if not layout.packed:
        if tuple(packed_output.shape) != layout.original_shape:
            raise HybridModelError(
                "unpacked Mamba output shape does not match original input"
            )
        return packed_output

    batch, sequence, hidden = layout.original_shape
    if packed_output.ndim != 3 or packed_output.shape[0] != 1:
        raise HybridModelError(
            "packed Mamba output must have shape [1, valid_tokens, hidden]"
        )
    if packed_output.shape[-1] != hidden:
        raise HybridModelError("packed Mamba output hidden width changed")
    assert layout.flat_indices is not None
    if packed_output.shape[1] != layout.flat_indices.numel():
        raise HybridModelError("packed Mamba output token count changed")

    flat = torch.zeros(
        (batch * sequence, hidden),
        dtype=packed_output.dtype,
        device=packed_output.device,
    )
    flat.index_copy_(
        0,
        layout.flat_indices,
        packed_output.squeeze(0),
    )
    return flat.view(batch, sequence, hidden)


class Mamba3ResidualLayer(nn.Module):
    def __init__(
        self,
        model_config: IQModelConfig,
        mamba_config: Mamba3MIMOConfig,
        *,
        mamba_layer_idx: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            model_config.hidden_size,
            model_config.rms_norm_eps,
        ).to(device=device, dtype=dtype)
        self.mamba = Mamba3MIMOState(
            mamba_config,
            layer_idx=mamba_layer_idx,
            dtype=dtype,
            device=device,
        )
        self.residual_dropout = nn.Dropout(
            model_config.residual_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm(x)
        layout = pack_mamba_varlen(
            normalized,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        mixed = self.mamba(
            layout.packed_hidden_states,
            cu_seqlens=layout.cu_seqlens,
        )
        mixed = unpack_mamba_varlen(mixed, layout)
        return x + self.residual_dropout(mixed)


class DenseContextResidualLayer(nn.Module):
    def __init__(
        self,
        config: IQModelConfig,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attention = DenseContextAttention(config)
        self.residual_dropout = nn.Dropout(
            config.residual_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return x + self.residual_dropout(
            self.attention(
                self.norm(x),
                position_ids=position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        )


class CompressedContextResidualLayer(nn.Module):
    def __init__(
        self,
        model_config: IQModelConfig,
        compressed_config: CompressedContextConfig,
        *,
        mode: HybridLayerType,
    ) -> None:
        super().__init__()
        if mode not in {HybridLayerType.CSA, HybridLayerType.HCA}:
            raise ValueError("compressed context mode must be CSA or HCA")
        self.norm = RMSNorm(
            model_config.hidden_size,
            model_config.rms_norm_eps,
        )
        self.attention = (
            CompressedSparseContextAttention(compressed_config)
            if mode is HybridLayerType.CSA
            else HeavilyCompressedContextAttention(compressed_config)
        )
        self.residual_dropout = nn.Dropout(
            model_config.residual_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return x + self.residual_dropout(
            self.attention(
                self.norm(x),
                position_ids=position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        )


class IQHybridForCausalLM(nn.Module):
    """Executable heterogeneous Mamba-3 / MoE / dense-attention backbone.

    CSA, HCA, and executive layers deliberately fail construction until their
    exact reference implementations are present. They are never substituted by
    dense attention or another layer type.
    """

    def __init__(
        self,
        config: IQHybridConfig,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.model_config = config.model
        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise HybridModelError(
                "IQHybridForCausalLM requires the production CUDA Mamba-3 MIMO runtime"
            )

        unsupported = tuple(
            layer
            for layer in config.schedule.layers
            if layer is HybridLayerType.EXECUTIVE
        )
        if unsupported:
            names = ", ".join(layer.value for layer in unsupported)
            raise HybridModelError(
                "schedule requests layer types whose exact runtime is not implemented yet: "
                f"{names}"
            )

        self.embed_tokens = nn.Embedding(
            config.model.vocab_size,
            config.model.hidden_size,
            device=device_obj,
            dtype=dtype,
        )
        self.layers = nn.ModuleList()
        mamba_layer_idx = 0
        for layer_type in config.schedule.layers:
            if layer_type is HybridLayerType.MAMBA3:
                layer = Mamba3ResidualLayer(
                    config.model,
                    config.mamba3,
                    mamba_layer_idx=mamba_layer_idx,
                    dtype=dtype,
                    device=device_obj,
                )
                mamba_layer_idx += 1
            elif layer_type is HybridLayerType.MOE:
                layer = RoutedSwiGLUMoELayer(
                    config.moe,
                    norm_eps=config.model.rms_norm_eps,
                    residual_dropout=config.model.residual_dropout,
                ).to(device=device_obj, dtype=dtype)
            elif layer_type is HybridLayerType.DENSE_ATTENTION:
                layer = DenseContextResidualLayer(
                    config.model,
                ).to(device=device_obj, dtype=dtype)
            elif layer_type in {HybridLayerType.CSA, HybridLayerType.HCA}:
                assert config.compressed_context is not None
                layer = CompressedContextResidualLayer(
                    config.model,
                    config.compressed_context,
                    mode=layer_type,
                ).to(device=device_obj, dtype=dtype)
            else:
                raise HybridModelError(
                    f"unsupported hybrid layer type: {layer_type.value}"
                )
            self.layers.append(layer)

        self.norm = RMSNorm(
            config.model.hidden_size,
            config.model.rms_norm_eps,
        ).to(device=device_obj, dtype=dtype)
        self.lm_head = nn.Linear(
            config.model.hidden_size,
            config.model.vocab_size,
            bias=False,
            device=device_obj,
            dtype=dtype,
        )
        nn.init.normal_(
            self.embed_tokens.weight,
            mean=0.0,
            std=config.model.initializer_range,
        )
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=config.model.initializer_range,
        )
        if config.model.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> HybridCausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token ids")
        if input_ids.numel() and (
            int(input_ids.min()) < 0
            or int(input_ids.max()) >= self.model_config.vocab_size
        ):
            raise ValueError("input_ids contain token ids outside vocabulary")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("labels must be integer token ids")

        x = self.embed_tokens(input_ids)
        moe_outputs: list[MoEOutput] = []
        for layer_type, layer in zip(
            self.config.schedule.layers,
            self.layers,
            strict=True,
        ):
            if layer_type is HybridLayerType.MAMBA3:
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                )
            elif layer_type is HybridLayerType.MOE:
                moe_output = layer(x)
                x = moe_output.hidden_states
                moe_outputs.append(moe_output)
            elif layer_type in {
                HybridLayerType.DENSE_ATTENTION,
                HybridLayerType.CSA,
                HybridLayerType.HCA,
            }:
                x = layer(
                    x,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                )
            else:
                raise HybridModelError(
                    f"unsupported runtime layer type: {layer_type.value}"
                )

        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        language_loss: torch.Tensor | None = None
        if labels is not None:
            if input_ids.shape[1] < 2:
                raise ValueError(
                    "sequence length must be at least 2 when labels are provided"
                )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                target_valid = attention_mask[:, 1:].to(
                    dtype=torch.bool,
                    device=shift_labels.device,
                )
                shift_labels = shift_labels.masked_fill(
                    ~target_valid,
                    -100,
                )
            language_loss = F.cross_entropy(
                shift_logits.view(-1, self.model_config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        load_balance_loss = (
            torch.stack(
                [output.load_balance_loss for output in moe_outputs]
            ).mean()
            if moe_outputs
            else None
        )
        router_z_loss = (
            torch.stack(
                [output.router_z_loss for output in moe_outputs]
            ).mean()
            if moe_outputs
            else None
        )
        return HybridCausalLMOutput(
            logits=logits,
            loss=language_loss,
            hidden_states=hidden if return_hidden_states else None,
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
            expert_counts=tuple(
                output.expert_counts for output in moe_outputs
            ),
            schedule_fingerprint=self.config.schedule.fingerprint,
        )
