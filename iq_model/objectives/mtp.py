from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..attention import DenseContextAttention
from ..attention.masking import prepare_causal_attention
from ..config import IQModelConfig
from ..mlp import SwiGLU
from ..norm import RMSNorm


class MTPConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MTPConfig:
    num_prediction_layers: int = 1

    def __post_init__(self) -> None:
        if self.num_prediction_layers <= 0:
            raise MTPConfigError("num_prediction_layers must be positive")


@dataclass
class MTPDepthOutput:
    depth: int
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: torch.Tensor | None
    valid_target_count: int


@dataclass
class MTPOutput:
    loss: torch.Tensor | None
    depth_outputs: tuple[MTPDepthOutput, ...]

    @property
    def depth_losses(self) -> tuple[torch.Tensor | None, ...]:
        return tuple(output.loss for output in self.depth_outputs)


class MTPPredictionBlock(nn.Module):
    """Additional Transformer block used inside one sequential MTP depth."""

    def __init__(self, config: IQModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = DenseContextAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.residual_dropout(
            self.attn(
                self.attn_norm(x),
                position_ids=position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        )
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class _MTPDepth(nn.Module):
    def __init__(self, config: IQModelConfig) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )
        self.block = MTPPredictionBlock(config)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if inputs_embeds.shape != previous_hidden_state.shape:
            raise ValueError(
                "MTP embedding and previous-hidden tensors must have identical shape"
            )
        projection_input = torch.cat(
            [
                self.enorm(inputs_embeds),
                self.hnorm(previous_hidden_state),
            ],
            dim=-1,
        )
        hidden_states = self.eh_proj(projection_input)
        hidden_states = self.block(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        return self.post_norm(hidden_states)


class MultiTokenPrediction(nn.Module):
    """DeepSeek-style sequential multi-token prediction objective.

    Depth k combines h_i^(k-1) with the shared embedding of token t_(i+k),
    projects back to model width, runs one extra Transformer block, and uses
    the main model's shared LM head to predict t_(i+k+1).
    """

    def __init__(
        self,
        model_config: IQModelConfig,
        mtp_config: MTPConfig,
        *,
        shared_embedding: nn.Embedding,
        shared_head: nn.Linear,
    ) -> None:
        super().__init__()
        if shared_embedding.embedding_dim != model_config.hidden_size:
            raise MTPConfigError("shared embedding width does not match model hidden_size")
        if shared_embedding.num_embeddings != model_config.vocab_size:
            raise MTPConfigError("shared embedding vocabulary does not match model config")
        if shared_head.in_features != model_config.hidden_size:
            raise MTPConfigError("shared head input width does not match model hidden_size")
        if shared_head.out_features != model_config.vocab_size:
            raise MTPConfigError("shared head vocabulary does not match model config")

        self.model_config = model_config
        self.mtp_config = mtp_config
        self.embed_tokens = shared_embedding
        self.shared_head = shared_head
        self.layers = nn.ModuleList(
            [_MTPDepth(model_config) for _ in range(mtp_config.num_prediction_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        main_hidden_states: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> MTPOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token ids")
        if main_hidden_states.ndim != 3:
            raise ValueError(
                "main_hidden_states must have shape [batch, sequence, hidden]"
            )
        if tuple(main_hidden_states.shape[:2]) != tuple(input_ids.shape):
            raise ValueError(
                "main_hidden_states batch/sequence dimensions must match input_ids"
            )
        if main_hidden_states.shape[-1] != self.model_config.hidden_size:
            raise ValueError("main_hidden_states width does not match model config")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("labels must be integer token ids")

        batch_size, sequence_length = input_ids.shape
        if sequence_length < 3:
            raise ValueError("MTP requires sequence length >= 3")

        prepared = prepare_causal_attention(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=input_ids.device,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        base_valid = prepared.valid_tokens
        base_positions = prepared.position_ids
        docs = document_ids.to(input_ids.device) if document_ids is not None else None

        previous_hidden = main_hidden_states
        previous_valid = base_valid
        depth_outputs: list[MTPDepthOutput] = []
        valid_losses: list[torch.Tensor] = []

        for depth, layer in enumerate(self.layers, start=1):
            current_length = sequence_length - depth
            if current_length <= 0:
                break

            previous_aligned = previous_hidden[:, :-1, :]
            previous_valid_aligned = previous_valid[:, :-1]
            current_ids = input_ids[:, depth:]
            current_embeds = self.embed_tokens(current_ids)
            current_valid = base_valid[:, depth:]
            pair_valid = previous_valid_aligned & current_valid

            current_docs = docs[:, depth:] if docs is not None else None
            if docs is not None:
                previous_docs = docs[:, depth - 1 : sequence_length - 1]
                pair_valid = pair_valid & (previous_docs == current_docs)

            current_positions = base_positions[:, depth:]
            hidden_states = layer(
                current_embeds,
                previous_aligned,
                position_ids=current_positions,
                attention_mask=pair_valid,
                document_ids=current_docs,
            )
            hidden_states = hidden_states * pair_valid.unsqueeze(-1).to(
                hidden_states.dtype
            )
            logits = self.shared_head(hidden_states)

            depth_loss: torch.Tensor | None = None
            valid_target_count = 0
            if labels is not None and logits.shape[1] > 1:
                targets = labels[:, depth + 1 :].clone()
                target_valid = pair_valid[:, :-1] & base_valid[:, depth + 1 :]
                if docs is not None:
                    target_valid = target_valid & (
                        docs[:, depth : sequence_length - 1]
                        == docs[:, depth + 1 :]
                    )
                targets = targets.masked_fill(~target_valid, -100)
                valid_target_count = int((targets != -100).sum().item())
                if valid_target_count > 0:
                    depth_loss = F.cross_entropy(
                        logits[:, :-1, :]
                        .contiguous()
                        .view(-1, self.model_config.vocab_size),
                        targets.contiguous().view(-1),
                        ignore_index=-100,
                    )
                    valid_losses.append(depth_loss)

            depth_outputs.append(
                MTPDepthOutput(
                    depth=depth,
                    logits=logits,
                    hidden_states=hidden_states,
                    loss=depth_loss,
                    valid_target_count=valid_target_count,
                )
            )
            previous_hidden = hidden_states
            previous_valid = pair_valid

        if len(depth_outputs) != self.mtp_config.num_prediction_layers:
            raise ValueError(
                "sequence is too short for configured MTP depth: "
                f"sequence_length={sequence_length}, "
                f"depth={self.mtp_config.num_prediction_layers}"
            )

        loss = torch.stack(valid_losses).mean() if valid_losses else None
        if labels is not None and loss is None:
            raise ValueError("MTP batch contains no valid future-token targets")
        return MTPOutput(loss=loss, depth_outputs=tuple(depth_outputs))
