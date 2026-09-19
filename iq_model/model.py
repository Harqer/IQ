from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import GroupedQueryAttention
from .config import IQModelConfig
from .mlp import SwiGLU
from .norm import RMSNorm
from .outputs import CausalLMOutput


class IQDecoderBlock(nn.Module):
    def __init__(self, config: IQModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.residual_dropout(
            self.attn(
                self.input_norm(x),
                position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        )
        x = x + self.residual_dropout(
            self.mlp(self.post_attention_norm(x))
        )
        return x


class IQForCausalLM(nn.Module):
    def __init__(self, config: IQModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                IQDecoderBlock(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token ids")
        if input_ids.numel() and (
            int(input_ids.min()) < 0
            or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError(
                "input_ids contain token ids outside vocabulary"
            )
        if (
            attention_mask is not None
            and attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "attention_mask must have the same shape as input_ids"
            )
        if document_ids is not None:
            if document_ids.shape != input_ids.shape:
                raise ValueError(
                    "document_ids must have the same shape as input_ids"
                )
            if document_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "document_ids must be integer typed"
                )
        if position_ids is not None:
            if position_ids.shape != input_ids.shape:
                raise ValueError(
                    "position_ids must have the same shape as input_ids"
                )
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "position_ids must be integer typed"
                )

        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(
                x,
                position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        loss: torch.Tensor | None = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels must have the same shape as input_ids"
                )
            if labels.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "labels must be integer token ids"
                )
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
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden if return_hidden_states else None,
        )
