from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)

from .config import IQArchitectureConfig


@dataclass
class IQOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    mtp_logits: tuple[torch.Tensor, ...] = ()
    verifier_scores: Optional[torch.Tensor] = None
    core_pass_states: tuple[torch.Tensor, ...] = ()


class IQRecurrentPhiModel(nn.Module):
    """Research v0 IQ backbone.

    The model keeps Phi-3/Phi-4-mini compatible token embeddings, GQA blocks,
    SwiGLU MLPs, RMSNorm, and RoPE mechanics. Its architectural change is the
    shared recurrent middle core.

    This is deliberately a PyTorch reference model for transfer/training. Mojo
    kernels should be implemented only after the topology and transfer behavior are
    validated.
    """

    def __init__(self, phi_config: Phi3Config, iq_config: IQArchitectureConfig | None = None) -> None:
        super().__init__()
        self.phi_config = phi_config
        self.iq_config = iq_config or IQArchitectureConfig()
        self.iq_config.validate_teacher_depth(phi_config.num_hidden_layers)

        if self.iq_config.latent_slots:
            raise NotImplementedError("latent workspace is reserved for a later controlled experiment")
        if self.iq_config.use_adaptive_halting:
            raise NotImplementedError("adaptive halting is disabled until fixed-depth recurrence is validated")

        h = phi_config.hidden_size
        self.embed_tokens = nn.Embedding(phi_config.vocab_size, h, phi_config.pad_token_id)
        self.embed_dropout = nn.Dropout(getattr(phi_config, "embd_pdrop", 0.0))
        self.rotary_emb = Phi3RotaryEmbedding(phi_config)

        self.prelude = nn.ModuleList(
            [
                Phi3DecoderLayer(phi_config, layer_idx=i)
                for i in range(self.iq_config.prelude_layers)
            ]
        )

        core_start = self.iq_config.prelude_layers
        self.recurrent_core = nn.ModuleList(
            [
                Phi3DecoderLayer(phi_config, layer_idx=core_start + i)
                for i in range(self.iq_config.recurrent_layers)
            ]
        )

        coda_start = (
            self.iq_config.prelude_layers
            + self.iq_config.recurrent_layers * self.iq_config.recurrent_passes
        )
        self.coda = nn.ModuleList(
            [
                Phi3DecoderLayer(phi_config, layer_idx=coda_start + i)
                for i in range(self.iq_config.coda_layers)
            ]
        )

        if self.iq_config.use_pass_embeddings:
            self.pass_embeddings = nn.Parameter(
                torch.zeros(self.iq_config.recurrent_passes, h)
            )
        else:
            self.register_parameter("pass_embeddings", None)

        self.final_norm = Phi3RMSNorm(h, eps=phi_config.rms_norm_eps)
        self.lm_head = nn.Linear(h, phi_config.vocab_size, bias=False)

        self.mtp_heads = nn.ModuleList(
            [nn.Linear(h, phi_config.vocab_size, bias=False) for _ in range(self.iq_config.mtp_heads)]
        )
        self.verifier_head = (
            nn.Linear(h, 1, bias=True) if self.iq_config.use_verifier_head else None
        )

        if getattr(phi_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def physical_depth(self) -> int:
        return self.iq_config.physical_layers

    @property
    def effective_depth(self) -> int:
        return self.iq_config.effective_depth

    def _build_causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        mask_value = torch.finfo(dtype).min

        causal = torch.full((seq_len, seq_len), mask_value, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)
        causal = causal.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)

        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError(
                    f"attention_mask must have shape {(batch, seq_len)}, got {tuple(attention_mask.shape)}"
                )
            key_padding = attention_mask[:, None, None, :].to(device=device)
            causal = causal.masked_fill(key_padding == 0, mask_value)

        return causal

    def _position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch, -1)
        if position_ids.shape != (batch, seq_len):
            raise ValueError(
                f"position_ids must have shape {(batch, seq_len)}, got {tuple(position_ids.shape)}"
            )
        return self.rotary_emb(hidden_states, position_ids)

    @staticmethod
    def _run_block(
        block: Phi3DecoderLayer,
        hidden_states: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return block(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        capture_core_passes: bool = False,
    ) -> IQOutput:
        hidden_states = self.embed_dropout(self.embed_tokens(input_ids))
        causal_mask = self._build_causal_mask(hidden_states, attention_mask)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)

        for block in self.prelude:
            hidden_states = self._run_block(
                block,
                hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        pass_states: list[torch.Tensor] = []
        delta_scale = self.iq_config.recurrent_delta_scale

        for pass_index in range(self.iq_config.recurrent_passes):
            if self.pass_embeddings is not None:
                hidden_states = hidden_states + self.pass_embeddings[pass_index].view(1, 1, -1)

            for block in self.recurrent_core:
                before = hidden_states
                candidate = self._run_block(
                    block,
                    before,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden_states = before + delta_scale * (candidate - before)

            if capture_core_passes:
                pass_states.append(hidden_states)

        for block in self.coda:
            hidden_states = self._run_block(
                block,
                hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        mtp_logits = tuple(head(hidden_states) for head in self.mtp_heads)
        verifier_scores = self.verifier_head(hidden_states) if self.verifier_head is not None else None

        return IQOutput(
            logits=logits,
            hidden_states=hidden_states,
            mtp_logits=mtp_logits,
            verifier_scores=verifier_scores,
            core_pass_states=tuple(pass_states),
        )
