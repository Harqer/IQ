from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm, Phi3RotaryEmbedding

from .components import IQBlock, build_iq_block
from .config import IQArchitectureConfig
from .mixers import MixerContext
from .reasoning import RecurrentReasoningCore


@dataclass
class IQOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    mtp_logits: tuple[torch.Tensor, ...] = ()
    verifier_scores: Optional[torch.Tensor] = None
    core_pass_states: tuple[torch.Tensor, ...] = ()


class IQRecurrentPhiModel(nn.Module):
    """IQ hybrid-v1 reference backbone.

    Unique prelude/coda blocks stay Phi-compatible for the first transfer. The
    recurrent middle core is heterogeneous and uses the configured physical mixer
    schedule (Gated DeltaNet + sparse attention anchors by default).
    """

    def __init__(self, phi_config: Phi3Config, iq_config: IQArchitectureConfig | None = None) -> None:
        super().__init__()
        self.phi_config = phi_config
        self.iq_config = iq_config or IQArchitectureConfig()
        self.iq_config.validate_teacher_depth(phi_config.num_hidden_layers)

        if self.iq_config.latent_slots:
            raise NotImplementedError("latent workspace is reserved for its controlled implementation stage")
        if self.iq_config.use_adaptive_halting:
            raise NotImplementedError("adaptive halting is disabled until fixed-depth recurrence is validated")
        if (
            self.iq_config.attention_position_strategy == "path"
            and "nsa" in self.iq_config.core_mixer_schedule
        ):
            raise NotImplementedError(
                "PaTH-inside-NSA is not treated as a drop-in composition. Use a "
                "path_attention anchor schedule to test PaTH independently, or keep "
                "NSA on its validated RoPE path until the joint operator is derived."
            )

        h = phi_config.hidden_size
        self.embed_tokens = nn.Embedding(phi_config.vocab_size, h, phi_config.pad_token_id)
        self.embed_dropout = nn.Dropout(getattr(phi_config, "embd_pdrop", 0.0))
        self.rotary_emb = Phi3RotaryEmbedding(phi_config)

        self.prelude = nn.ModuleList(
            [
                build_iq_block(
                    phi_config,
                    self.iq_config,
                    layer_idx=i,
                    mixer_kind=self.iq_config.prelude_mixer,
                )
                for i in range(self.iq_config.prelude_layers)
            ]
        )

        core_start = self.iq_config.prelude_layers
        core_blocks = nn.ModuleList(
            [
                build_iq_block(
                    phi_config,
                    self.iq_config,
                    layer_idx=core_start + i,
                    mixer_kind=self.iq_config.mixer_for_core_block(i),
                )
                for i in range(self.iq_config.recurrent_layers)
            ]
        )
        self.reasoning_core = RecurrentReasoningCore(
            core_blocks,
            hidden_size=h,
            passes=self.iq_config.recurrent_passes,
            use_pass_embeddings=self.iq_config.use_pass_embeddings,
            delta_scale=self.iq_config.recurrent_delta_scale,
        )

        coda_start = (
            self.iq_config.prelude_layers
            + self.iq_config.recurrent_layers * self.iq_config.recurrent_passes
        )
        self.coda = nn.ModuleList(
            [
                build_iq_block(
                    phi_config,
                    self.iq_config,
                    layer_idx=coda_start + i,
                    mixer_kind=self.iq_config.coda_mixer,
                )
                for i in range(self.iq_config.coda_layers)
            ]
        )

        self.final_norm = Phi3RMSNorm(h, eps=phi_config.rms_norm_eps)
        self.lm_head = nn.Linear(h, phi_config.vocab_size, bias=False)

        self.mtp_heads = nn.ModuleList(
            [nn.Linear(h, phi_config.vocab_size, bias=False) for _ in range(self.iq_config.mtp_heads)]
        )
        self.verifier_head = nn.Linear(h, 1, bias=True) if self.iq_config.use_verifier_head else None

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
        block: IQBlock,
        hidden_states: torch.Tensor,
        *,
        context: MixerContext,
    ) -> torch.Tensor:
        return block(hidden_states, context=context)

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
        context = MixerContext(
            padding_mask=attention_mask,
            causal_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        for block in self.prelude:
            hidden_states = self._run_block(block, hidden_states, context=context)

        core_output = self.reasoning_core(
            hidden_states,
            context=context,
            capture_passes=capture_core_passes,
        )
        hidden_states = core_output.hidden_states

        for block in self.coda:
            hidden_states = self._run_block(block, hidden_states, context=context)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        mtp_logits = tuple(head(hidden_states) for head in self.mtp_heads)
        verifier_scores = self.verifier_head(hidden_states) if self.verifier_head is not None else None

        return IQOutput(
            logits=logits,
            hidden_states=hidden_states,
            mtp_logits=mtp_logits,
            verifier_scores=verifier_scores,
            core_pass_states=core_output.pass_states,
        )
