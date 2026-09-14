from __future__ import annotations

from torch import nn
import torch

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3MLP, Phi3RMSNorm


class IQBlock(nn.Module):
    """Pluggable decoder block used by the IQ backbone.

    The initial mixer/FFN are Phi-compatible so weight transfer has a clean target.
    Later research can replace either module independently without changing the
    recurrent topology.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        mixer: nn.Module,
        feed_forward: nn.Module,
        resid_pdrop: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_norm = Phi3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_mixer_norm = Phi3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mixer = mixer
        self.feed_forward = feed_forward
        self.resid_mixer_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        mixed, _ = self.mixer(
            hidden_states=self.input_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = residual + self.resid_mixer_dropout(mixed)

        residual = hidden_states
        hidden_states = self.feed_forward(self.post_mixer_norm(hidden_states))
        return residual + self.resid_ffn_dropout(hidden_states)


def build_phi_compatible_block(config: Phi3Config, *, layer_idx: int) -> IQBlock:
    """Build a block with an exact Phi-compatible parameterization."""
    return IQBlock(
        hidden_size=config.hidden_size,
        rms_norm_eps=config.rms_norm_eps,
        mixer=Phi3Attention(config=config, layer_idx=layer_idx),
        feed_forward=Phi3MLP(config),
        resid_pdrop=config.resid_pdrop,
    )
