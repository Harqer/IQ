from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm

from .config import IQArchitectureConfig
from .mixers import MixerContext, build_mixer


class PhiCompatibleSwiGLU(nn.Module):
    """Exact Phi-style gated SiLU FFN with transfer-compatible parameter names.

    An optional activation observer exposes the intermediate gated-neuron activity
    for keystone-neuron analysis without changing the forward computation.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self._activation_observer: Callable[[torch.Tensor], None] | None = None

    def set_activation_observer(
        self,
        observer: Callable[[torch.Tensor], None] | None,
    ) -> None:
        self._activation_observer = observer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        activated = F.silu(gate) * up
        if self._activation_observer is not None:
            self._activation_observer(activated.detach())
        return self.down_proj(activated)


class IQBlock(nn.Module):
    """Pre-norm decoder block whose mixer is selected independently per layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        mixer: nn.Module,
        feed_forward: nn.Module,
        resid_pdrop: float = 0.0,
        mixer_kind: str,
    ) -> None:
        super().__init__()
        self.mixer_kind = mixer_kind
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
        context: MixerContext,
    ) -> torch.Tensor:
        residual = hidden_states
        mixed = self.mixer(self.input_norm(hidden_states), context=context)
        hidden_states = residual + self.resid_mixer_dropout(mixed)

        residual = hidden_states
        hidden_states = self.feed_forward(self.post_mixer_norm(hidden_states))
        return residual + self.resid_ffn_dropout(hidden_states)


def build_iq_block(
    config: Phi3Config,
    architecture: IQArchitectureConfig,
    *,
    layer_idx: int,
    mixer_kind: str,
) -> IQBlock:
    if architecture.ffn_kind != "dense_swiglu":
        raise NotImplementedError(
            "shared_routed_moe is an intentional later FFN transition; "
            "do not silently substitute dense SwiGLU"
        )

    return IQBlock(
        hidden_size=config.hidden_size,
        rms_norm_eps=config.rms_norm_eps,
        mixer=build_mixer(mixer_kind, config, architecture, layer_idx=layer_idx),
        feed_forward=PhiCompatibleSwiGLU(config.hidden_size, config.intermediate_size),
        resid_pdrop=config.resid_pdrop,
        mixer_kind=mixer_kind,
    )
