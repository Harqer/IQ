"""Actual Hugging Face CausalLM wrapper for IQ's native CUDA architecture.

Same graph: model = IQHybridForCausalLM(config.validated_iq_hybrid()).
No proxy GPT, synthetic weights, or Mamba-2 implementation.

This class supports native Hugging Face model state names `model.*` and the
PreTrainedModel checkpoint interface. Full 78-block instantiation is
CUDA-only; from_pretrained(save_pretrained) with real massive weights is
NOT YET QUALIFIED and may require HF's meta/streaming initialization work.

Transformers v5.17.0:
https://huggingface.co/docs/transformers/main_classes/model
https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/llama/modeling_llama.py
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from iq_model.hybrid import IQHybridForCausalLM
from iq_training.hf_model_config import (
    IQHFCausalConfig,
    register_iq_hf_config,
)


class IQHFModelError(RuntimeError):
    pass


class IQHFForCausalLM(PreTrainedModel, GenerationMixin):
    """Transformers runtime wrapping the actual fused IQ model verbatim."""

    config_class = IQHFCausalConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _supports_context_parallel = False
    _supports_flash_attn = False
    _supports_sdpa = False
    supports_gradient_checkpointing = False
    _no_split_modules = ["ParallelMambaContextMoEBlock"]

    def __init__(self, config: IQHFCausalConfig) -> None:
        if not isinstance(config, IQHFCausalConfig):
            raise IQHFModelError("IQHFForCausalLM requires IQHFCausalConfig")
        hybrid = config.validated_iq_hybrid()
        if not torch.cuda.is_available():
            raise IQHFModelError(
                "full IQ Mamba-3 MIMO requires NVIDIA CUDA; "
                "CPU/HF meta-loading support is not implemented"
            )
        if hybrid.expert_parallel_size != 1:
            raise IQHFModelError(
                "standalone HF IQ weights require EP=1; use MCore for EP shards"
            )
        super().__init__(config)
        self.model = IQHybridForCausalLM(
            hybrid, device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.bfloat16,
        )
        # Do not call self.post_init(): IQHybridForCausalLM already applies its
        # own parameter initialization and must not be silently reinitialized.
        # Standard transformers checkpoint materialization must be separately
        # validated for the full pretrained run.

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        if not isinstance(value, torch.nn.Embedding):
            raise IQHFModelError("IQ input embedding must remain an nn.Embedding")
        if tuple(value.weight.shape) != (
            self.config.vocab_size, self.config.hidden_size
        ):
            raise IQHFModelError("replacement input embedding geometry differs")
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.lm_head

    def set_output_embeddings(self, value: torch.nn.Module) -> None:
        if not isinstance(value, torch.nn.Linear) or value.bias is not None:
            raise IQHFModelError("IQ output embedding must remain a bias-free linear")
        if tuple(value.weight.shape) != (
            self.config.vocab_size, self.config.hidden_size
        ):
            raise IQHFModelError("replacement output embedding geometry differs")
        self.model.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        *,
        use_cache: bool | None = None,
        past_key_values: Any | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        if input_ids is None:
            raise IQHFModelError("IQ requires explicit input_ids")
        if past_key_values is not None or use_cache:
            raise IQHFModelError(
                "HF Cache conversion to IQHybridInferenceCache is not implemented"
            )
        if output_attentions:
            raise IQHFModelError(
                "IQ CSA per-head attention tensors are not exposed through HF"
            )
        if kwargs:
            raise IQHFModelError(
                "unsupported HF IQ forward arguments: " + ", ".join(sorted(kwargs))
            )
        output = self.model(
            input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_hidden_states=bool(output_hidden_states),
        )
        result = CausalLMOutputWithPast(
            loss=output.loss, logits=output.logits,
            past_key_values=None,
            hidden_states=(
                (output.hidden_states,) if output_hidden_states
                and output.hidden_states is not None else None
            ),
            attentions=None,
        )
        return result if return_dict else result.to_tuple()


def register_iq_hf_causal_lm() -> None:
    """Register both config and the REAL IQ CausalLM class, no fabricated GPT."""
    register_iq_hf_config()
    AutoModelForCausalLM.register(IQHFCausalConfig, IQHFForCausalLM)
