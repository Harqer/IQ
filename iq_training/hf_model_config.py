"""A real Hugging Face-compatible *configuration* for IQ's custom backbone.

This exposes canonical IQ architecture identity and stable serialization;
it intentionally does not register an AutoModelForCausalLM or claim that
Megatron's GPTModel is IQ's fused Mamba-3/CSA architecture.

NVIDIA's Bridge integration requires a distinct HF source class, provider
configuration and fully mapped native MCore target. This module completes
only the config/round-trip prerequisite, without allocating model weights.

https://huggingface.co/docs/transformers/main_classes/configuration
https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from transformers import AutoConfig, PretrainedConfig

from iq_model.hybrid import IQHybridConfig
from iq_transfer.complete_transplant import nvidia_parallel_78_config


class IQHFConfigError(ValueError):
    pass


class IQHFCausalConfig(PretrainedConfig):
    model_type = "iq_mamba3_csa"
    is_composition = False

    def __init__(
        self,
        *,
        iq_hybrid: dict[str, Any] | None = None,
        iq_hybrid_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        raw = (
            nvidia_parallel_78_config().to_dict()
            if iq_hybrid is None else deepcopy(iq_hybrid)
        )
        if not isinstance(raw, dict):
            raise IQHFConfigError("iq_hybrid must be an actual IQ hybrid config object")
        hybrid = IQHybridConfig.from_dict(raw)
        canonical = hybrid.to_dict()
        if iq_hybrid_sha256 is not None and iq_hybrid_sha256 != hybrid.fingerprint:
            raise IQHFConfigError("IQ architecture fingerprint mismatch")
        for flat, expected in (
            ("vocab_size", hybrid.model.vocab_size),
            ("hidden_size", hybrid.model.hidden_size),
            ("num_hidden_layers", 78),
            ("tie_word_embeddings", hybrid.model.tie_word_embeddings),
        ):
            if flat in kwargs and kwargs[flat] != expected:
                raise IQHFConfigError(
                    f"{flat} contradicts IQ's canonical nested architecture"
                )
        # Remove duplicated HF flat fields: they are always derived from the
        # canonical IQ config and must not diverge after JSON serialization.
        for duplicate in (
            "vocab_size", "hidden_size", "num_hidden_layers",
            "tie_word_embeddings",
        ):
            kwargs.pop(duplicate, None)
        super().__init__(
            tie_word_embeddings=hybrid.model.tie_word_embeddings,
            **kwargs,
        )
        self.iq_hybrid = canonical
        self.iq_hybrid_sha256 = hybrid.fingerprint
        self.vocab_size = hybrid.model.vocab_size
        self.hidden_size = hybrid.model.hidden_size
        self.num_hidden_layers = len(hybrid.schedule.layers)
        self.num_attention_heads = hybrid.model.num_attention_heads
        self.num_key_value_heads = hybrid.model.num_key_value_heads
        self.max_position_embeddings = hybrid.model.max_position_embeddings

    def validated_iq_hybrid(self) -> IQHybridConfig:
        result = IQHybridConfig.from_dict(self.iq_hybrid)
        if result.fingerprint != self.iq_hybrid_sha256:
            raise IQHFConfigError("IQ HF config was modified without fingerprint update")
        return result


def register_iq_hf_config() -> None:
    """Register only AutoConfig; no nonexistent HF auto-model is invented."""
    AutoConfig.register(IQHFCausalConfig.model_type, IQHFCausalConfig)
