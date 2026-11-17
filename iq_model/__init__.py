"""Lazy public API: importing this package must not load unrelated model donors."""
from __future__ import annotations

from importlib import import_module

# Preserve existing exported names without eagerly importing every pipeline.
_MODULE_EXPORTS = {
    "adapters": (
        "DoRALinear", "install_dora",
    ),
    "architecture": (
        "ArchitectureError", "HybridLayerType", "HybridSchedule",
    ),
    "attention": (
        "CompressedAttentionCache", "CompressedContextConfig", "CompressedContextError", "CompressedSparseContextAttention", "DenseContextAttention",
        "GroupedLowRankOutput", "GroupedQueryAttention", "HeavilyCompressedContextAttention", "IndexerSegmentScores",
    ),
    "config": (
        "IQConfigError", "IQModelConfig",
    ),
    "energy": (
        "ReasoningEnergyCritic", "ReasoningEnergyCriticConfig", "energy_margin_ranking_loss",
    ),
    "hybrid": (
        "CompressedContextResidualLayer", "DenseContextResidualLayer", "HybridCausalLMOutput", "HybridModelError", "IQHybridConfig",
        "IQHybridForCausalLM", "Mamba3ResidualLayer", "MambaPackedLayout", "pack_mamba_varlen", "unpack_mamba_varlen",
        "validate_canonical_hybrid_backbone",
    ),
    "mlp": (
        "MoEConfigError", "MoEOutput", "RoutedMoEConfig", "RoutedSwiGLUMoE", "RoutedSwiGLUMoELayer",
        "SiTUAndMul", "SiTUExpert", "StableLatentMoE", "StableLatentMoEConfig", "StableLatentMoEError",
        "StableLatentMoELayer", "StableLatentMoEOutput", "SwiGLU", "SwiGLUExpert",
    ),
    "model": (
        "IQDecoderBlock", "IQForCausalLM",
    ),
    "norm": (
        "HeadRMSNorm", "RMSNorm", "UnweightedRMSNorm",
    ),
    "objectives": (
        "IndexerObjectiveError", "MTPConfig", "MTPConfigError", "MTPDepthOutput", "MTPOutput",
        "MTPPredictionBlock", "MultiTokenPrediction", "lightning_indexer_kl_loss", "lightning_indexer_topk_recall",
    ),
    "outputs": (
        "CausalLMOutput",
    ),
    "reasoning": (
        "AdaptiveHaltingHead", "ReasoningRecurrence", "ReasoningRecurrenceConfig", "ReasoningRecurrenceOutput", "ReasoningStateInjector",
        "ReasoningStateTransition", "ReasoningVerifier", "ReasoningVerifierProtocol", "ReasoningVerifierSignal", "SpectralDepthEncoding",
    ),
    "residual": (
        "AttentionResidualMixer", "AttnResError", "BlockAttentionResidual", "BlockAttnResConfig", "BlockAttnResState",
        "ManifoldHyperConnection", "MHCConfig", "MHCError", "MHCHead", "MHCWeights",
        "expand_mhc_streams", "sinkhorn_doubly_stochastic",
    ),
    "state": (
        "MAMBA3_UPSTREAM_COMMIT", "Mamba3MIMOConfig", "Mamba3MIMORuntimeError", "Mamba3MIMORuntimeInfo", "Mamba3MIMOState",
        "inspect_mamba3_mimo_runtime", "recommended_mamba3_chunk_size", "require_mamba3_mimo_runtime",
    ),
    "multimodal": (
        "IQMultimodalConfig", "IQMultimodalPathway", "MultimodalError", "VisualMemory",
    ),
}

__all__ = [
    "MAMBA3_UPSTREAM_COMMIT", "AdaptiveHaltingHead", "ArchitectureError", "AttentionResidualMixer", "AttnResError",
    "BlockAttentionResidual", "BlockAttnResConfig", "BlockAttnResState", "CausalLMOutput", "CompressedAttentionCache",
    "CompressedContextConfig", "CompressedContextError", "CompressedContextResidualLayer", "CompressedSparseContextAttention", "DenseContextAttention",
    "DenseContextResidualLayer", "DoRALinear", "GroupedLowRankOutput", "GroupedQueryAttention", "HeadRMSNorm",
    "HeavilyCompressedContextAttention", "HybridCausalLMOutput", "HybridLayerType", "HybridModelError", "HybridSchedule",
    "IQConfigError", "IQDecoderBlock", "IQForCausalLM", "IQHybridConfig", "IQHybridForCausalLM",
    "IQModelConfig", "IQMultimodalConfig", "IQMultimodalPathway", "IndexerObjectiveError", "IndexerSegmentScores",
    "MHCConfig", "MHCError", "MHCHead", "MHCWeights", "MTPConfig",
    "MTPConfigError", "MTPDepthOutput", "MTPOutput", "MTPPredictionBlock", "Mamba3MIMOConfig",
    "Mamba3MIMORuntimeError", "Mamba3MIMORuntimeInfo", "Mamba3MIMOState", "Mamba3ResidualLayer", "MambaPackedLayout",
    "ManifoldHyperConnection", "MoEConfigError", "MoEOutput", "MultiTokenPrediction", "MultimodalError",
    "RMSNorm", "ReasoningEnergyCritic", "ReasoningEnergyCriticConfig", "ReasoningRecurrence", "ReasoningRecurrenceConfig",
    "ReasoningRecurrenceOutput", "ReasoningStateInjector", "ReasoningStateTransition", "ReasoningVerifier", "ReasoningVerifierProtocol",
    "ReasoningVerifierSignal", "RoutedMoEConfig", "RoutedSwiGLUMoE", "RoutedSwiGLUMoELayer", "SiTUAndMul",
    "SiTUExpert", "SpectralDepthEncoding", "StableLatentMoE", "StableLatentMoEConfig", "StableLatentMoEError",
    "StableLatentMoELayer", "StableLatentMoEOutput", "SwiGLU", "SwiGLUExpert", "UnweightedRMSNorm",
    "VisualMemory", "energy_margin_ranking_loss", "expand_mhc_streams", "inspect_mamba3_mimo_runtime", "install_dora",
    "lightning_indexer_kl_loss", "lightning_indexer_topk_recall", "pack_mamba_varlen", "recommended_mamba3_chunk_size", "require_mamba3_mimo_runtime",
    "sinkhorn_doubly_stochastic", "unpack_mamba_varlen", "validate_canonical_hybrid_backbone",
]

_SYMBOL_MODULE = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}


def __getattr__(name: str):
    module = _SYMBOL_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
