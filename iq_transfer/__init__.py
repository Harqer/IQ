"""Lazy public API: importing this package must not load unrelated model donors."""
from __future__ import annotations

from importlib import import_module

# Preserve existing exported names without eagerly importing every pipeline.
_MODULE_EXPORTS = {
    "alignment": (
        "AlignmentError", "ByteSpan", "CausalTokenAlignment", "PairedSpanActivations", "TokenByteSpan",
        "align_glm_states_to_mamba_tokens", "align_token_activations_by_bytes", "pool_activations_by_byte_spans",
    ),
    "apply": (
        "ApplyError", "ParameterUpdate", "apply_parameter_updates",
    ),
    "batches": (
        "BatchArtifactError", "TokenBatchArtifact", "load_token_batches", "save_token_batches",
    ),
    "calibration": (
        "ActivationPair", "CalibrationError", "CalibrationManifest", "CalibrationRecord", "CalibrationSplit",
        "LayerCorrespondence", "PhiLayerCalibration", "PhiLayerMapSolution", "merge_coordinate_maps", "solve_activation_pair",
        "solve_layer_correspondence", "solve_phi_layer_maps",
    ),
    "capture": (
        "ActivationTap", "CaptureError", "TorchActivationCapture", "load_capture_records", "save_capture_records",
    ),
    "capture_runner": (
        "ActivationBundle", "CaptureRunnerError", "GLM53CaptureLayout", "PhiCaptureLayout", "build_phi_layer_calibration_from_bundles",
        "capture_glm53_activations", "capture_iq_activations", "capture_iq_hybrid_residuals", "capture_phi_activations", "glm53_capture_taps",
        "iq_capture_taps", "iq_hybrid_residual_taps", "load_activation_bundle", "load_local_glm53_causal_lm", "load_local_phi_causal_lm",
        "make_activation_pair", "phi_capture_taps", "save_activation_bundle",
    ),
    "checkpoint": (
        "SafetensorsSource",
    ),
    "complete_transplant": (
        "RESIDUAL_EMBED_SCALE", "CompleteTransplantError", "canonical_complete_config", "canonical_complete_schedule", "canonical_glm53_config",
        "canonical_glm53_multimodal_config", "compile_complete_iq_checkpoint", "donor_layer_positions", "load_complete_iq_checkpoint",
    ),
    "donor": (
        "DonorConfig", "DonorError", "DonorInspector", "LayerRef", "MappingTensorSource",
        "OperatorRef", "TensorSource", "ValidationReport",
    ),
    "executor": (
        "DonorRuntime", "ExecutionError", "TransportExecutionReport", "execute_transport_plan",
    ),
    "glm53": (
        "GLM53_BF16_REPO", "GLM53_FLASH_BF16_REPO", "GLM53Inspector", "GLM53Layout",
    ),
    "glm53_calibration": (
        "GLM53CalibrationError", "GLM53CalibrationSolution", "GLM53StageCalibration", "bootstrap_glm53_calibration", "solve_glm53_calibration",
    ),
    "glm53_compile": (
        "GLM53CompileError", "GLM53CompileResult", "compile_glm53_iq_checkpoint",
    ),
    "glm53_direct": (
        "GLM53DirectTransformError", "GLM53MLALayout", "GLM53MLATransform", "SubcloningMap", "fit_importance_subcloning_map",
        "fit_mla_compressed_subspace", "fit_orthogonal_subspace", "fit_weight_importance_subcloning_map", "grouped_output_identity", "transform_embedding_and_lm_head",
        "transform_glm53_mla",
    ),
    "glm53_dsa": (
        "GLM53DSATransform", "GLM53DSATransformError", "resolve_indexer_source_layer", "transform_glm53_dsa_indexer",
    ),
    "glm53_job": (
        "GLM53DonorArtifact", "GLM53TransferError", "validate_glm53_donor", "validate_glm53_streaming_donor",
    ),
    "glm53_moe": (
        "ExpertWeights", "GLM53DenseMLPTransform", "GLM53MoETransform", "GLM53MoETransformError", "latent_codec_weights",
        "router_usage_from_topk", "router_weight_prior", "select_experts_by_usage", "transform_glm53_dense_mlp", "transform_glm53_moe",
        "transform_glm53_moe_selected",
    ),
    "glm53_policy": (
        "GLM53_SOURCE_POLICIES", "IQ_RECIPIENT_NATIVE_FAMILIES", "GLM53SourcePolicy", "GLM53TransferDisposition", "classify_glm53_source_role",
    ),
    "glm53_shards": (
        "GLM53ShardPlanError", "load_safetensors_weight_map", "plan_glm53_bootstrap_shards", "plan_glm53_compile_shards",
    ),
    "gpt_oss20b": (
        "GPT_OSS_20B_ORIGINAL_SHA256", "GPT_OSS_20B_REPO", "GPT_OSS_20B_REVISION", "GptOss20BConfig", "GptOss20BError",
        "GptOss20BOriginalCheckpoint",
    ),
    "gqa_transport": (
        "GQAProjection", "GQATransportError", "fit_gqa_group_projection",
    ),
    "job": (
        "PhiTransferJobResult", "TransferJobError", "hash_tokenizer_files", "load_transferred_iq_artifact", "run_phi_dense_transfer",
        "run_phi_dense_transfer_loaded",
    ),
    "mamba3_direct": (
        "MAMBA3_MIMO_15B_BIN_SHA256", "MAMBA3_MIMO_15B_REPO", "MAMBA3_MIMO_15B_REVISION", "Mamba3DirectTransferError", "Mamba3DirectTransferResult",
        "Mamba3DonorConfig", "Mamba3LayerPlacement", "Mamba3OverlayApplyReport", "apply_mamba3_transplant_overlay", "compile_official_mamba3_mimo_15b_transplant",
        "evenly_spaced_layer_placements", "expand_mamba3_layer", "validate_official_mamba3_mimo_15b_config", "validate_official_mamba3_mimo_state",
    ),
    "mamba3_init": (
        "TRANSFORMER_TO_MAMBA3", "Mamba3BootstrapReport", "Mamba3BootstrapWeights", "Mamba3InitError", "Mamba3Layout",
        "apply_mamba3_bootstrap",
    ),
    "manifest": (
        "CheckpointFile", "DonorManifest", "ManifestError", "TensorInventoryItem", "build_donor_manifest",
    ),
    "mohawk": (
        "MohawkConfig", "MohawkStage", "hidden_alignment_loss", "logits_distillation_loss", "matrix_orientation_loss",
        "mohawk_loss",
    ),
    "phi4": (
        "Phi4Inspector",
    ),
    "phi_pipeline": (
        "PhiDensePlanSpec", "PhiLayerMapIds", "PhiPipelineError", "build_phi_dense_plan_spec", "build_phi_dense_transport_plan",
    ),
    "plan": (
        "PlanError", "TransportPlan",
    ),
    "provenance": (
        "ParameterProvenance", "ProvenanceError", "ProvenanceLedger",
    ),
    "scaling": (
        "ScaleGate", "TransferMetrics",
    ),
    "shadows": (
        "FunctionalShadow", "MeasurementPlan", "ShadowError", "extract_shadow", "match_layers_monotonic",
        "shadow_distance",
    ),
    "slots": (
        "SlotError", "TargetAssignment", "TargetRegistry", "TargetSlot", "TransferMethod",
    ),
    "transport": (
        "CoordinateMap", "MapDiagnostics", "TransportError", "channel_correlation_cost", "fit_ot_coordinate_map",
        "fit_ridge_coordinate_map", "load_coordinate_map", "save_coordinate_map", "sinkhorn_transport", "transport_linear",
    ),
    "warm": (
        "WarmRemap", "WarmRemapDiagnostics", "WarmRemapError", "WarmWeightOperator", "fit_weight_orthogonal_remap",
        "orthogonality_error",
    ),
}

__all__ = [
    "GLM53_BF16_REPO", "GLM53_FLASH_BF16_REPO", "GLM53_SOURCE_POLICIES", "GPT_OSS_20B_ORIGINAL_SHA256", "GPT_OSS_20B_REPO",
    "GPT_OSS_20B_REVISION", "IQ_RECIPIENT_NATIVE_FAMILIES", "MAMBA3_MIMO_15B_BIN_SHA256", "MAMBA3_MIMO_15B_REPO", "MAMBA3_MIMO_15B_REVISION",
    "RESIDUAL_EMBED_SCALE", "TRANSFORMER_TO_MAMBA3", "ActivationBundle", "ActivationPair", "ActivationTap",
    "AlignmentError", "ApplyError", "BatchArtifactError", "ByteSpan", "CalibrationError",
    "CalibrationManifest", "CalibrationRecord", "CalibrationSplit", "CaptureError", "CaptureRunnerError",
    "CausalTokenAlignment", "CheckpointFile", "CompleteTransplantError", "CoordinateMap", "DonorConfig",
    "DonorError", "DonorInspector", "DonorManifest", "DonorRuntime", "ExecutionError",
    "ExpertWeights", "FunctionalShadow", "GLM53CalibrationError", "GLM53CalibrationSolution", "GLM53CaptureLayout",
    "GLM53CompileError", "GLM53CompileResult", "GLM53DSATransform", "GLM53DSATransformError", "GLM53DenseMLPTransform",
    "GLM53DirectTransformError", "GLM53DonorArtifact", "GLM53Inspector", "GLM53Layout", "GLM53MLALayout",
    "GLM53MLATransform", "GLM53MoETransform", "GLM53MoETransformError", "GLM53ShardPlanError", "GLM53SourcePolicy",
    "GLM53StageCalibration", "GLM53TransferDisposition", "GLM53TransferError", "GQAProjection", "GQATransportError",
    "GptOss20BConfig", "GptOss20BError", "GptOss20BOriginalCheckpoint", "LayerCorrespondence", "LayerRef",
    "Mamba3BootstrapReport", "Mamba3BootstrapWeights", "Mamba3DirectTransferError", "Mamba3DirectTransferResult", "Mamba3DonorConfig",
    "Mamba3InitError", "Mamba3LayerPlacement", "Mamba3Layout", "Mamba3OverlayApplyReport", "ManifestError",
    "MapDiagnostics", "MappingTensorSource", "MeasurementPlan", "MohawkConfig", "MohawkStage",
    "OperatorRef", "PairedSpanActivations", "ParameterProvenance", "ParameterUpdate", "Phi4Inspector",
    "PhiCaptureLayout", "PhiDensePlanSpec", "PhiLayerCalibration", "PhiLayerMapIds", "PhiLayerMapSolution",
    "PhiPipelineError", "PhiTransferJobResult", "PlanError", "ProvenanceError", "ProvenanceLedger",
    "SafetensorsSource", "ScaleGate", "ShadowError", "SlotError", "SubcloningMap",
    "TargetAssignment", "TargetRegistry", "TargetSlot", "TensorInventoryItem", "TensorSource",
    "TokenBatchArtifact", "TokenByteSpan", "TorchActivationCapture", "TransferJobError", "TransferMethod",
    "TransferMetrics", "TransportError", "TransportExecutionReport", "TransportPlan", "ValidationReport",
    "WarmRemap", "WarmRemapDiagnostics", "WarmRemapError", "WarmWeightOperator", "align_glm_states_to_mamba_tokens",
    "align_token_activations_by_bytes", "apply_mamba3_bootstrap", "apply_mamba3_transplant_overlay", "apply_parameter_updates", "bootstrap_glm53_calibration",
    "build_donor_manifest", "build_phi_dense_plan_spec", "build_phi_dense_transport_plan", "build_phi_layer_calibration_from_bundles", "canonical_complete_config",
    "canonical_complete_schedule", "canonical_glm53_config", "canonical_glm53_multimodal_config", "capture_glm53_activations", "capture_iq_activations",
    "capture_iq_hybrid_residuals", "capture_phi_activations", "channel_correlation_cost", "classify_glm53_source_role", "compile_complete_iq_checkpoint",
    "compile_glm53_iq_checkpoint", "compile_official_mamba3_mimo_15b_transplant", "donor_layer_positions", "evenly_spaced_layer_placements", "execute_transport_plan",
    "expand_mamba3_layer", "extract_shadow", "fit_gqa_group_projection", "fit_importance_subcloning_map", "fit_mla_compressed_subspace",
    "fit_orthogonal_subspace", "fit_ot_coordinate_map", "fit_ridge_coordinate_map", "fit_weight_importance_subcloning_map", "fit_weight_orthogonal_remap",
    "glm53_capture_taps", "grouped_output_identity", "hash_tokenizer_files", "hidden_alignment_loss", "iq_capture_taps",
    "iq_hybrid_residual_taps", "latent_codec_weights", "load_activation_bundle", "load_capture_records", "load_complete_iq_checkpoint",
    "load_coordinate_map", "load_local_glm53_causal_lm", "load_local_phi_causal_lm", "load_safetensors_weight_map", "load_token_batches",
    "load_transferred_iq_artifact", "logits_distillation_loss", "make_activation_pair", "match_layers_monotonic", "matrix_orientation_loss",
    "merge_coordinate_maps", "mohawk_loss", "orthogonality_error", "phi_capture_taps", "plan_glm53_bootstrap_shards",
    "plan_glm53_compile_shards", "pool_activations_by_byte_spans", "resolve_indexer_source_layer", "router_usage_from_topk", "router_weight_prior",
    "run_phi_dense_transfer", "run_phi_dense_transfer_loaded", "save_activation_bundle", "save_capture_records", "save_coordinate_map",
    "save_token_batches", "select_experts_by_usage", "shadow_distance", "sinkhorn_transport", "solve_activation_pair",
    "solve_glm53_calibration", "solve_layer_correspondence", "solve_phi_layer_maps", "transform_embedding_and_lm_head", "transform_glm53_dense_mlp",
    "transform_glm53_dsa_indexer", "transform_glm53_mla", "transform_glm53_moe", "transform_glm53_moe_selected", "transport_linear",
    "validate_glm53_donor", "validate_glm53_streaming_donor", "validate_official_mamba3_mimo_15b_config", "validate_official_mamba3_mimo_state",
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
