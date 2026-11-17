"""Lazy public API: importing this package must not load unrelated model donors."""
from __future__ import annotations

from importlib import import_module

# Preserve existing exported names without eagerly importing every pipeline.
_MODULE_EXPORTS = {
    "pretraining": (
        "IQPretrainingModel", "PretrainingConfigError", "PretrainingObjectiveConfig", "PretrainingOutput",
    ),
    "checkpoint": (
        "CheckpointError", "CheckpointMetadata", "load_checkpoint", "save_checkpoint",
    ),
    "optimizer": (
        "IQOptimizer", "OptimizerConfig", "OptimizerConfigError", "OptimizerCoverage", "build_optimizer",
        "classify_parameters",
    ),
    "train": (
        "TrainingError", "TrainStepConfig", "TrainStepMetrics", "train_step",
    ),
    "mohawk": (
        "IQGLM53MohawkStage2", "MohawkStage2Output", "MohawkTrainingError", "freeze_for_glm53_stage2",
    ),
}

__all__ = [
    "CheckpointError", "IQGLM53MohawkStage2", "MohawkStage2Output", "MohawkTrainingError", "freeze_for_glm53_stage2",
    "CheckpointMetadata", "IQOptimizer", "IQPretrainingModel", "OptimizerConfig", "PretrainingConfigError",
    "PretrainingObjectiveConfig", "PretrainingOutput", "OptimizerConfigError", "OptimizerCoverage", "TrainStepConfig",
    "TrainStepMetrics", "TrainingError", "build_optimizer", "classify_parameters", "load_checkpoint",
    "save_checkpoint", "train_step",
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
