from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence
import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class BiasFilterError(RuntimeError):
    pass


@dataclass(frozen=True)
class BiasFilterGate:
    """Activation-level acceptance gate for a candidate concept eraser."""

    min_initial_leakage: float = 1e-6
    min_leakage_reduction: float = 0.80
    max_relative_mse: float = 0.01
    min_mean_cosine: float = 0.995

    def __post_init__(self) -> None:
        if self.min_initial_leakage < 0:
            raise BiasFilterError("min_initial_leakage must be non-negative")
        if not 0.0 <= self.min_leakage_reduction <= 1.0:
            raise BiasFilterError("min_leakage_reduction must be in [0, 1]")
        if self.max_relative_mse < 0:
            raise BiasFilterError("max_relative_mse must be non-negative")
        if not -1.0 <= self.min_mean_cosine <= 1.0:
            raise BiasFilterError("min_mean_cosine must be in [-1, 1]")


@dataclass(frozen=True)
class BiasFilterMetrics:
    leakage_before: float
    leakage_after: float
    leakage_reduction: float
    relative_mse: float
    mean_cosine: float
    protected_dimensions: int


@dataclass(frozen=True)
class CausalValidation:
    """Evidence that intervention changes the targeted asymmetry, not controls."""

    asymmetry_before: float
    asymmetry_after: float
    control_delta: float
    min_relative_reduction: float = 0.50
    max_control_delta: float = 0.02

    def require_valid(self) -> None:
        before = abs(float(self.asymmetry_before))
        after = abs(float(self.asymmetry_after))
        if before <= 0.0:
            raise BiasFilterError("causal validation requires non-zero baseline asymmetry")
        reduction = 1.0 - after / before
        if reduction < self.min_relative_reduction:
            raise BiasFilterError(
                f"causal asymmetry reduction {reduction:.6f} is below "
                f"required {self.min_relative_reduction:.6f}"
            )
        if abs(float(self.control_delta)) > self.max_control_delta:
            raise BiasFilterError(
                f"control delta {self.control_delta:.6f} exceeds "
                f"allowed {self.max_control_delta:.6f}"
            )


@dataclass(frozen=True)
class CapabilityMetricRule:
    name: str
    max_degradation: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise BiasFilterError("capability metric name must be non-empty")
        if self.max_degradation < 0:
            raise BiasFilterError("max_degradation must be non-negative")


@dataclass(frozen=True)
class CapabilityGateResult:
    degradations: Mapping[str, float]


@dataclass(frozen=True)
class BiasFilterApproval:
    artifact_fingerprint: str
    metrics: BiasFilterMetrics
    causal_validation: CausalValidation
    capability_gate: CapabilityGateResult


@dataclass(frozen=True)
class BiasFilterArtifact:
    """Compact low-rank LEACE projection scoped to selected feature dimensions."""

    feature_dim: int
    concept_dim: int
    target_concept: str
    space_name: str
    active_dimensions: torch.Tensor
    proj_left: torch.Tensor
    proj_right: torch.Tensor
    bias: torch.Tensor
    protected_modalities: tuple[str, ...] = ()
    required_capability_metrics: tuple[str, ...] = ("coding", "nlp")
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise BiasFilterError(f"unsupported bias-filter schema: {self.schema_version}")
        if self.feature_dim <= 0 or self.concept_dim <= 0:
            raise BiasFilterError("feature_dim and concept_dim must be positive")
        if not self.target_concept.strip() or not self.space_name.strip():
            raise BiasFilterError("target_concept and space_name must be non-empty")
        if not self.required_capability_metrics:
            raise BiasFilterError("required_capability_metrics must be non-empty")
        if any(not name.strip() for name in self.required_capability_metrics):
            raise BiasFilterError("required capability metric names must be non-empty")
        if len(set(self.required_capability_metrics)) != len(self.required_capability_metrics):
            raise BiasFilterError("required capability metric names must be unique")
        active = self.active_dimensions
        if active.ndim != 1 or active.dtype != torch.long:
            raise BiasFilterError("active_dimensions must be a rank-1 torch.long tensor")
        if active.numel() == 0:
            raise BiasFilterError("at least one active dimension is required")
        if int(active.min()) < 0 or int(active.max()) >= self.feature_dim:
            raise BiasFilterError("active_dimensions contain an out-of-range index")
        if torch.unique(active).numel() != active.numel():
            raise BiasFilterError("active_dimensions must be unique")
        active_dim = int(active.numel())
        if self.proj_left.ndim != 2 or self.proj_left.shape[0] != active_dim:
            raise BiasFilterError("proj_left has an invalid shape")
        if self.proj_right.ndim != 2 or self.proj_right.shape[1] != active_dim:
            raise BiasFilterError("proj_right has an invalid shape")
        if self.proj_left.shape[1] != self.proj_right.shape[0]:
            raise BiasFilterError("low-rank projection factors are incompatible")
        if tuple(self.bias.shape) != (active_dim,):
            raise BiasFilterError("bias has an invalid shape")
        for name, tensor in (
            ("proj_left", self.proj_left),
            ("proj_right", self.proj_right),
            ("bias", self.bias),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise BiasFilterError(f"{name} contains non-finite values")

    @property
    def protected_dimension_count(self) -> int:
        return self.feature_dim - int(self.active_dimensions.numel())

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.feature_dim:
            raise BiasFilterError(
                f"feature dimension mismatch: got {x.shape[-1]}, expected {self.feature_dim}"
            )
        if not bool(torch.isfinite(x).all()):
            raise BiasFilterError("cannot sanitize non-finite activations")

        index = self.active_dimensions.to(device=x.device)
        selected = x.index_select(-1, index)
        compute_dtype = (
            torch.float32
            if selected.dtype in {torch.float16, torch.bfloat16}
            else selected.dtype
        )
        selected_compute = selected.to(compute_dtype)
        left = self.proj_left.to(device=x.device, dtype=compute_dtype)
        right = self.proj_right.to(device=x.device, dtype=compute_dtype)
        bias = self.bias.to(device=x.device, dtype=compute_dtype)
        delta = selected_compute - bias
        scrubbed = selected_compute - (delta @ right.mH) @ left.mH

        result = x.clone()
        result.index_copy_(-1, index, scrubbed.to(dtype=x.dtype))
        return result

    @property
    def fingerprint(self) -> str:
        digest = sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "feature_dim": self.feature_dim,
                    "concept_dim": self.concept_dim,
                    "target_concept": self.target_concept,
                    "space_name": self.space_name,
                    "protected_modalities": list(self.protected_modalities),
                    "required_capability_metrics": list(self.required_capability_metrics),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for tensor in (
            self.active_dimensions,
            self.proj_left,
            self.proj_right,
            self.bias,
        ):
            value = tensor.detach().contiguous().cpu()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def write(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "active_dimensions": self.active_dimensions.detach().cpu(),
                "proj_left": self.proj_left.detach().cpu(),
                "proj_right": self.proj_right.detach().cpu(),
                "bias": self.bias.detach().cpu(),
            },
            str(root / "eraser.safetensors"),
        )
        metadata = {
            "schema_version": self.schema_version,
            "feature_dim": self.feature_dim,
            "concept_dim": self.concept_dim,
            "target_concept": self.target_concept,
            "space_name": self.space_name,
            "protected_modalities": list(self.protected_modalities),
            "required_capability_metrics": list(self.required_capability_metrics),
            "fingerprint": self.fingerprint,
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "BiasFilterArtifact":
        root = Path(path)
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BiasFilterError("bias-filter metadata is missing or invalid") from exc
        tensors = load_file(str(root / "eraser.safetensors"), device="cpu")
        required = {"active_dimensions", "proj_left", "proj_right", "bias"}
        if set(tensors) != required:
            raise BiasFilterError("bias-filter tensor artifact has an invalid schema")
        artifact = cls(
            feature_dim=int(metadata["feature_dim"]),
            concept_dim=int(metadata["concept_dim"]),
            target_concept=str(metadata["target_concept"]),
            space_name=str(metadata["space_name"]),
            active_dimensions=tensors["active_dimensions"].to(torch.long),
            proj_left=tensors["proj_left"],
            proj_right=tensors["proj_right"],
            bias=tensors["bias"],
            protected_modalities=tuple(str(x) for x in metadata.get("protected_modalities", ())),
            required_capability_metrics=tuple(
                str(x)
                for x in metadata.get(
                    "required_capability_metrics", ("coding", "nlp")
                )
            ),
            schema_version=int(metadata.get("schema_version", -1)),
        )
        if metadata.get("fingerprint") != artifact.fingerprint:
            raise BiasFilterError("bias-filter fingerprint mismatch")
        return artifact


def _require_leace_fitter():
    try:
        from concept_erasure import LeaceFitter
    except ImportError as exc:
        raise BiasFilterError(
            "concept-erasure>=0.2.1 is required to fit LEACE bias filters"
        ) from exc
    return LeaceFitter


def _flatten_samples(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise BiasFilterError("activation tensors must be at least rank-2")
    return x.reshape(-1, x.shape[-1])


def paired_counterfactual_batch(
    side_a: torch.Tensor,
    side_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair-center matched prompts so shared topic/language magnitude cancels."""

    if side_a.shape != side_b.shape:
        raise BiasFilterError("counterfactual activation tensors must have identical shapes")
    a = _flatten_samples(side_a)
    b = _flatten_samples(side_b)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise BiasFilterError("counterfactual activations contain non-finite values")
    midpoint = (a + b) * 0.5
    x = torch.cat((a - midpoint, b - midpoint), dim=0)
    z = torch.cat(
        (
            -torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype),
            torch.ones(b.shape[0], 1, device=b.device, dtype=b.dtype),
        ),
        dim=0,
    )
    return x, z


def detect_magnitude_outlier_dimensions(
    x: torch.Tensor,
    *,
    mad_threshold: float = 8.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Flag unusually large RMS dimensions using robust log-RMS MAD statistics."""

    if mad_threshold <= 0:
        raise BiasFilterError("mad_threshold must be positive")
    samples = _flatten_samples(x).to(torch.float64)
    if not bool(torch.isfinite(samples).all()):
        raise BiasFilterError("cannot detect outliers in non-finite activations")
    rms = samples.square().mean(dim=0).sqrt().clamp_min(eps)
    values = rms.log()
    median = values.median()
    mad = (values - median).abs().median()
    if float(mad) <= eps:
        return torch.zeros(values.shape, dtype=torch.bool, device=x.device)
    robust_z = 0.6744897501960817 * (values - median) / mad
    return robust_z.abs().gt(mad_threshold).to(device=x.device)


def _cross_covariance_norm(x: torch.Tensor, z: torch.Tensor) -> float:
    x2 = _flatten_samples(x).to(torch.float64)
    z2 = z.reshape(x2.shape[0], -1).to(device=x2.device, dtype=torch.float64)
    x2 = x2 - x2.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)
    denom = max(x2.shape[0] - 1, 1)
    return float(torch.linalg.matrix_norm(x2.mH @ z2 / denom).item())


def evaluate_filter(
    artifact: BiasFilterArtifact,
    x: torch.Tensor,
    z: torch.Tensor,
) -> BiasFilterMetrics:
    clean = artifact.apply(x)
    active = artifact.active_dimensions.to(device=x.device)
    editable_x = x.index_select(-1, active)
    editable_clean = clean.index_select(-1, active)
    before = _cross_covariance_norm(editable_x, z)
    after = _cross_covariance_norm(editable_clean, z)
    reduction = 0.0 if before <= 0.0 else 1.0 - after / before
    x64 = x.to(torch.float64)
    clean64 = clean.to(torch.float64)
    relative_mse = float(
        (clean64 - x64).square().mean().div(x64.square().mean().clamp_min(1e-30)).item()
    )
    mean_cosine = float(
        F.cosine_similarity(
            _flatten_samples(x64),
            _flatten_samples(clean64),
            dim=-1,
            eps=1e-12,
        ).mean().item()
    )
    return BiasFilterMetrics(
        leakage_before=before,
        leakage_after=after,
        leakage_reduction=reduction,
        relative_mse=relative_mse,
        mean_cosine=mean_cosine,
        protected_dimensions=artifact.protected_dimension_count,
    )


def require_activation_gate(metrics: BiasFilterMetrics, gate: BiasFilterGate) -> None:
    if metrics.leakage_before < gate.min_initial_leakage:
        raise BiasFilterError(
            "candidate concept is not sufficiently linearly present to justify erasure"
        )
    if metrics.leakage_reduction < gate.min_leakage_reduction:
        raise BiasFilterError(
            f"leakage reduction {metrics.leakage_reduction:.6f} is below "
            f"required {gate.min_leakage_reduction:.6f}"
        )
    if metrics.relative_mse > gate.max_relative_mse:
        raise BiasFilterError(
            f"relative activation MSE {metrics.relative_mse:.6f} exceeds "
            f"allowed {gate.max_relative_mse:.6f}"
        )
    if metrics.mean_cosine < gate.min_mean_cosine:
        raise BiasFilterError(
            f"mean activation cosine {metrics.mean_cosine:.6f} is below "
            f"required {gate.min_mean_cosine:.6f}"
        )


def fit_leace_bias_filter(
    fit_x: torch.Tensor,
    fit_z: torch.Tensor,
    *,
    validation_x: torch.Tensor,
    validation_z: torch.Tensor,
    target_concept: str,
    space_name: str,
    protected_dimensions: torch.Tensor | None = None,
    protected_modalities: Sequence[str] = (),
    required_capability_metrics: Sequence[str] = ("coding", "nlp"),
    statistics_dtype: torch.dtype = torch.float64,
    svd_tol: float = 0.01,
) -> tuple[BiasFilterArtifact, BiasFilterMetrics]:
    """Fit a shrinkage-LEACE candidate without approving it for transfer."""

    fit = _flatten_samples(fit_x)
    validation = _flatten_samples(validation_x)
    if fit.shape[1] != validation.shape[1]:
        raise BiasFilterError("fit and validation feature dimensions differ")
    feature_dim = int(fit.shape[1])
    fit_labels = fit_z.reshape(fit.shape[0], -1)
    validation_labels = validation_z.reshape(validation.shape[0], -1)
    if fit_labels.shape[1] != validation_labels.shape[1]:
        raise BiasFilterError("fit and validation concept dimensions differ")
    if not bool(torch.isfinite(fit).all()) or not bool(torch.isfinite(fit_labels).all()):
        raise BiasFilterError("fit data contains non-finite values")

    if protected_dimensions is None:
        protected = torch.zeros(feature_dim, dtype=torch.bool, device=fit.device)
    else:
        protected = protected_dimensions.to(device=fit.device, dtype=torch.bool)
        if tuple(protected.shape) != (feature_dim,):
            raise BiasFilterError("protected_dimensions must have shape [feature_dim]")
    active = torch.nonzero(~protected, as_tuple=False).flatten()
    if active.numel() == 0:
        raise BiasFilterError("all feature dimensions are protected")

    selected = fit.index_select(-1, active).to(dtype=statistics_dtype)
    labels = fit_labels.to(device=selected.device, dtype=statistics_dtype)
    LeaceFitter = _require_leace_fitter()
    fitter = LeaceFitter(
        int(active.numel()),
        int(labels.shape[1]),
        method="leace",
        affine=True,
        constrain_cov_trace=True,
        device=selected.device,
        dtype=statistics_dtype,
        shrinkage=True,
        svd_tol=svd_tol,
    )
    fitter.update(selected, labels)
    eraser = fitter.eraser
    if eraser.bias is None:
        raise BiasFilterError("LEACE unexpectedly returned a non-affine eraser")

    artifact = BiasFilterArtifact(
        feature_dim=feature_dim,
        concept_dim=int(labels.shape[1]),
        target_concept=target_concept,
        space_name=space_name,
        active_dimensions=active.detach().cpu().to(torch.long),
        proj_left=eraser.proj_left.detach().cpu(),
        proj_right=eraser.proj_right.detach().cpu(),
        bias=eraser.bias.detach().cpu(),
        protected_modalities=tuple(str(x) for x in protected_modalities),
        required_capability_metrics=tuple(
            str(x) for x in required_capability_metrics
        ),
    )
    metrics = evaluate_filter(artifact, validation_x, validation_z)
    return artifact, metrics


def validate_capability_metrics(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    rules: Sequence[CapabilityMetricRule],
) -> CapabilityGateResult:
    if not rules:
        raise BiasFilterError("at least one capability preservation rule is required")
    degradations: dict[str, float] = {}
    for rule in rules:
        if rule.name not in baseline or rule.name not in candidate:
            raise BiasFilterError(f"missing required capability metric: {rule.name}")
        base = float(baseline[rule.name])
        value = float(candidate[rule.name])
        degradation = base - value if rule.higher_is_better else value - base
        degradations[rule.name] = degradation
        if degradation > rule.max_degradation:
            raise BiasFilterError(
                f"capability metric {rule.name!r} degraded by {degradation:.6f}; "
                f"allowed {rule.max_degradation:.6f}"
            )
    return CapabilityGateResult(degradations=degradations)


def approve_bias_filter(
    artifact: BiasFilterArtifact,
    metrics: BiasFilterMetrics,
    *,
    activation_gate: BiasFilterGate,
    causal_validation: CausalValidation,
    baseline_capabilities: Mapping[str, float],
    filtered_capabilities: Mapping[str, float],
    capability_rules: Sequence[CapabilityMetricRule],
) -> BiasFilterApproval:
    """Fail closed unless erasure, causality, and capability retention all pass."""

    require_activation_gate(metrics, activation_gate)
    causal_validation.require_valid()
    provided_rules = {rule.name for rule in capability_rules}
    missing_required = sorted(
        set(artifact.required_capability_metrics) - provided_rules
    )
    if missing_required:
        raise BiasFilterError(
            "missing required capability preservation rules: "
            + ", ".join(missing_required)
        )
    capability_gate = validate_capability_metrics(
        baseline_capabilities,
        filtered_capabilities,
        capability_rules,
    )
    return BiasFilterApproval(
        artifact_fingerprint=artifact.fingerprint,
        metrics=metrics,
        causal_validation=causal_validation,
        capability_gate=capability_gate,
    )


def apply_approved_filter(
    artifact: BiasFilterArtifact,
    approval: BiasFilterApproval,
    x: torch.Tensor,
) -> torch.Tensor:
    if approval.artifact_fingerprint != artifact.fingerprint:
        raise BiasFilterError("approval does not match the supplied bias-filter artifact")
    return artifact.apply(x)


def sanitize_activation_pair(
    pair,
    artifact: BiasFilterArtifact,
    approval: BiasFilterApproval,
):
    """Apply an approved filter only to the donor/source side of a transfer pair."""

    from .calibration import ActivationPair

    if not isinstance(pair, ActivationPair):
        raise BiasFilterError("pair must be an ActivationPair")
    if pair.source_space != artifact.space_name:
        raise BiasFilterError(
            f"artifact space {artifact.space_name!r} does not match "
            f"activation-pair source space {pair.source_space!r}"
        )
    return ActivationPair(
        source_fit=apply_approved_filter(artifact, approval, pair.source_fit),
        target_fit=pair.target_fit,
        source_validation=apply_approved_filter(
            artifact, approval, pair.source_validation
        ),
        target_validation=pair.target_validation,
        source_space=pair.source_space,
        target_space=pair.target_space,
    )
