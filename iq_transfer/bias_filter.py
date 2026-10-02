from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


class BiasFilterError(RuntimeError):
    pass


@dataclass(frozen=True)
class BiasFilterConfig:
    """Conservative transfer-time linear concept filter.

    The fitter operates on calibration activations, never directly on donor
    weights. It uses robust centering/scaling, shrinkage covariance and a
    whiten-project-unwhiten erasure. Protected coordinates are excluded from
    the editable subspace so known super-activation/superweight channels are
    not modified by the filter.
    """

    shrinkage: float = 0.05
    mad_epsilon: float = 1e-6
    svd_rtol: float = 1e-6
    max_rank: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.shrinkage < 1.0:
            raise BiasFilterError("shrinkage must be in [0, 1)")
        if self.mad_epsilon <= 0.0:
            raise BiasFilterError("mad_epsilon must be positive")
        if self.svd_rtol <= 0.0:
            raise BiasFilterError("svd_rtol must be positive")
        if self.max_rank is not None and self.max_rank <= 0:
            raise BiasFilterError("max_rank must be positive when set")


@dataclass(frozen=True)
class BiasFilter:
    mean: torch.Tensor
    scale: torch.Tensor
    editable_indices: torch.Tensor
    basis: torch.Tensor
    whitening: torch.Tensor
    unwhitening: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def apply(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.shape[-1] != self.mean.numel():
            raise BiasFilterError(
                f"activation width {activations.shape[-1]} does not match filter width {self.mean.numel()}"
            )
        output = activations.clone()
        if self.rank == 0 or self.editable_indices.numel() == 0:
            return output

        idx = self.editable_indices.to(activations.device)
        mean = self.mean.to(device=activations.device, dtype=activations.dtype)[idx]
        scale = self.scale.to(device=activations.device, dtype=activations.dtype)[idx]
        whitening = self.whitening.to(
            device=activations.device, dtype=activations.dtype
        )
        unwhitening = self.unwhitening.to(
            device=activations.device, dtype=activations.dtype
        )
        basis = self.basis.to(device=activations.device, dtype=activations.dtype)

        x = (output[..., idx] - mean) / scale
        x_white = x @ whitening.T
        x_white = x_white - (x_white @ basis) @ basis.T
        output[..., idx] = (x_white @ unwhitening.T) * scale + mean
        return output


@dataclass(frozen=True)
class CapabilityGate:
    """Promotion gate for a fitted filter.

    Scores are normalized so higher is better. Bias/asymmetry is normalized so
    lower is better. A filter is promotable only when the requested asymmetry
    reduction is met without exceeding any retention budget.
    """

    min_bias_reduction: float = 0.30
    max_nlp_drop: float = 0.01
    max_coding_drop: float = 0.01
    max_multimodal_drop: float = 0.01

    def accepts(
        self,
        *,
        bias_before: float,
        bias_after: float,
        nlp_before: float,
        nlp_after: float,
        coding_before: float,
        coding_after: float,
        multimodal_before: float | None = None,
        multimodal_after: float | None = None,
    ) -> bool:
        if bias_before <= 0.0:
            return False
        bias_reduction = (bias_before - bias_after) / bias_before
        if bias_reduction < self.min_bias_reduction:
            return False
        if nlp_before - nlp_after > self.max_nlp_drop:
            return False
        if coding_before - coding_after > self.max_coding_drop:
            return False
        if (multimodal_before is None) != (multimodal_after is None):
            raise BiasFilterError("multimodal scores must be provided together")
        if multimodal_before is not None and multimodal_after is not None:
            if multimodal_before - multimodal_after > self.max_multimodal_drop:
                return False
        return True


def robust_feature_scale(
    x: torch.Tensor, *, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise BiasFilterError("expected activations shaped [samples, features]")
    median = x.median(dim=0).values
    mad = (x - median).abs().median(dim=0).values
    scale = (1.4826 * mad).clamp_min(epsilon)
    return median, scale


def _symmetric_whitener(
    covariance: torch.Tensor, *, rtol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(covariance)
    threshold = evals.max().clamp_min(torch.finfo(evals.dtype).eps) * rtol
    keep = evals > threshold
    if not torch.any(keep):
        raise BiasFilterError("activation covariance is numerically rank zero")
    kept_evals = evals[keep]
    kept_evecs = evecs[:, keep]
    whitening = (kept_evecs / kept_evals.sqrt()).T
    unwhitening = kept_evecs * kept_evals.sqrt()
    return whitening, unwhitening


def fit_bias_filter(
    activations: torch.Tensor,
    concept: torch.Tensor,
    *,
    protected_indices: Iterable[int] = (),
    config: BiasFilterConfig = BiasFilterConfig(),
) -> BiasFilter:
    """Fit a conservative linear concept eraser on calibration activations.

    `concept` may be one-dimensional labels or a dense concept matrix. The
    method follows LEACE's whitening -> concept-subspace projection ->
    unwhitening structure, while adding robust feature scaling and an explicit
    protected-coordinate contract for transfer-time use.
    """

    if activations.ndim != 2:
        raise BiasFilterError("activations must be [samples, features]")
    if concept.ndim == 1:
        concept = concept[:, None]
    if concept.ndim != 2 or concept.shape[0] != activations.shape[0]:
        raise BiasFilterError(
            "concept must have the same sample count as activations"
        )
    if activations.shape[0] < 2:
        raise BiasFilterError("at least two calibration samples are required")

    work = activations.detach().to(torch.float64)
    z = concept.detach().to(torch.float64)
    mean, scale = robust_feature_scale(work, epsilon=config.mad_epsilon)

    width = work.shape[1]
    protected = sorted(set(int(i) for i in protected_indices))
    if any(i < 0 or i >= width for i in protected):
        raise BiasFilterError("protected index outside activation width")
    protected_set = set(protected)
    editable = torch.tensor(
        [i for i in range(width) if i not in protected_set],
        dtype=torch.long,
    )
    if editable.numel() == 0:
        empty = torch.empty(0, 0, dtype=work.dtype)
        return BiasFilter(mean, scale, editable, empty, empty, empty)

    x = (work[:, editable] - mean[editable]) / scale[editable]
    x = x - x.mean(dim=0, keepdim=True)
    z = z - z.mean(dim=0, keepdim=True)
    n = x.shape[0]

    covariance = (x.T @ x) / (n - 1)
    trace_mean = torch.trace(covariance) / covariance.shape[0]
    covariance = (
        (1.0 - config.shrinkage) * covariance
        + config.shrinkage
        * trace_mean
        * torch.eye(covariance.shape[0], dtype=covariance.dtype)
    )
    cross_cov = (x.T @ z) / (n - 1)

    whitening, unwhitening = _symmetric_whitener(
        covariance, rtol=config.svd_rtol
    )
    concept_in_white = whitening @ cross_cov
    if torch.linalg.norm(concept_in_white) == 0:
        basis = torch.empty(whitening.shape[0], 0, dtype=work.dtype)
    else:
        u, s, _ = torch.linalg.svd(concept_in_white, full_matrices=False)
        threshold = s.max() * config.svd_rtol
        rank = int((s > threshold).sum().item())
        if config.max_rank is not None:
            rank = min(rank, config.max_rank)
        basis = u[:, :rank]

    return BiasFilter(
        mean=mean,
        scale=scale,
        editable_indices=editable,
        basis=basis,
        whitening=whitening,
        unwhitening=unwhitening,
    )
