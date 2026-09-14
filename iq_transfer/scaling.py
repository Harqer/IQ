from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferMetrics:
    donor_score: float
    recipient_score: float
    scratch_score: float
    adaptation_compute: float
    scratch_compute: float

    @property
    def retention(self) -> float:
        return 0.0 if self.donor_score == 0 else self.recipient_score / self.donor_score

    @property
    def gain_over_scratch(self) -> float:
        return self.recipient_score - self.scratch_score

    @property
    def compute_ratio(self) -> float:
        return float("inf") if self.scratch_compute == 0 else self.adaptation_compute / self.scratch_compute


@dataclass(frozen=True)
class ScaleGate:
    min_retention: float = 0.80
    min_gain_over_scratch: float = 0.0
    max_compute_ratio: float = 0.50

    def passes(self, metrics: TransferMetrics) -> bool:
        return (
            metrics.retention >= self.min_retention
            and metrics.gain_over_scratch >= self.min_gain_over_scratch
            and metrics.compute_ratio <= self.max_compute_ratio
        )
