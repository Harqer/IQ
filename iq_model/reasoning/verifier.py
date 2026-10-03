from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True)
class ReasoningVerifierSignal:
    """Backend-neutral scalar evidence for a generated reasoning state.

    score is always higher-is-better regardless of verifier backend.
    Recurrence may consume the detached score as auxiliary halting evidence,
    but a verifier never owns or mutates the reasoning-state transition.
    """

    score: torch.Tensor
    backend: str
    raw_score: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.score.ndim == 0:
            raise ValueError("verifier score must include a batch dimension")
        if not bool(torch.isfinite(self.score).all()):
            raise RuntimeError("reasoning verifier produced non-finite scores")
        if self.raw_score is not None:
            if self.raw_score.shape != self.score.shape:
                raise ValueError("raw verifier score must match normalized score shape")
            if not bool(torch.isfinite(self.raw_score).all()):
                raise RuntimeError("reasoning verifier produced non-finite raw scores")
        if not self.backend.strip():
            raise ValueError("verifier backend name must be non-empty")


@runtime_checkable
class ReasoningVerifierProtocol(Protocol):
    """Minimal contract accepted by reasoning recurrence."""

    @property
    def state_dim(self) -> int: ...

    @property
    def context_dim(self) -> int: ...

    def verify(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> ReasoningVerifierSignal: ...


class ReasoningVerifier(nn.Module):
    """Base class for pluggable reasoning verifiers."""

    backend: str = "unknown"

    @property
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    def context_dim(self) -> int:
        raise NotImplementedError

    def verify(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> ReasoningVerifierSignal:
        raise NotImplementedError
