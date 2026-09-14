from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeystoneActivationSummary:
    task: str
    observations: int
    mean_abs: torch.Tensor


class KeystoneActivationMonitor:
    """Collect cross-task FFN activation strength without changing model behavior.

    This utility deliberately stops at measurement. The ICML 2026 keystone-neuron
    result identifies neurons from cross-task activation and then validates them via
    ablation. IQ should reproduce that identification/ablation protocol before any
    neuron is frozen, protected, or given a special optimization rule.
    """

    def __init__(self, intermediate_size: int) -> None:
        self.intermediate_size = int(intermediate_size)
        self._sum_abs: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._active_task: str | None = None

    def set_task(self, task: str | None) -> None:
        self._active_task = task

    @torch.no_grad()
    def observe(self, activations: torch.Tensor) -> None:
        if self._active_task is None:
            return
        if activations.shape[-1] != self.intermediate_size:
            raise ValueError(
                f"expected activation width {self.intermediate_size}, got {activations.shape[-1]}"
            )

        flat = activations.detach().float().reshape(-1, self.intermediate_size)
        batch_sum = flat.abs().sum(dim=0).cpu()
        task = self._active_task
        if task not in self._sum_abs:
            self._sum_abs[task] = torch.zeros(self.intermediate_size, dtype=torch.float64)
        self._sum_abs[task] += batch_sum.to(torch.float64)
        self._counts[task] += flat.shape[0]

    def summaries(self) -> tuple[KeystoneActivationSummary, ...]:
        result = []
        for task in sorted(self._sum_abs):
            count = self._counts[task]
            result.append(
                KeystoneActivationSummary(
                    task=task,
                    observations=count,
                    mean_abs=(self._sum_abs[task] / max(count, 1)).to(torch.float32),
                )
            )
        return tuple(result)

    def cross_task_mean_abs(self) -> torch.Tensor:
        summaries = self.summaries()
        if not summaries:
            raise RuntimeError("no activation observations have been collected")
        return torch.stack([summary.mean_abs for summary in summaries], dim=0).mean(dim=0)

    def reset(self) -> None:
        self._sum_abs.clear()
        self._counts.clear()
        self._active_task = None
