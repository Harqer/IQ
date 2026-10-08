from __future__ import annotations

import logging
from math import ceil
from collections.abc import Mapping

import torch
from torch import nn

from iq_model.mlp.latent_moe import StableLatentMoE, StableLatentMoEOutput


logger = logging.getLogger(__name__)


class QuantileBalancingError(RuntimeError):
    """The complete logical optimizer batch cannot be balanced correctly."""


class QuantileBalancingWindow:
    """Kimi-K3 exact or globally reduced histogram QB for one optimizer update.

    Hooks only observe genuine StableLatentMoE outputs while train_step runs.
    The old routing biases remain fixed throughout all gradient-accumulation
    microbatches. Proposals are computed before optimizer.step(), but are
    committed *only after* the optimizer update succeeds.

    K3 uses globally reduced histograms at distributed scale. An exact local
    quantile is not a valid silent replacement for a distributed global batch.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        backend: str = "exact",
        histogram_bins: int = 1000,
        process_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        if backend not in {"exact", "histogram"}:
            raise ValueError("QB backend must be exact or histogram")
        if histogram_bins <= 1:
            raise ValueError("QB histogram_bins must exceed 1")
        self.backend = backend
        self.histogram_bins = histogram_bins
        self.process_group = process_group
        self._global_tokens: int | None = None
        self.layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, StableLatentMoE)
        }
        self._scores: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.layers
        }
        self._histograms: dict[str, torch.Tensor] = {}
        self._ranges: dict[str, tuple[float, float]] = {}
        self._counts: dict[str, int] = {name: 0 for name in self.layers}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._mask: torch.Tensor | None = None
        self._in_microbatch = False
        self._tokens = 0

    @property
    def tokens(self) -> int:
        return self._tokens

    def __enter__(self) -> QuantileBalancingWindow:
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if self.process_group is not None and not distributed:
            raise QuantileBalancingError(
                "QB process_group provided but torch.distributed is not initialized"
            )
        if self.layers and distributed and torch.distributed.get_world_size() > 1:
            if self.backend == "exact":
                raise QuantileBalancingError(
                    "distributed Stable LatentMoE requires the histogram backend "
                    "with a global per-expert all-reduce; local quantiles are invalid"
                )
            if self.process_group is None:
                raise QuantileBalancingError(
                    "distributed K3 histogram requires an explicit process_group "
                    "matching the model's synchronized expert replicas"
                )
        for name, module in self.layers.items():
            if self.backend == "histogram":
                current = module.routing_bias.detach().float()
                self._ranges[name] = (
                    float(current.min()) - 1.0,
                    float(current.max()) + 1.0,
                )
                self._histograms[name] = torch.zeros(
                    module.config.num_experts,
                    self.histogram_bins,
                    device=current.device,
                    dtype=torch.int64,
                )
                # Even a rank with no valid tokens must enter the collective
                # once; an absent local histogram would deadlock other ranks.
            self._handles.append(
                module.register_forward_hook(self._capture(name))
            )
        return self

    def _capture(self, name: str):
        def record(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: StableLatentMoEOutput,
        ) -> None:
            if not self._in_microbatch:
                raise QuantileBalancingError(
                    "Stable LatentMoE forward outside balancing microbatch"
                )
            if not isinstance(output, StableLatentMoEOutput):
                raise QuantileBalancingError(
                    "Stable LatentMoE output did not expose routing scores"
                )
            scores = output.raw_router_scores.detach()
            if scores.ndim != 3:
                raise QuantileBalancingError(
                    "routing scores must have shape [batch, sequence, experts]"
                )
            if self._mask is not None:
                mask = self._mask.to(device=scores.device, dtype=torch.bool)
                if mask.shape != scores.shape[:2]:
                    raise QuantileBalancingError(
                        "routing mask does not match [batch, sequence]"
                    )
                scores = scores[mask]
            else:
                scores = scores.reshape(-1, scores.shape[-1])
            if scores.numel():
                if not bool(torch.isfinite(scores).all()):
                    raise QuantileBalancingError("router scores are not finite")
                scores = scores.float()
                if self.backend == "exact":
                    self._scores[name].append(scores)
                else:
                    self._accumulate_histogram(name, module, scores)
                # All stable layers see the same logical tokens, so count once.
                if name == next(iter(self.layers)):
                    self._tokens += scores.shape[0]

        return record

    @torch.no_grad()
    def _accumulate_histogram(
        self, name: str, module: StableLatentMoE, scores: torch.Tensor
    ) -> None:
        # Kimi K3 Appendix D: required bias r = alpha - raw_score lies in
        # [old_bias.min() - 1, old_bias.max() + 1], since scores are sigmoid.
        current = module.routing_bias.detach().float()
        lower = float(current.min()) - 1.0
        upper = float(current.max()) + 1.0
        if name not in self._ranges:
            self._ranges[name] = (lower, upper)
        elif self._ranges[name] != (lower, upper):
            raise QuantileBalancingError(
                "routing bias changed within the logical QB batch"
            )
        cutoff = (
            scores + current.unsqueeze(0)
        ).topk(module.config.top_k + 1, dim=-1).values[:, -1]
        required = cutoff.unsqueeze(-1) - scores
        if bool(((required < lower - 1e-5) | (required > upper + 1e-5)).any()):
            raise QuantileBalancingError("QB margin exceeded the published histogram range")
        bins = self.histogram_bins
        indices = (
            ((required - lower) / (upper - lower) * bins)
            .floor()
            .long()
            .clamp_(0, bins - 1)
        )
        expert_ids = torch.arange(
            scores.shape[1], device=scores.device
        ).expand(scores.shape[0], -1)
        flattened_indices = (expert_ids * bins + indices).reshape(-1)
        counts = torch.bincount(
            flattened_indices,
            minlength=scores.shape[1] * bins,
        ).reshape(scores.shape[1], bins).to(torch.int64)
        self._histograms[name].add_(counts)
        self._counts[name] += scores.shape[0]

    def begin_microbatch(self, mask: torch.Tensor | None) -> None:
        if self._in_microbatch:
            raise QuantileBalancingError("nested balancing microbatches")
        self._mask = mask
        self._in_microbatch = True

    def end_microbatch(self) -> None:
        self._mask = None
        self._in_microbatch = False

    @torch.no_grad()
    def propose(self) -> dict[str, torch.Tensor]:
        if self._in_microbatch:
            raise QuantileBalancingError("cannot propose during an active forward")
        next_bias: dict[str, torch.Tensor] = {}
        for name, module in self.layers.items():
            if self.backend == "exact":
                if not self._scores[name]:
                    raise QuantileBalancingError(
                        f"no valid routing observations for layer {name}"
                    )
                scores = torch.cat(self._scores[name], dim=0)
                candidate = module.compute_next_routing_bias(scores)
            else:
                candidate = self._histogram_proposal(name, module)
            if candidate.shape != module.routing_bias.shape or not bool(
                torch.isfinite(candidate).all()
            ):
                raise QuantileBalancingError(
                    f"non-finite or invalid QB proposal for layer {name}"
                )
            next_bias[name] = candidate
        return next_bias

    @torch.no_grad()
    def _histogram_proposal(
        self, name: str, module: StableLatentMoE
    ) -> torch.Tensor:
        if name not in self._histograms:
            raise QuantileBalancingError(
                f"no routing histogram available for layer {name}"
            )
        counts = self._histograms[name].clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                # The single count all-reduce implements Kimi K3 Appendix D.
                # All ranks must start with synchronized routing biases.
                torch.distributed.all_reduce(
                    counts,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.process_group,
                )
        totals = counts.sum(dim=1)
        if bool((totals == 0).any()) or not bool(torch.equal(totals, totals[0].expand_as(totals))):
            raise QuantileBalancingError(
                "QB histogram experts have inconsistent global token counts"
            )
        total = int(totals[0])
        self._global_tokens = total
        target = total * module.config.top_k / module.config.num_experts
        cumulative = counts.cumsum(dim=1)
        selected = (cumulative >= ceil(target)).to(torch.int64).argmax(dim=1)
        selected_count = counts.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        prefix_count = cumulative.gather(
            1, (selected - 1).clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        prefix_count = torch.where(selected > 0, prefix_count, torch.zeros_like(prefix_count))
        if bool((selected_count == 0).any()):
            raise QuantileBalancingError("QB quantile bin is empty")
        fraction = (
            (target - prefix_count.float()) / selected_count.float()
        ).clamp(0.0, 1.0)
        lower, upper = self._ranges[name]
        width = (upper - lower) / self.histogram_bins
        estimated = lower + (selected.float() + fraction) * width
        return estimated - estimated.mean()

    @torch.no_grad()
    def commit(self, proposals: Mapping[str, torch.Tensor]) -> None:
        if proposals.keys() != self.layers.keys():
            raise QuantileBalancingError("QB proposals do not cover every layer")
        for name, module in self.layers.items():
            proposed = proposals[name]
            if proposed.shape != module.routing_bias.shape or not bool(
                torch.isfinite(proposed).all()
            ):
                raise QuantileBalancingError(f"invalid QB proposal for layer {name}")
        for name, module in self.layers.items():
            module.commit_routing_bias(proposals[name])
        if self.layers:
            logger.info(
                "quantile_balancing_committed",
                extra={
                    "event": "quantile_balancing_committed",
                    "moe_layers": len(self.layers),
                    "valid_tokens": self._tokens,
                    "global_valid_tokens": self._global_tokens,
                    "backend": self.backend,
                },
            )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._in_microbatch = False
        self._mask = None
        for entries in self._scores.values():
            entries.clear()
        self._histograms.clear()
        self._ranges.clear()
        self._counts.clear()
        if exc_type is not None and self.layers:
            logger.warning(
                "quantile_balancing_discarded",
                extra={
                    "event": "quantile_balancing_discarded",
                    "moe_layers": len(self.layers),
                    "valid_tokens": self._tokens,
                    "reason": exc_type.__name__,
                },
            )
