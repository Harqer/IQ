from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from torch import nn

from iq_model.mlp.latent_moe import StableLatentMoE, StableLatentMoEOutput


logger = logging.getLogger(__name__)


class QuantileBalancingError(RuntimeError):
    """The complete logical optimizer batch cannot be balanced correctly."""


class QuantileBalancingWindow:
    """Exact single-process Kimi-K3 QB statistics for one optimizer update.

    Hooks only observe genuine StableLatentMoE outputs while train_step runs.
    The old routing biases remain fixed throughout all gradient-accumulation
    microbatches. Proposals are computed before optimizer.step(), but are
    committed *only after* the optimizer update succeeds.

    K3 uses globally reduced histograms at distributed scale. An exact local
    quantile is not a valid silent replacement for a distributed global batch.
    """

    def __init__(self, model: nn.Module) -> None:
        self.layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, StableLatentMoE)
        }
        self._scores: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.layers
        }
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._mask: torch.Tensor | None = None
        self._in_microbatch = False
        self._tokens = 0

    @property
    def tokens(self) -> int:
        return self._tokens

    def __enter__(self) -> QuantileBalancingWindow:
        if self.layers and torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                raise QuantileBalancingError(
                    "distributed Stable LatentMoE needs global all-reduced "
                    "K3 histogram quantiles; local exact quantiles are not valid"
                )
        for name, module in self.layers.items():
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
                self._scores[name].append(scores.float())
                # All stable layers see the same logical tokens, so count once.
                if name == next(iter(self.layers)):
                    self._tokens += scores.shape[0]

        return record

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
            if not self._scores[name]:
                raise QuantileBalancingError(
                    f"no valid routing observations for layer {name}"
                )
            scores = torch.cat(self._scores[name], dim=0)
            candidate = module.compute_next_routing_bias(scores)
            if candidate.shape != module.routing_bias.shape or not bool(
                torch.isfinite(candidate).all()
            ):
                raise QuantileBalancingError(
                    f"non-finite or invalid QB proposal for layer {name}"
                )
            next_bias[name] = candidate
        return next_bias

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
