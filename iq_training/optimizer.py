from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class OptimizerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = "match_rms_adamw"

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise OptimizerConfigError("lr must be positive")
        if self.weight_decay < 0:
            raise OptimizerConfigError("weight_decay must be non-negative")
        if len(self.adam_betas) != 2 or not all(0.0 <= x < 1.0 for x in self.adam_betas):
            raise OptimizerConfigError("adam_betas must contain two values in [0, 1)")
        if self.adam_eps <= 0:
            raise OptimizerConfigError("adam_eps must be positive")
        if not (0.0 <= self.muon_momentum < 1.0):
            raise OptimizerConfigError("muon_momentum must be in [0, 1)")
        if self.muon_ns_steps <= 0:
            raise OptimizerConfigError("muon_ns_steps must be positive")


@dataclass(frozen=True)
class OptimizerCoverage:
    muon: tuple[str, ...]
    adamw: tuple[str, ...]
    frozen: tuple[str, ...]

    @property
    def trainable(self) -> tuple[str, ...]:
        return self.muon + self.adamw


def _is_muon_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim != 2:
        return False
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return False
    if name.endswith(".magnitude"):
        return False
    return True


def classify_parameters(model: torch.nn.Module) -> OptimizerCoverage:
    muon: list[str] = []
    adamw: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(name)
        elif _is_muon_parameter(name, parameter):
            muon.append(name)
        else:
            adamw.append(name)

    trainable = muon + adamw
    if len(trainable) != len(set(trainable)):
        raise RuntimeError("optimizer parameter classification produced duplicate trainable names")
    expected = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if sorted(trainable) != sorted(expected):
        missing = sorted(set(expected) - set(trainable))
        extra = sorted(set(trainable) - set(expected))
        raise RuntimeError(f"optimizer coverage mismatch: missing={missing}, extra={extra}")
    return OptimizerCoverage(tuple(muon), tuple(adamw), tuple(frozen))


class IQOptimizer:
    """Muon for eligible hidden matrices plus AdamW for all remaining trainable parameters."""

    def __init__(
        self,
        *,
        muon: torch.optim.Optimizer | None,
        adamw: torch.optim.Optimizer | None,
        coverage: OptimizerCoverage,
    ) -> None:
        if muon is None and adamw is None:
            raise ValueError("at least one underlying optimizer is required")
        self.muon = muon
        self.adamw = adamw
        self.coverage = coverage

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("IQOptimizer does not support closures across two optimizers")
        muon_loss = self.muon.step() if self.muon is not None else None
        adam_loss = self.adamw.step() if self.adamw is not None else None
        return adam_loss if adam_loss is not None else muon_loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
            "coverage": {
                "muon": list(self.coverage.muon),
                "adamw": list(self.coverage.adamw),
                "frozen": list(self.coverage.frozen),
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("schema_version", -1)) != 1:
            raise ValueError("unsupported IQOptimizer state schema")
        expected = {
            "muon": list(self.coverage.muon),
            "adamw": list(self.coverage.adamw),
            "frozen": list(self.coverage.frozen),
        }
        if state_dict.get("coverage") != expected:
            raise ValueError("optimizer coverage changed since checkpoint creation")
        if self.muon is None:
            if state_dict.get("muon") is not None:
                raise ValueError("checkpoint contains Muon state but current optimizer does not")
        else:
            if state_dict.get("muon") is None:
                raise ValueError("checkpoint is missing Muon state")
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is None:
            if state_dict.get("adamw") is not None:
                raise ValueError("checkpoint contains AdamW state but current optimizer does not")
        else:
            if state_dict.get("adamw") is None:
                raise ValueError("checkpoint is missing AdamW state")
            self.adamw.load_state_dict(state_dict["adamw"])


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig = OptimizerConfig()) -> IQOptimizer:
    coverage = classify_parameters(model)
    by_name = dict(model.named_parameters())
    muon_params = [by_name[name] for name in coverage.muon]
    adam_params = [by_name[name] for name in coverage.adamw]

    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_params and muon_cls is None:
        raise RuntimeError("this IQ training path requires a PyTorch build that provides torch.optim.Muon")

    muon = (
        muon_cls(
            muon_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.muon_momentum,
            nesterov=config.muon_nesterov,
            ns_steps=config.muon_ns_steps,
            adjust_lr_fn=config.muon_adjust_lr_fn,
        )
        if muon_params
        else None
    )
    adamw = (
        torch.optim.AdamW(
            adam_params,
            lr=config.lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        if adam_params
        else None
    )
    return IQOptimizer(muon=muon, adamw=adamw, coverage=coverage)
