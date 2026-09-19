from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class ApplyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParameterUpdate:
    target_parameter: str
    value: Any
    source_description: str


def _resolve_parameter(model: torch.nn.Module, path: str) -> torch.nn.Parameter:
    current: Any = model
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ApplyError(f"invalid parameter path: {path!r}")
    for part in parts[:-1]:
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as exc:
                raise ApplyError(f"cannot resolve parameter path {path!r} at index {part}") from exc
        else:
            if not hasattr(current, part):
                raise ApplyError(f"cannot resolve parameter path {path!r}: missing {part!r}")
            current = getattr(current, part)
    parameter = getattr(current, parts[-1], None)
    if not isinstance(parameter, torch.nn.Parameter):
        raise ApplyError(f"target is not a torch Parameter: {path!r}")
    return parameter


@torch.no_grad()
def apply_parameter_updates(
    model: torch.nn.Module,
    updates: list[ParameterUpdate] | tuple[ParameterUpdate, ...],
    *,
    require_unique: bool = True,
) -> tuple[str, ...]:
    if not updates:
        raise ApplyError("at least one parameter update is required")
    paths = [update.target_parameter for update in updates]
    if require_unique and len(paths) != len(set(paths)):
        raise ApplyError("parameter update bundle contains duplicate targets")

    staged: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    for update in updates:
        if not update.source_description.strip():
            raise ApplyError(f"source_description is empty for {update.target_parameter}")
        parameter = _resolve_parameter(model, update.target_parameter)
        value = update.value if isinstance(update.value, torch.Tensor) else torch.as_tensor(update.value)
        if tuple(value.shape) != tuple(parameter.shape):
            raise ApplyError(
                f"shape mismatch for {update.target_parameter}: got {tuple(value.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ApplyError(f"non-finite transported tensor for {update.target_parameter}")
        staged.append((update.target_parameter, parameter, value.to(parameter.device, parameter.dtype)))

    # Validate the complete bundle before mutating any model parameter.
    for _path, parameter, value in staged:
        parameter.copy_(value)
    return tuple(path for path, _parameter, _value in staged)
