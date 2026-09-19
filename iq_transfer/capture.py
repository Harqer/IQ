from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActivationTap:
    name: str
    module_path: str
    capture: Literal["input", "output"] = "output"
    tensor_index: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.module_path.strip():
            raise CaptureError("activation tap name and module_path must be non-empty")
        if self.tensor_index is not None and self.tensor_index < 0:
            raise CaptureError("tensor_index must be non-negative")


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise CaptureError("PyTorch is required for runtime activation capture") from exc
    return torch


def _resolve_module(model: Any, path: str) -> Any:
    current = model
    for part in path.split("."):
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as exc:
                raise CaptureError(f"cannot resolve module path {path!r} at index {part}") from exc
        else:
            if not hasattr(current, part):
                raise CaptureError(f"cannot resolve module path {path!r}: missing {part!r}")
            current = getattr(current, part)
    if not hasattr(current, "register_forward_hook"):
        raise CaptureError(f"resolved object is not a torch module: {path!r}")
    return current


def _select_tensor(value: Any, index: int | None) -> Any:
    if index is not None:
        if not isinstance(value, (tuple, list)):
            raise CaptureError("tensor_index was set but captured value is not a sequence")
        if index >= len(value):
            raise CaptureError(f"tensor_index {index} is out of range for captured sequence length {len(value)}")
        value = value[index]
    elif isinstance(value, (tuple, list)):
        tensors = [item for item in value if hasattr(item, "detach")]
        if len(tensors) != 1:
            raise CaptureError("captured sequence has multiple tensors; set tensor_index explicitly")
        value = tensors[0]
    if not hasattr(value, "detach"):
        raise CaptureError(f"captured value is not a tensor: {type(value).__name__}")
    return value


class TorchActivationCapture:
    """Forward-hook activation capture with explicit tap semantics and finite-value checks."""

    def __init__(self, model: Any, taps: tuple[ActivationTap, ...] | list[ActivationTap], *, check_finite: bool = True) -> None:
        self.model = model
        self.taps = tuple(taps)
        if not self.taps:
            raise CaptureError("at least one activation tap is required")
        names = [tap.name for tap in self.taps]
        if len(names) != len(set(names)):
            raise CaptureError("activation tap names must be unique")
        self.check_finite = check_finite
        self._handles: list[Any] = []
        self._records: dict[str, list[Any]] = {tap.name: [] for tap in self.taps}

    def _store(self, tap: ActivationTap, value: Any) -> None:
        torch = _require_torch()
        tensor = _select_tensor(value, tap.tensor_index).detach()
        if self.check_finite and not bool(torch.isfinite(tensor).all()):
            raise CaptureError(f"non-finite values captured at tap {tap.name!r}")
        self._records[tap.name].append(tensor.cpu())

    def __enter__(self) -> "TorchActivationCapture":
        for tap in self.taps:
            module = _resolve_module(self.model, tap.module_path)
            if tap.capture == "output":
                handle = module.register_forward_hook(
                    lambda _module, _inputs, output, tap=tap: self._store(tap, output)
                )
            else:
                handle = module.register_forward_pre_hook(
                    lambda _module, inputs, tap=tap: self._store(tap, inputs)
                )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def records(self, *, require_complete: bool = True) -> dict[str, tuple[Any, ...]]:
        if require_complete:
            missing = [name for name, values in self._records.items() if not values]
            if missing:
                raise CaptureError(f"no activations captured for taps: {', '.join(missing)}")
        return {name: tuple(values) for name, values in self._records.items()}

    def clear(self) -> None:
        for values in self._records.values():
            values.clear()


def save_capture_records(records: dict[str, tuple[Any, ...]], path: str) -> tuple[str, str]:
    """Persist captured tensors as safetensors plus a deterministic JSON index."""
    from pathlib import Path
    import json

    torch = _require_torch()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise CaptureError("safetensors is required to persist activation capture artifacts") from exc

    base = Path(path)
    tensor_path = base.with_suffix(".safetensors")
    metadata_path = base.with_suffix(".json")
    flat: dict[str, Any] = {}
    index: dict[str, list[str]] = {}
    for tap_name in sorted(records):
        values = records[tap_name]
        if not values:
            raise CaptureError(f"cannot persist empty activation tap {tap_name!r}")
        keys: list[str] = []
        for i, value in enumerate(values):
            if not isinstance(value, torch.Tensor):
                raise CaptureError(f"activation record {tap_name}[{i}] is not a torch tensor")
            if not bool(torch.isfinite(value).all()):
                raise CaptureError(f"activation record {tap_name}[{i}] contains non-finite values")
            key = f"{tap_name}/{i:06d}"
            flat[key] = value.detach().contiguous().cpu()
            keys.append(key)
        index[tap_name] = keys
    if not flat:
        raise CaptureError("cannot persist an empty capture record set")
    save_file(flat, str(tensor_path))
    metadata = {"schema_version": 1, "index": index}
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    return str(tensor_path), str(metadata_path)


def load_capture_records(path: str) -> dict[str, tuple[Any, ...]]:
    from pathlib import Path
    import json

    _require_torch()
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise CaptureError("safetensors is required to load activation capture artifacts") from exc

    base = Path(path)
    tensor_path = base.with_suffix(".safetensors")
    metadata_path = base.with_suffix(".json")
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise CaptureError(f"activation capture artifact is incomplete: {base}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or not isinstance(metadata.get("index"), dict):
        raise CaptureError("unsupported or invalid activation capture metadata")
    tensors = load_file(str(tensor_path), device="cpu")
    expected = {key for keys in metadata["index"].values() for key in keys}
    if set(tensors) != expected:
        raise CaptureError("activation capture tensor/index mismatch")
    return {
        tap: tuple(tensors[key] for key in keys)
        for tap, keys in sorted(metadata["index"].items())
    }
