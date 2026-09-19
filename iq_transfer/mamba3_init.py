from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np


class Mamba3InitError(RuntimeError):
    pass


TRANSFORMER_TO_MAMBA3 = {
    "attn.v": "x",
    "attn.k": "B",
    "attn.q": "C",
    "attn.o": "out_proj",
}


@dataclass(frozen=True)
class Mamba3Layout:
    d_model: int
    d_state: int = 128
    expand: float = 2.0
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 0.5
    is_mimo: bool = False
    mimo_rank: int = 4

    def __post_init__(self) -> None:
        ints = {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "headdim": self.headdim,
            "ngroups": self.ngroups,
            "mimo_rank": self.mimo_rank,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise Mamba3InitError(f"Mamba-3 dimensions must be positive: {', '.join(bad)}")
        if self.expand <= 0:
            raise Mamba3InitError("Mamba-3 expand must be positive")
        if self.rope_fraction not in (0.5, 1.0):
            raise Mamba3InitError("Mamba-3 rope_fraction must be 0.5 or 1.0")
        if self.d_inner % self.headdim != 0:
            raise Mamba3InitError("Mamba-3 d_inner must be divisible by headdim")
        if self.num_rope_angles <= 0:
            raise Mamba3InitError("Mamba-3 configuration yields no RoPE angles")

    @property
    def d_inner(self) -> int:
        return int(self.expand * self.d_model)

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim

    @property
    def effective_mimo_rank(self) -> int:
        return self.mimo_rank if self.is_mimo else 1

    @property
    def num_rope_angles(self) -> int:
        split_tensor_size = int(self.d_state * self.rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        return split_tensor_size // 2

    @property
    def split_sizes(self) -> tuple[int, ...]:
        bc = self.d_state * self.ngroups * self.effective_mimo_rank
        return (
            self.d_inner,
            self.d_inner,
            bc,
            bc,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        )

    @property
    def in_proj_shape(self) -> tuple[int, int]:
        return sum(self.split_sizes), self.d_model

    @property
    def out_proj_shape(self) -> tuple[int, int]:
        return self.d_model, self.d_inner

    def slices(self) -> Mapping[str, slice]:
        names = ("z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angle")
        start = 0
        result: dict[str, slice] = {}
        for name, size in zip(names, self.split_sizes):
            result[name] = slice(start, start + size)
            start += size
        return result


@dataclass(frozen=True)
class Mamba3BootstrapWeights:
    x: Any
    B: Any
    C: Any
    out_proj: Any


@dataclass(frozen=True)
class Mamba3BootstrapReport:
    written_slices: tuple[str, ...]
    preserved_slices: tuple[str, ...]


def _shape(value: Any) -> tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise Mamba3InitError(f"bootstrap value has no shape: {type(value).__name__}")
    return tuple(int(x) for x in value.shape)


def _clone(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return np.array(value, copy=True)


def _copy_slice(target: Any, row_slice: slice, source: Any) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(target, torch.Tensor):
        source_tensor = source if isinstance(source, torch.Tensor) else torch.as_tensor(source)
        source_tensor = source_tensor.to(device=target.device, dtype=target.dtype)
        with torch.no_grad():
            target[row_slice, :].copy_(source_tensor)
    else:
        target[row_slice, :] = np.asarray(source, dtype=np.asarray(target).dtype)


def _copy_full(target: Any, source: Any) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(target, torch.Tensor):
        source_tensor = source if isinstance(source, torch.Tensor) else torch.as_tensor(source)
        source_tensor = source_tensor.to(device=target.device, dtype=target.dtype)
        with torch.no_grad():
            target.copy_(source_tensor)
    else:
        target[...] = np.asarray(source, dtype=np.asarray(target).dtype)


def apply_mamba3_bootstrap(
    native_in_proj_weight: Any,
    native_out_proj_weight: Any,
    layout: Mamba3Layout,
    bootstrap: Mamba3BootstrapWeights,
) -> tuple[Any, Any, Mamba3BootstrapReport]:
    """Write only attention-justified Mamba-3 slices into native initialized weights.

    Expected upstream packing: [z, x, B, C, dd_dt, dd_A, trap, angle].
    Native-only slices remain bitwise unchanged in the returned copy.
    """
    if _shape(native_in_proj_weight) != layout.in_proj_shape:
        raise Mamba3InitError(
            f"native in_proj shape mismatch: got {_shape(native_in_proj_weight)}, expected {layout.in_proj_shape}"
        )
    if _shape(native_out_proj_weight) != layout.out_proj_shape:
        raise Mamba3InitError(
            f"native out_proj shape mismatch: got {_shape(native_out_proj_weight)}, expected {layout.out_proj_shape}"
        )

    slices = layout.slices()
    expected = {
        "x": (slices["x"].stop - slices["x"].start, layout.d_model),
        "B": (slices["B"].stop - slices["B"].start, layout.d_model),
        "C": (slices["C"].stop - slices["C"].start, layout.d_model),
        "out_proj": layout.out_proj_shape,
    }
    actual = {name: _shape(getattr(bootstrap, name)) for name in expected}
    mismatches = [f"{name}: got {actual[name]}, expected {shape}" for name, shape in expected.items() if actual[name] != shape]
    if mismatches:
        raise Mamba3InitError("bootstrap shape mismatch: " + "; ".join(mismatches))

    new_in = _clone(native_in_proj_weight)
    new_out = _clone(native_out_proj_weight)
    _copy_slice(new_in, slices["x"], bootstrap.x)
    _copy_slice(new_in, slices["B"], bootstrap.B)
    _copy_slice(new_in, slices["C"], bootstrap.C)
    _copy_full(new_out, bootstrap.out_proj)

    return new_in, new_out, Mamba3BootstrapReport(
        written_slices=("x", "B", "C", "out_proj"),
        preserved_slices=("z", "dd_dt", "dd_A", "trap", "angle"),
    )
