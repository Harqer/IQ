from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
from typing import Any

import torch
from torch import nn


MAMBA3_UPSTREAM_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


class Mamba3MIMORuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mamba3MIMOConfig:
    d_model: int
    num_layers: int
    d_state: int = 128
    headdim: int = 64
    mimo_rank: int = 4
    expand: float = 2.0
    rope_fraction: float = 0.5
    chunk_size: int = 16
    outproj_norm: bool = False

    def __post_init__(self) -> None:
        ints = {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "d_state": self.d_state,
            "headdim": self.headdim,
            "mimo_rank": self.mimo_rank,
            "chunk_size": self.chunk_size,
        }
        bad = [name for name, value in ints.items() if int(value) <= 0]
        if bad:
            raise ValueError(f"positive Mamba-3 fields required: {', '.join(bad)}")
        if self.mimo_rank < 2:
            raise ValueError("IQ requires Mamba-3 MIMO; mimo_rank must be >= 2")
        if self.expand <= 0:
            raise ValueError("Mamba-3 expand must be positive")
        d_inner = int(self.expand * self.d_model)
        if d_inner % self.headdim != 0:
            raise ValueError("Mamba-3 inner width must be divisible by headdim")
        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError("Mamba-3 rope_fraction must be 0.5 or 1.0")

    @property
    def d_inner(self) -> int:
        return int(self.expand * self.d_model)

    @classmethod
    def production_4096x32(cls) -> "Mamba3MIMOConfig":
        return cls(
            d_model=4096,
            num_layers=32,
            d_state=128,
            headdim=64,
            mimo_rank=4,
            expand=2.0,
            rope_fraction=0.5,
            chunk_size=16,
            outproj_norm=False,
        )


@dataclass(frozen=True)
class Mamba3MIMORuntimeInfo:
    installed_version: str | None
    source_commit: str | None
    mimo_kernel_available: bool
    cuda_available: bool
    device_capability: tuple[int, int] | None

    @property
    def ready(self) -> bool:
        return (
            self.mimo_kernel_available
            and self.cuda_available
            and self.source_commit == MAMBA3_UPSTREAM_COMMIT
        )


def recommended_mamba3_chunk_size(
    *,
    mimo_rank: int,
    dtype: torch.dtype,
) -> int:
    if mimo_rank < 2:
        raise Mamba3MIMORuntimeError(
            "IQ requires Mamba-3 MIMO; mimo_rank must be >= 2"
        )
    base = 64 if dtype is torch.bfloat16 else 32
    if base % mimo_rank != 0:
        raise Mamba3MIMORuntimeError(
            f"Mamba-3 chunk-size base {base} is not divisible by mimo_rank={mimo_rank}"
        )
    chunk = base // mimo_rank
    if chunk <= 0:
        raise Mamba3MIMORuntimeError("computed Mamba-3 chunk size is invalid")
    return chunk


def _installed_source_commit() -> str | None:
    try:
        dist = metadata.distribution("mamba_ssm")
    except metadata.PackageNotFoundError:
        return None

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return None
    try:
        data = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    vcs = data.get("vcs_info")
    if not isinstance(vcs, dict):
        return None
    commit = vcs.get("commit_id")
    return str(commit) if commit else None


def inspect_mamba3_mimo_runtime(
    device: torch.device | None = None,
) -> Mamba3MIMORuntimeInfo:
    version: str | None = None
    mimo_available = False
    try:
        import mamba_ssm
        from mamba_ssm.modules import mamba3 as mamba3_module

        version = str(getattr(mamba_ssm, "__version__", "unknown"))
        mimo_available = getattr(mamba3_module, "mamba3_mimo_combined", None) is not None
    except Exception:
        mimo_available = False

    cuda_available = torch.cuda.is_available()
    capability: tuple[int, int] | None = None
    if cuda_available:
        target = device if device is not None else torch.device("cuda")
        if target.type == "cuda":
            capability = tuple(int(x) for x in torch.cuda.get_device_capability(target))

    return Mamba3MIMORuntimeInfo(
        installed_version=version,
        source_commit=_installed_source_commit(),
        mimo_kernel_available=mimo_available,
        cuda_available=cuda_available,
        device_capability=capability,
    )


def require_mamba3_mimo_runtime(
    device: torch.device | None = None,
) -> Mamba3MIMORuntimeInfo:
    if device is not None and device.type != "cuda":
        raise Mamba3MIMORuntimeError(
            "IQ Mamba-3 MIMO requires a CUDA target device; no CPU/SISO fallback is permitted"
        )
    info = inspect_mamba3_mimo_runtime(device)
    if not info.cuda_available:
        raise Mamba3MIMORuntimeError(
            "IQ Mamba-3 MIMO requires CUDA; no CPU/SISO fallback is permitted"
        )
    if not info.mimo_kernel_available:
        raise Mamba3MIMORuntimeError(
            "Mamba-3 MIMO kernel is unavailable. Install the pinned state-spaces/mamba "
            "source with TileLang support; IQ will not silently fall back to SISO."
        )
    if info.source_commit != MAMBA3_UPSTREAM_COMMIT:
        raise Mamba3MIMORuntimeError(
            "Mamba-3 source revision is not the pinned IQ production revision: "
            f"installed={info.source_commit!r}, required={MAMBA3_UPSTREAM_COMMIT}"
        )
    return info


class Mamba3MIMOState(nn.Module):
    """IQ's production Mamba-3 state mixer.

    MIMO is architectural, not optional: this wrapper always instantiates upstream
    Mamba-3 with is_mimo=True and never substitutes a SISO implementation.
    """

    def __init__(
        self,
        config: Mamba3MIMOConfig,
        *,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if layer_idx < 0 or layer_idx >= config.num_layers:
            raise ValueError(
                f"layer_idx must be in [0, {config.num_layers}), got {layer_idx}"
            )

        device_obj = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        require_mamba3_mimo_runtime(device_obj)

        try:
            from mamba_ssm import Mamba3
            from mamba_ssm.modules import mamba3 as mamba3_module
        except Exception as exc:
            raise Mamba3MIMORuntimeError(
                "failed to import pinned upstream Mamba-3"
            ) from exc
        if getattr(mamba3_module, "mamba3_mimo_combined", None) is None:
            raise Mamba3MIMORuntimeError(
                "upstream Mamba-3 imported without the MIMO kernel; no SISO fallback is allowed"
            )

        expected_chunk = recommended_mamba3_chunk_size(
            mimo_rank=config.mimo_rank,
            dtype=dtype,
        )
        if config.chunk_size != expected_chunk:
            raise Mamba3MIMORuntimeError(
                "IQ Mamba-3 chunk_size must match the upstream MIMO recommendation for "
                f"dtype/rank: configured={config.chunk_size}, expected={expected_chunk}"
            )

        self.config = config
        self.layer_idx = layer_idx
        self.core = Mamba3(
            d_model=config.d_model,
            d_state=config.d_state,
            expand=config.expand,
            headdim=config.headdim,
            rope_fraction=config.rope_fraction,
            is_outproj_norm=config.outproj_norm,
            is_mimo=True,
            mimo_rank=config.mimo_rank,
            chunk_size=config.chunk_size,
            layer_idx=layer_idx,
            n_layer=config.num_layers,
            device=device_obj,
            dtype=dtype,
        )

        if not bool(getattr(self.core, "is_mimo", False)):
            raise Mamba3MIMORuntimeError(
                "upstream Mamba-3 did not construct in MIMO mode"
            )
        if int(getattr(self.core, "mimo_rank", -1)) != config.mimo_rank:
            raise Mamba3MIMORuntimeError(
                "upstream Mamba-3 MIMO rank does not match IQ config"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        inference_params: Any | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                "Mamba-3 hidden_states must have shape "
                f"[batch, sequence, {self.config.d_model}]"
            )
        if hidden_states.device.type != "cuda":
            raise Mamba3MIMORuntimeError(
                "IQ Mamba-3 MIMO forward requires CUDA; no CPU/SISO fallback is allowed"
            )
        output = self.core(
            hidden_states,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params,
        )
        if output.shape != hidden_states.shape:
            raise Mamba3MIMORuntimeError(
                f"Mamba-3 output shape changed unexpectedly: {tuple(output.shape)}"
            )
        if not bool(torch.isfinite(output).all()):
            raise Mamba3MIMORuntimeError("Mamba-3 produced non-finite output")
        return output

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        if batch_size <= 0 or max_seqlen <= 0:
            raise ValueError("batch_size and max_seqlen must be positive")
        return self.core.allocate_inference_cache(
            batch_size,
            max_seqlen,
            device=device,
            dtype=dtype,
            **kwargs,
        )
