"""Chunked-layout Mamba-3 MIMO runtime ported to plain CPU tensors.

Separate, explicitly-gated CPU runtime for IQ's production Mamba-3 MIMO
mixer (`iq_model/state/mamba3.py` is deliberately CUDA-only and is not
touched). No imports from `mamba_ssm` or TileLang; every tensor op below is
plain PyTorch.

Reference math (state-spaces/mamba at commit
e9594ce1c732d97440f0332fdc43170a2294dbfa — the exact commit pinned by
production `iq_model/state/mamba3.py:11 MAMBA3_UPSTREAM_COMMIT` and by
`MAMBA3_CPU_SOURCE_COMMIT` here). The plan brief listed these paths without
the `mamba_ssm/` prefix (those URLs 404); the real paths at that commit are:

- `mamba_ssm/modules/mamba3.py` — production module: in_proj layout and
  split order [z, x, B, C, dd_dt, dd_A, trap, angle] (:106-108, :177-186,
  :340-352), heavy_tail activation (:27-41), dt/A parameterization
  (:54-57, :111-118), B/C RMSNorm eps 1e-5 (:126-127), MIMO projections
  (:129-137), decode step wiring (:314-440), MIMO pre/post-process
  (:281-302, :304-312), inference-cache shapes/dtypes (:442-482).
- `mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py:741-813`
  `selective_state_update_fused_ref_v2` — the decode recurrence ported
  verbatim: alpha=exp(A*dt), beta=(1-trap)*dt*alpha, gamma=trap*dt, the
  two-term MIMO state update (current-token and previous-token terms via
  stored k/v states), D skip, silu(z) gating, mimo_o reduction.
- `mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py:239-324`
  `apply_rotary_qk_inference_reference` — halved-branch rotary ported
  verbatim: angle_state += tanh(angle_proj) * dt * pi (fp32 angle cache,
  kernel computes tanh in fp32, :70-76), bias_q=C_bias/bias_k=B_bias
  permuted (R,N,S) (modules/mamba3.py:357-358), cos=1/sin=0 identity
  padding beyond the rotary range (:300-322).
- `mamba_ssm/ops/triton/layernorm_gated.py:18-39` `rms_norm_ref` — RMSNorm
  ported verbatim (fp32 upcast, eps inside sqrt, silu gate placement,
  group_size chunking) for the B/C norms and the optional outproj norm.

Production semantics that override the plan brief's wiring table:
- in_proj split order is (z, x, B, C, dd_dt, dd_A, trap, angle); the brief's
  "(x, xs, B_bias, C_bias, ...)" order does not match the module.
- The IQ checkpoint (`checkpoints/iq/iq_config.json`) sets
  `outproj_norm=false`, so the decode step uses the kernel-internal gating
  path: out_r * silu(z * mimo_z), then reduction with mimo_o
  (mamba3_step_fn.py:801-809, modules/mamba3.py:406-427). The brief's
  rms_norm_gated postprocess describes the outproj_norm=true path, which is
  also implemented here behind the config flag (modules/mamba3.py:304-312,
  :382-405).
- ngroups is fixed at 1 (production instantiates Mamba3 without ngroups,
  iq_model/state/mamba3.py:239-253).
- The kernel computes tanh as sigmoid(2x)*2-1 (rotary file :76); this port
  uses torch.tanh as the reference function does (rotary file :255) — a
  bounded fp32-level difference to be quantified by the Task 9 H200 parity
  fixture.
- `prefill` is a loop of `step` calls (v1 per the task brief): numerically
  identical to sequential decode, with fp accumulation order that differs
  from the CUDA chunk kernel (`mamba3_mimo` TileLang); the Task 9 fixture
  quantifies that drift.

Parameter initialization is deterministic (linears zero, RMSNorm ones,
production formulas for dt/bias/mimo with a fixed generator); real weights
are loaded from the checkpoint later, so init only needs determinism.

State layout mirrors production `allocate_inference_cache`
(modules/mamba3.py:442-482): angle fp32 [B, nheads, num_rope_angles], ssm
fp32 [B, nheads, headdim, d_state], k [B, mimo_rank, nheads, d_state] and
v [B, nheads, headdim] in the model dtype (bf16 for production configs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MAMBA3_CPU_SOURCE_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"

_DT_MIN = 0.001
_DT_MAX = 0.1
_DT_INIT_FLOOR = 1e-4
_A_FLOOR = 1e-4
_BC_NORM_EPS = 1e-5


class Mamba3CPURuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CPUConfig:
    """Field names/defaults mirror production `Mamba3MIMOConfig`
    (iq_model/state/mamba3.py:18-50) including its validation."""

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

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim

    @property
    def num_rope_angles(self) -> int:
        split = int(self.d_state * self.rope_fraction)
        if split % 2 != 0:
            split -= 1
        return split // 2

    @classmethod
    def production_4096x32(cls) -> CPUConfig:
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


Mamba3MIMOConfigCPU = CPUConfig


@dataclass
class CPUMamba3State:
    """Decode state, mirroring production allocate_inference_cache
    (modules/mamba3.py:442-482)."""

    angle: torch.Tensor
    ssm: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor

    @property
    def batch(self) -> int:
        return self.angle.shape[0]

    @classmethod
    def create(
        cls,
        batch: int,
        config: CPUConfig,
        *,
        device: torch.device | str = "cpu",
        kv_dtype: torch.dtype = torch.bfloat16,
    ) -> CPUMamba3State:
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        zeros = torch.zeros
        return cls(
            angle=zeros(
                (batch, config.nheads, config.num_rope_angles),
                device=device,
                dtype=torch.float32,
            ),
            ssm=zeros(
                (batch, config.nheads, config.headdim, config.d_state),
                device=device,
                dtype=torch.float32,
            ),
            k=zeros(
                (batch, config.mimo_rank, config.nheads, config.d_state),
                device=device,
                dtype=kv_dtype,
            ),
            v=zeros(
                (batch, config.nheads, config.headdim),
                device=device,
                dtype=kv_dtype,
            ),
        )

    def clone_state(self) -> CPUMamba3State:
        return CPUMamba3State(
            angle=self.angle.detach().clone(),
            ssm=self.ssm.detach().clone(),
            k=self.k.detach().clone(),
            v=self.v.detach().clone(),
        )

    def drop_state(self) -> CPUMamba3State:
        self.angle.zero_()
        self.ssm.zero_()
        self.k.zero_()
        self.v.zero_()
        return self


def _heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """Verbatim port of modules/mamba3.py:27-41."""
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def _rms_norm_ref_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    upcast: bool = True,
) -> torch.Tensor:
    """Verbatim port of ops/triton/layernorm_gated.py:18-39 rms_norm_ref."""
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = x.reshape(*x.shape[:-1], -1, group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).reshape(x.shape) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


def _rotary_qk_step_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    angle_state: torch.Tensor,
    angle_proj: torch.Tensor,
    dt: torch.Tensor,
    bias_q: torch.Tensor,
    bias_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Halved-branch decode rotary, ported from
    ops/triton/mamba3/mamba3_mimo_rotary_step.py:239-324
    apply_rotary_qk_inference_reference (rotate_pairwise=False is the MIMO
    branch; modules/mamba3.py:363-368). tanh runs in fp32 because the
    production kernel upcasts the projection before tanh (rotary file
    :70-76). Outputs are cast back to the input dtype like the kernel's
    empty_like buffers (rotary file :198-199).
    """
    batch, rank, nheads, dim = q.shape
    rotary_dim = angle_state.shape[-1] * 2
    out_dtype_q = q.dtype
    out_dtype_k = k.dtype
    angle_proj = torch.tanh(angle_proj.to(torch.float32))
    angle = angle_state + angle_proj * dt[:, :, None] * math.pi
    angle_state_new = angle
    angle = angle.unsqueeze(1).expand(-1, rank, -1, -1)

    q = q.to(torch.float32) + bias_q[None, :, :, :]
    k = k.to(torch.float32) + bias_k[None, :, :, :]

    half = dim // 2
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    rdim_half = rotary_dim // 2
    if half > rdim_half:
        pad_shape = list(cos.shape)
        pad_shape[-1] = half - rdim_half
        cos = torch.cat([cos, torch.ones(pad_shape, dtype=cos.dtype, device=cos.device)], dim=-1)
        sin = torch.cat([sin, torch.zeros(pad_shape, dtype=sin.dtype, device=sin.device)], dim=-1)

    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]
    qo0 = q0 * cos - q1 * sin
    qo1 = q0 * sin + q1 * cos
    ko0 = k0 * cos - k1 * sin
    ko1 = k0 * sin + k1 * cos
    q_out = torch.cat([qo0, qo1], dim=-1).to(out_dtype_q)
    k_out = torch.cat([ko0, ko1], dim=-1).to(out_dtype_k)
    return q_out, k_out, angle_state_new


class _RMSNormCPU(nn.Module):
    """CPU mirror of ops/triton/layernorm_gated.py RMSNorm (:415-437):
    ones-initialized weight, no bias, rms_norm_ref math."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        return _rms_norm_ref_cpu(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


class _Mamba3MIMOCore(nn.Module):
    """Parameter set and math of the production upstream Mamba-3 MIMO core
    (mamba_ssm/modules/mamba3.py at MAMBA3_CPU_SOURCE_COMMIT), on plain CPU
    tensors. Parameter names match the checkpoint's
    `layers.<i>.mamba.core.*` entries exactly."""

    def __init__(
        self,
        config: CPUConfig,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_inner = config.d_inner
        self.nheads = config.nheads
        self.num_rope_angles = config.num_rope_angles

        d_in_proj = (
            2 * self.d_inner
            + 2 * config.d_state * config.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.in_proj.weight)

        generator = torch.Generator().manual_seed(0)
        _dt = torch.exp(
            torch.rand(self.nheads, generator=generator, dtype=torch.float32)
            * (math.log(_DT_MAX) - math.log(_DT_MIN))
            + math.log(_DT_MIN)
        )
        _dt = torch.clamp(_dt, min=_DT_INIT_FLOOR)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)

        self.B_bias = nn.Parameter(
            1 + torch.zeros(
                (self.nheads, config.mimo_rank, config.d_state),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros(
                (self.nheads, config.mimo_rank, config.d_state),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=True,
        )
        self.B_norm = _RMSNormCPU(config.d_state, eps=_BC_NORM_EPS, device=device, dtype=dtype)
        self.C_norm = _RMSNormCPU(config.d_state, eps=_BC_NORM_EPS, device=device, dtype=dtype)

        self.mimo_x = nn.Parameter(
            torch.ones(self.nheads, config.mimo_rank, config.headdim, device=device)
            / config.mimo_rank,
            requires_grad=True,
        )
        self.mimo_z = nn.Parameter(
            torch.ones(self.nheads, config.mimo_rank, config.headdim, device=device),
            requires_grad=True,
        )
        self.mimo_o = nn.Parameter(
            torch.ones(self.nheads, config.mimo_rank, config.headdim, device=device)
            / config.mimo_rank,
            requires_grad=True,
        )

        self.D = nn.Parameter(torch.ones(self.nheads, device=device), requires_grad=True)

        if config.outproj_norm:
            self.norm = _RMSNormCPU(
                self.d_inner,
                eps=_BC_NORM_EPS,
                norm_before_gate=True,
                group_size=config.headdim,
                device=device,
                dtype=dtype,
            )

        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.out_proj.weight)

    def step(
        self,
        u: torch.Tensor,
        state: CPUMamba3State,
    ) -> tuple[torch.Tensor, CPUMamba3State]:
        config = self.config
        batch, dim = u.shape
        d_inner = self.d_inner
        nheads = self.nheads
        headdim = config.headdim
        d_state = config.d_state
        rank = config.mimo_rank

        zxBCdtAtrap = self.in_proj(u)
        z, x, b, c, dd_dt, dd_a, trap_raw, angles_raw = torch.split(
            zxBCdtAtrap,
            [
                d_inner,
                d_inner,
                d_state * rank,
                d_state * rank,
                nheads,
                nheads,
                nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        a_mat = -_heavy_tail_activation(dd_a.to(torch.float32))
        a_mat = torch.clamp(a_mat, max=-_A_FLOOR)
        dt = F.softplus(dd_dt.to(torch.float32) + self.dt_bias)
        trap = torch.sigmoid(trap_raw)

        b4 = b.reshape(batch, rank, 1, d_state)
        c4 = c.reshape(batch, rank, 1, d_state)
        b_n = self.B_norm(b4).expand(-1, -1, nheads, -1)
        c_n = self.C_norm(c4).expand(-1, -1, nheads, -1)
        x4 = x.reshape(batch, nheads, headdim)
        z4 = z.reshape(batch, nheads, headdim)
        angles = angles_raw.unsqueeze(1).expand(-1, nheads, -1)

        q, k_rot, angle_state_new = _rotary_qk_step_cpu(
            q=c_n,
            k=b_n,
            angle_state=state.angle,
            angle_proj=angles,
            dt=dt,
            bias_q=self.C_bias.permute(1, 0, 2),
            bias_k=self.B_bias.permute(1, 0, 2),
        )

        nxt_v = x4
        nxt_k = k_rot.to(state.k.dtype)

        alpha = torch.exp(a_mat * dt)
        beta = (1.0 - trap.to(torch.float32)) * dt * alpha
        gamma = trap.to(torch.float32) * dt

        # The transferred checkpoint can store MIMO factors in BF16. Contract
        # them in FP32 with the FP32 recurrent state; restore BF16 only before
        # the final learned out_proj, as in the existing CPU math path.
        xpj = self.mimo_x.permute(1, 0, 2).to(torch.float32)
        x_vals = x4.to(torch.float32)[:, None, :, :] * xpj[None, :, :, :]
        xs_vals = state.v.to(torch.float32)[:, None, :, :] * xpj[None, :, :, :]

        xbt_state = torch.einsum(
            "brnh,brns->bnhs",
            x_vals * gamma.unsqueeze(-1).unsqueeze(1),
            k_rot.to(torch.float32),
        )
        xbt_prev = torch.einsum(
            "brnh,brns->bnhs",
            xs_vals * beta.unsqueeze(-1).unsqueeze(1),
            state.k.to(torch.float32),
        )

        new_state = state.ssm * alpha[:, :, None, None] + xbt_state + xbt_prev
        out_r = torch.einsum("bnhs,brns->brnh", new_state, q.to(torch.float32))
        out_r = out_r + (x_vals * self.D[None, :, None])

        if config.outproj_norm:
            y = self._postprocess(out_r, z4)
        else:
            zpj = self.mimo_z.permute(1, 0, 2).to(torch.float32)
            z_vals = z4.to(torch.float32)[:, None, :, :] * zpj[None, :, :, :]
            out_r = out_r * z_vals * torch.sigmoid(z_vals)
            outpj = self.mimo_o.permute(1, 0, 2).to(torch.float32)
            y = torch.einsum("brnh,rnh->bnh", out_r, outpj)

        out = y.reshape(batch, d_inner).to(u.dtype)
        out = self.out_proj(out)

        new_state_obj = CPUMamba3State(
            angle=angle_state_new,
            ssm=new_state,
            k=nxt_k,
            v=nxt_v,
        )
        return out, new_state_obj

    def _postprocess(
        self,
        y_r: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """Verbatim port of modules/mamba3.py:304-312 _postprocess (the
        outproj_norm=true decode path; modules/mamba3.py:382-405)."""
        batch, rank, nheads, headdim = y_r.shape
        zpj = self.mimo_z.permute(1, 0, 2).to(torch.float32)
        z_r = torch.einsum("bhp,rhp->brhp", z4.to(torch.float32), zpj)
        z_r = z_r.reshape(batch, rank, nheads * headdim)
        y = y_r.reshape(batch, rank, nheads * headdim).to(torch.float32)
        y = self.norm(y, z_r)
        y = y.reshape(batch, rank, nheads, headdim)
        outpj = self.mimo_o.permute(1, 0, 2).to(torch.float32)
        return torch.einsum("brhp,rhp->bhp", y, outpj)


class Mamba3CPU(nn.Module):
    """Explicitly-gated CPU Mamba-3 MIMO mixer with production state_dict
    layout: `core.<param>` mirrors the checkpoint's
    `layers.<i>.mamba.core.<param>` names."""

    def __init__(
        self,
        config: CPUConfig,
        *,
        layer_idx: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if layer_idx < 0 or layer_idx >= config.num_layers:
            raise ValueError(
                f"layer_idx must be in [0, {config.num_layers}), got {layer_idx}"
            )
        self.config = config
        self.layer_idx = layer_idx
        self.core = _Mamba3MIMOCore(config, dtype=dtype, device=device)

    @property
    def _kv_dtype(self) -> torch.dtype:
        return self.core.in_proj.weight.dtype

    def _validate_input(self, x_b: torch.Tensor, state: CPUMamba3State | None) -> None:
        if x_b.device.type != "cpu":
            raise Mamba3CPURuntimeError(
                "Mamba3CPU is the explicit CPU runtime; route CUDA traffic to "
                "iq_model.state.mamba3.Mamba3MIMOState instead"
            )
        if x_b.ndim != 2 or x_b.shape[-1] != self.config.d_model:
            raise ValueError(
                "Mamba3CPU.step expects x_b of shape [batch, "
                f"{self.config.d_model}], got {tuple(x_b.shape)}"
            )
        if not x_b.is_floating_point():
            raise ValueError("Mamba3CPU.step requires a floating-point input")
        if state is not None:
            expected = {
                "angle": (x_b.shape[0], self.config.nheads, self.config.num_rope_angles),
                "ssm": (x_b.shape[0], self.config.nheads, self.config.headdim, self.config.d_state),
                "k": (x_b.shape[0], self.config.mimo_rank, self.config.nheads, self.config.d_state),
                "v": (x_b.shape[0], self.config.nheads, self.config.headdim),
            }
            for name, shape in expected.items():
                actual = tuple(getattr(state, name).shape)
                if actual != shape:
                    raise ValueError(
                        f"Mamba3CPU state.{name} must have shape {shape}, got {actual}"
                    )
            for name in ("angle", "ssm", "k", "v"):
                if getattr(state, name).device != x_b.device:
                    raise Mamba3CPURuntimeError(
                        f"Mamba3CPU state.{name} device {getattr(state, name).device} "
                        f"does not match input device {x_b.device}"
                    )

    def step(
        self,
        x_b: torch.Tensor,
        state: CPUMamba3State,
    ) -> tuple[torch.Tensor, CPUMamba3State]:
        self._validate_input(x_b, state)
        y_b, new_state = self.core.step(x_b, state)
        if not bool(torch.isfinite(y_b).all()):
            raise Mamba3CPURuntimeError("Mamba3CPU produced non-finite output")
        return y_b, new_state

    def prefill(
        self,
        x: torch.Tensor,
        state: CPUMamba3State | None = None,
    ) -> tuple[torch.Tensor, CPUMamba3State]:
        if x.device.type != "cpu":
            raise Mamba3CPURuntimeError(
                "Mamba3CPU is the explicit CPU runtime; route CUDA traffic to "
                "iq_model.state.mamba3.Mamba3MIMOState instead"
            )
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError(
                "Mamba3CPU.prefill expects x of shape [batch, seq, "
                f"{self.config.d_model}], got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("Mamba3CPU.prefill requires a floating-point input")
        if state is None:
            state = CPUMamba3State.create(
                batch=x.shape[0],
                config=self.config,
                device=x.device,
                kv_dtype=self._kv_dtype,
            )
        else:
            self._validate_input(x[:, 0], state)
        if x.shape[1] == 0:
            return x.new_empty((x.shape[0], 0, self.config.d_model)), state
        ys = []
        for t in range(x.shape[1]):
            y, state = self.core.step(x[:, t], state)
            ys.append(y)
        y_all = torch.stack(ys, dim=1)
        if not bool(torch.isfinite(y_all).all()):
            raise Mamba3CPURuntimeError("Mamba3CPU produced non-finite output")
        return y_all, state
