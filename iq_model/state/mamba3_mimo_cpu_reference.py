"""CPU-only rank-R Mamba-3 MIMO reference; no pretrained weights or SISO fallback.

Mirrors the upstream MIMO step equations (angle integration, split RoPE,
trapezoidal two-tap SSM, rank-specific x/z/o projections, and gating).
Source: state-spaces/mamba@e9594ce1, test_mamba3_mimo.py::mamba3_MIMO_step_ref
and mamba_ssm/modules/mamba3.py.

This is a *validation oracle*, NOT an IQ production CPU backend. Production
Mamba3MIMOState still enforces the pinned CUDA MIMO kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class MIMOReferenceState:
    angle: Tensor           # [batch, heads, angles]
    ssm: Tensor             # [batch, heads, headdim, d_state]
    previous_key: Tensor    # [batch, heads, rank, d_state]
    previous_value: Tensor  # [batch, heads, rank, headdim]


def _rotate_mimo(qk: Tensor, phase: Tensor, *, rotary_dim_divisor: int) -> Tensor:
    """MIMO's split or quarter RoPE convention, not SISO's pairwise rotation."""
    n = qk.shape[-1]
    if rotary_dim_divisor == 4:
        if n % 4 or phase.shape[-1] != n // 4:
            raise ValueError("half-RoPE MIMO requires d_state divisible by 4")
        x = qk.reshape(*qk.shape[:-1], 4, n // 4)
        cos, sin = phase.cos(), phase.sin()
        a = x[..., 0, :] * cos - x[..., 2, :] * sin
        c = x[..., 0, :] * sin + x[..., 2, :] * cos
        return torch.stack((a, x[..., 1, :], c, x[..., 3, :]), -2).reshape_as(qk)
    if rotary_dim_divisor == 2:
        if n % 2 or phase.shape[-1] != n // 2:
            raise ValueError("full-RoPE MIMO requires even d_state")
        x = qk.reshape(*qk.shape[:-1], 2, n // 2)
        cos, sin = phase.cos(), phase.sin()
        a = x[..., 0, :] * cos - x[..., 1, :] * sin
        b = x[..., 0, :] * sin + x[..., 1, :] * cos
        return torch.stack((a, b), -2).reshape_as(qk)
    raise ValueError("rotary_dim_divisor must be 2 or 4")


def _expand_groups(x: Tensor, heads: int) -> Tensor:
    """[B,R,G,N] -> [B,H,R,N] with grouped query/key heads."""
    if heads % x.shape[2]:
        raise ValueError("MIMO key groups must divide V head count")
    return x.repeat_interleave(heads // x.shape[2], dim=2).permute(0, 2, 1, 3)


def _validate(q, k, v, adt, dt, trap, q_bias, k_bias, angles,
              mimo_v, mimo_o, z, mimo_z, d, rotary_dim_divisor):
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Q/K require equal [batch,time,rank,groups,state] shapes")
    batch, steps, rank, groups, n = q.shape
    if rank < 2:
        raise ValueError("CPU reference requires actual MIMO rank >= 2")
    if min(batch, steps, groups, n) <= 0:
        raise ValueError("MIMO dimensions must be positive")
    if v.shape[:2] != (batch, steps):
        raise ValueError("V batch or sequence mismatch")
    heads, p = v.shape[2:]
    if heads <= 0 or p <= 0 or heads % groups:
        raise ValueError("invalid V head geometry")
    if adt.shape != (batch, heads, steps) or dt.shape != adt.shape or trap.shape != adt.shape:
        raise ValueError("ADT, DT and Trap require [batch,heads,time]")
    if q_bias.shape != (heads, rank, n) or k_bias.shape != q_bias.shape:
        raise ValueError("Q/K bias must have [heads,rank,d_state] shape")
    if mimo_v.shape != (heads, rank, p) or mimo_o.shape != mimo_v.shape:
        raise ValueError("MIMO projections must have [heads,rank,headdim] shape")
    if (z is None) != (mimo_z is None):
        raise ValueError("Z and MIMO_Z must both be present or absent")
    if z is not None and (z.shape != v.shape or mimo_z.shape != mimo_v.shape):
        raise ValueError("Z/gate shape mismatch")
    if d is not None and d.shape != (heads,):
        raise ValueError("D skip must have one entry per head")
    if rotary_dim_divisor not in (2, 4) or n % rotary_dim_divisor:
        raise ValueError("invalid MIMO rotary divisor or d_state")
    if angles.shape != (batch, steps, heads, n // rotary_dim_divisor):
        raise ValueError("MIMO angle projection shape mismatch")
    if any(not x.is_floating_point() for x in (q,k,v,adt,dt,trap,angles)):
        raise ValueError("MIMO inputs must be floating point")
    if bool((adt > 1e-6).any()) or bool((dt < 0).any()):
        raise ValueError("ADT must be nonpositive and DT nonnegative")
    return batch, steps, rank, heads, p, n


def mamba3_mimo_cpu_reference(
    q: Tensor, k: Tensor, v: Tensor, adt: Tensor, dt: Tensor,
    trap: Tensor, q_bias: Tensor, k_bias: Tensor, angles: Tensor,
    mimo_v: Tensor, mimo_o: Tensor, *, d: Tensor | None = None,
    z: Tensor | None = None, mimo_z: Tensor | None = None,
    state: MIMOReferenceState | None = None,
    rotary_dim_divisor: int = 4, fused_norm: bool = False,
    outproj_norm_weight: Tensor | None = None,
    outproj_norm_eps: float = 1e-5,
) -> tuple[Tensor, MIMOReferenceState]:
    """Real rank-R MIMO recurrence and state continuation on CPU.

    S_t = alpha_t*S_(t-1) + beta_t*K_(t-1)V_(t-1) + gamma_t*K_t V_t,
    where alpha=exp(ADT), beta=(1-sigmoid(Trap))*DT*alpha,
    gamma=sigmoid(Trap)*DT. K/V are rank-specific, not SISO.
    """
    batch, steps, rank, heads, p, n = _validate(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles,
        mimo_v, mimo_o, z, mimo_z, d, rotary_dim_divisor,
    )
    if fused_norm and z is None:
        raise ValueError("fused norm requires Z and MIMO_Z")
    if outproj_norm_weight is not None and outproj_norm_weight.shape != (heads,p):
        raise ValueError("outproj norm weight must have [heads,headdim] shape")
    if state is None:
        angle = angles.new_zeros(batch,heads,n//rotary_dim_divisor,dtype=torch.float32)
        ssm = q.new_zeros(batch,heads,p,n,dtype=torch.float32)
        previous_key = q.new_zeros(batch,heads,rank,n,dtype=torch.float32)
        previous_value = v.new_zeros(batch,heads,rank,p,dtype=torch.float32)
    else:
        angle, ssm = state.angle, state.ssm
        previous_key, previous_value = state.previous_key, state.previous_value
        if (angle.shape != (batch,heads,n//rotary_dim_divisor)
            or ssm.shape != (batch,heads,p,n)
            or previous_key.shape != (batch,heads,rank,n)
            or previous_value.shape != (batch,heads,rank,p)):
            raise ValueError("MIMO recurrent cache shape mismatch")
    outputs = []
    for t in range(steps):
        dt_t, adt_t = dt[:,:,t].float(), adt[:,:,t].float()
        trap_t = torch.sigmoid(trap[:,:,t].float())
        angle = angle + torch.tanh(angles[:,t].float()) * dt_t.unsqueeze(-1) * torch.pi
        phase = angle.unsqueeze(2)
        qt = _rotate_mimo(
            _expand_groups(q[:,t],heads).float() + q_bias[None].float(),
            phase, rotary_dim_divisor=rotary_dim_divisor)
        kt = _rotate_mimo(
            _expand_groups(k[:,t],heads).float() + k_bias[None].float(),
            phase, rotary_dim_divisor=rotary_dim_divisor)
        vt = v[:,t].float().unsqueeze(2) * mimo_v[None].float()
        alpha = adt_t.exp()
        beta = (1 - trap_t) * dt_t * alpha
        gamma = trap_t * dt_t
        prev_kv = torch.einsum("bhrn,bhrp->bhpn",previous_key,previous_value)
        curr_kv = torch.einsum("bhrn,bhrp->bhpn",kt,vt)
        ssm = (alpha[:,:,None,None]*ssm
               + beta[:,:,None,None]*prev_kv
               + gamma[:,:,None,None]*curr_kv)
        yt = torch.einsum("bhpn,bhrn->bhrp",ssm,qt)
        if d is not None:
            yt = yt + d[None,:,None,None].float()*vt
        if z is not None:
            zt = z[:,t].float().unsqueeze(2)*mimo_z[None].float()
            if fused_norm:
                yt = yt * torch.rsqrt(
                    yt.square().mean(dim=-1,keepdim=True) + outproj_norm_eps)
                if outproj_norm_weight is not None:
                    yt = yt*outproj_norm_weight[None,:,None,:]
            yt = yt*F.silu(zt)
        outputs.append((yt*mimo_o[None].float()).sum(dim=2))
        previous_key, previous_value = kt, vt
    return (torch.stack(outputs,dim=1).to(v.dtype),
            MIMOReferenceState(angle,ssm,previous_key,previous_value))


class _ReferenceRMSNorm(torch.nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return (x * torch.rsqrt(
            x.float().square().mean(dim=-1,keepdim=True)+self.eps).to(x.dtype)
            * self.weight)


class Mamba3MIMOTinyCPU(torch.nn.Module):
    """Random-weight reference *mixer*, matching upstream Mamba-3 projections.

    Only for CPU architectural tests, never automatic production fallback.
    """
    def __init__(self, d_model: int = 16, *, d_state: int = 16,
                 headdim: int = 8, mimo_rank: int = 4, expand: float = 2.0,
                 ngroups: int = 1, rope_fraction: float = 0.5,
                 is_outproj_norm: bool = False) -> None:
        super().__init__()
        if mimo_rank < 2 or min(d_model,d_state,headdim) <= 0:
            raise ValueError("MIMO rank >= 2 and positive dimensions required")
        if rope_fraction not in (0.5,1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0")
        self.d_model, self.d_state, self.mimo_rank = d_model,d_state,mimo_rank
        self.d_inner = int(expand*d_model)
        if self.d_inner <= 0 or self.d_inner % headdim:
            raise ValueError("expanded model width must divide head dimension")
        self.headdim, self.nheads = headdim,self.d_inner//headdim
        self.ngroups = ngroups
        if ngroups <= 0 or self.nheads % ngroups:
            raise ValueError("ngroups must divide number of heads")
        self.rotary_dim_divisor = int(2/rope_fraction)
        self.num_rope_angles = int(d_state*rope_fraction)//2
        if (self.num_rope_angles <= 0
            or self.num_rope_angles*self.rotary_dim_divisor != d_state):
            raise ValueError("CPU reference requires exact even rotary dimension")
        self.is_outproj_norm = is_outproj_norm
        projection_width = (2*self.d_inner + 2*d_state*ngroups*mimo_rank
                            + 3*self.nheads + self.num_rope_angles)
        self.in_proj = torch.nn.Linear(d_model,projection_width,bias=False)
        self.out_proj = torch.nn.Linear(self.d_inner,d_model,bias=False)
        self.dt_bias = torch.nn.Parameter(torch.full((self.nheads,),-3.0))
        self.B_bias = torch.nn.Parameter(torch.ones(self.nheads,mimo_rank,d_state))
        self.C_bias = torch.nn.Parameter(torch.ones(self.nheads,mimo_rank,d_state))
        self.B_norm = _ReferenceRMSNorm(d_state)
        self.C_norm = _ReferenceRMSNorm(d_state)
        self.mimo_x = torch.nn.Parameter(
            torch.ones(self.nheads,mimo_rank,headdim)/mimo_rank)
        self.mimo_z = torch.nn.Parameter(torch.ones(self.nheads,mimo_rank,headdim))
        self.mimo_o = torch.nn.Parameter(
            torch.ones(self.nheads,mimo_rank,headdim)/mimo_rank)
        self.D = torch.nn.Parameter(torch.ones(self.nheads))
        self.norm = _ReferenceRMSNorm(self.d_inner) if is_outproj_norm else None

    def forward(self, hidden: Tensor, *, state: MIMOReferenceState | None = None):
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("MIMO tiny mixer requires [batch,time,d_model]")
        b,t,_ = hidden.shape
        raw = self.in_proj(hidden)
        z,x,B,C,dd_dt,dd_A,trap,angles = torch.split(raw,[
            self.d_inner,self.d_inner,
            self.d_state*self.ngroups*self.mimo_rank,
            self.d_state*self.ngroups*self.mimo_rank,
            self.nheads,self.nheads,self.nheads,self.num_rope_angles,
        ],dim=-1)
        V = x.reshape(b,t,self.nheads,self.headdim)
        Z = z.reshape(b,t,self.nheads,self.headdim)
        K = self.B_norm(B.reshape(
            b,t,self.mimo_rank,self.ngroups,self.d_state))
        Q = self.C_norm(C.reshape(
            b,t,self.mimo_rank,self.ngroups,self.d_state))
        dt = F.softplus(dd_dt.float()+self.dt_bias)
        # Upstream heavy-tail A, with the same stability floor.
        neg = dd_A.float().clamp_max(0)
        pos = dd_A.float().clamp_min(0)
        a = -(pos+torch.reciprocal(1-neg)).clamp_min(1e-4)
        y,new_state = mamba3_mimo_cpu_reference(
            q=Q,k=K,v=V,adt=(a*dt).transpose(1,2),
            dt=dt.transpose(1,2),trap=trap.transpose(1,2),
            q_bias=self.C_bias,k_bias=self.B_bias,
            angles=angles.unsqueeze(2).expand(-1,-1,self.nheads,-1),
            mimo_v=self.mimo_x,mimo_o=self.mimo_o,d=self.D,
            z=Z,mimo_z=self.mimo_z,state=state,
            rotary_dim_divisor=self.rotary_dim_divisor,
            fused_norm=self.is_outproj_norm,
            outproj_norm_weight=(
                self.norm.weight.reshape(self.nheads,self.headdim)
                if self.norm is not None else None),
        )
        return self.out_proj(y.reshape(b,t,self.d_inner)),new_state


class Mamba3MIMOCPUReferenceState(torch.nn.Module):
    """Test-only MIMO mixer with IQ's actual packed-sequence interface.

    Each cu_seqlens span starts with independent zero SSM/angle/K/V state.
    Never silently replace the production CUDA kernel with this CPU oracle.
    """

    def __init__(self, config, *, layer_idx: int) -> None:
        super().__init__()
        if config.mimo_rank != 4:
            raise ValueError("full IQ CPU validation requires rank-4 MIMO")
        if not 0 <= layer_idx < config.num_layers:
            raise ValueError("invalid Mamba layer index")
        self.layer_idx = layer_idx
        self.core = Mamba3MIMOTinyCPU(
            d_model=config.d_model,
            d_state=config.d_state,
            headdim=config.headdim,
            mimo_rank=config.mimo_rank,
            expand=config.expand,
            rope_fraction=config.rope_fraction,
            is_outproj_norm=config.outproj_norm,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cu_seqlens: Tensor | None = None,
        inference_params=None,
    ) -> Tensor:
        if inference_params is not None:
            raise ValueError("CPU oracle does not implement production CUDA cache")
        if cu_seqlens is None:
            return self.core(hidden_states)[0]
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[0] != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("packed MIMO expects [1,total,hidden] and cu_seqlens")
        boundaries = cu_seqlens.tolist()
        if (
            boundaries[0] != 0
            or boundaries[-1] != hidden_states.shape[1]
            or any(b <= a for a, b in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("invalid packed MIMO segment boundaries")
        segments = []
        for start, end in zip(boundaries, boundaries[1:]):
            segment, _ = self.core(hidden_states[:,start:end])
            segments.append(segment)
        return torch.cat(segments, dim=1)
