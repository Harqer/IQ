"""CPU Mamba-3 MIMO port tests.

The parity oracle compares `iq_model.state.mamba3_cpu.Mamba3CPU` against a
naive recurrence reference written directly from the pinned production math,
because the production CUDA path cannot execute on CPU
(`iq_model/state/mamba3.py:158-180` requires CUDA; the step kernel is
CuteDSL, the prefill kernel is TileLang).

Reference sources (state-spaces/mamba at commit
e9594ce1c732d97440f0332fdc43170a2294dbfa, the commit pinned by
`iq_model/state/mamba3.py:11 MAMBA3_UPSTREAM_COMMIT`):
- modules/mamba3.py:27-41 heavy_tail_activation
- modules/mamba3.py:92-104 d_inner/nheads/num_rope_angles derivation
- modules/mamba3.py:106-108 in_proj layout [z, x, B, C, dd_dt, dd_A, trap, angle]
- modules/mamba3.py:281-302 _preprocess (A/DT/trap, B/C RMSNorm, head expand)
- modules/mamba3.py:314-440 step() decode wiring (bias_q=C_bias permute,
  halved rotary for MIMO, nxt_v=x, nxt_k=B, kernel call, out_proj)
- ops/cute/mamba3/mamba3_step_fn.py:741-813 selective_state_update_fused_ref_v2
  (alpha/beta/gamma, two-term MIMO state, D skip, silu gate, mimo_o)
- ops/triton/mamba3/mamba3_mimo_rotary_step.py:239-324
  apply_rotary_qk_inference_reference (tanh*dt*pi angle, halved branch)
- ops/triton/layernorm_gated.py:18-39 rms_norm_ref (fp32 upcast, eps, silu)
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from iq_model.hybrid import IQHybridConfig
from iq_model.state import mamba3 as production_mamba3
from iq_model.state.mamba3_cpu import (
    MAMBA3_CPU_SOURCE_COMMIT,
    CPUMamba3State,
    CPUConfig,
    Mamba3CPU,
    Mamba3MIMOConfigCPU,
)

CHECKPOINT_CONFIG = "checkpoints/iq/iq_config.json"

CHECKPOINT_CORE_PARAM_NAMES = frozenset(
    {
        "B_bias",
        "B_norm.weight",
        "C_bias",
        "C_norm.weight",
        "D",
        "dt_bias",
        "in_proj.weight",
        "mimo_o",
        "mimo_x",
        "mimo_z",
        "out_proj.weight",
    }
)

TINY_D_MODEL = 6
TINY_HEADDIM = 3
TINY_D_STATE = 4
TINY_MIMO_RANK = 2
PARITY_ATOL = 3e-4


def _tiny_config() -> CPUConfig:
    return CPUConfig(
        d_model=TINY_D_MODEL,
        num_layers=1,
        d_state=TINY_D_STATE,
        headdim=TINY_HEADDIM,
        mimo_rank=TINY_MIMO_RANK,
        expand=2.0,
        rope_fraction=0.5,
        chunk_size=16,
        outproj_norm=False,
    )


def _tiny_dims(config: CPUConfig) -> dict:
    d_inner = int(config.expand * config.d_model)
    nheads = d_inner // config.headdim
    split = int(config.d_state * config.rope_fraction)
    if split % 2 != 0:
        split -= 1
    num_rope_angles = split // 2
    return {
        "d_inner": d_inner,
        "nheads": nheads,
        "headdim": config.headdim,
        "d_state": config.d_state,
        "mimo_rank": config.mimo_rank,
        "num_rope_angles": num_rope_angles,
    }


def _randomize_core_params(core: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for _, param in core.named_parameters():
        value = torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.4
        if param.dtype == torch.bfloat16:
            value = value.to(torch.bfloat16)
        with torch.no_grad():
            param.copy_(value)


def _reference_params_from(core: nn.Module, config: CPUConfig) -> dict:
    params = {
        name: param.detach().clone()
        for name, param in core.named_parameters()
    }
    params.update(_tiny_dims(config))
    return params


def _rms_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """rms_norm_ref without z/group (ops/triton/layernorm_gated.py:18-39)."""
    dtype = x.dtype
    x = x.float()
    rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return (x * rstd * weight.float()).to(dtype)


def _naive_mamba3_step(
    u: torch.Tensor,
    params: dict,
    state: dict,
) -> tuple[torch.Tensor, dict]:
    """One decode step of the naive parity reference (see module docstring).

    Deliberately written with explicit broadcasting and sums instead of the
    port's einsums, and as a flat dict state, so the oracle checks semantics
    rather than shared code paths.
    """
    d_inner = params["d_inner"]
    nheads = params["nheads"]
    headdim = params["headdim"]
    d_state = params["d_state"]
    rank = params["mimo_rank"]
    n_ang = params["num_rope_angles"]
    a_floor = 1e-4  # modules/mamba3.py:57 default A_floor (production does not override)
    batch = u.shape[0]

    proj = F.linear(u, params["in_proj.weight"])  # bf16 [B, d_in_proj]
    o1 = 2 * d_inner + 2 * d_state * rank
    z = proj[:, :d_inner]
    x = proj[:, d_inner : 2 * d_inner]
    b_raw = proj[:, 2 * d_inner : 2 * d_inner + d_state * rank]
    c_raw = proj[:, 2 * d_inner + d_state * rank : o1]
    dd_dt = proj[:, o1 : o1 + nheads]
    dd_a = proj[:, o1 + nheads : o1 + 2 * nheads]
    trap_raw = proj[:, o1 + 2 * nheads : o1 + 3 * nheads]
    ang_raw = proj[:, o1 + 3 * nheads :]

    a_val = dd_a.to(torch.float32)
    a_val = torch.where(a_val >= 0, 1.0 + a_val, torch.reciprocal(1.0 - a_val))
    a_mat = (-a_val).clamp(max=-a_floor)
    dt = F.softplus(dd_dt.to(torch.float32) + params["dt_bias"])
    trap = torch.sigmoid(trap_raw)

    b4 = b_raw.reshape(batch, rank, 1, d_state)
    c4 = c_raw.reshape(batch, rank, 1, d_state)
    b_n = _rms_ref(b4, params["B_norm.weight"], 1e-5).expand(batch, rank, nheads, d_state)
    c_n = _rms_ref(c4, params["C_norm.weight"], 1e-5).expand(batch, rank, nheads, d_state)
    x4 = x.reshape(batch, nheads, headdim)
    z4 = z.reshape(batch, nheads, headdim)
    ang4 = ang_raw.unsqueeze(1).expand(batch, nheads, n_ang)

    angle_proj = ang4.to(torch.float32)
    angle = state["angle"] + torch.tanh(angle_proj) * dt[:, :, None] * math.pi
    bias_q = params["C_bias"].permute(1, 0, 2)
    bias_k = params["B_bias"].permute(1, 0, 2)
    q = c_n.to(torch.float32) + bias_q[None]
    k = b_n.to(torch.float32) + bias_k[None]
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)
    half = d_state // 2
    if half > n_ang:
        pad = (batch, 1, nheads, half - n_ang)
        cos = torch.cat([cos, torch.ones(pad, dtype=cos.dtype)], dim=-1)
        sin = torch.cat([sin, torch.zeros(pad, dtype=sin.dtype)], dim=-1)
    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]
    q_rot = torch.cat([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1).to(u.dtype)
    k_rot = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1).to(u.dtype)

    alpha = torch.exp(a_mat * dt)
    beta = (1.0 - trap.to(torch.float32)) * dt * alpha
    gamma = trap.to(torch.float32) * dt
    x_proj = params["mimo_x"].permute(1, 0, 2)
    x_vals = x4.to(torch.float32)[:, None] * x_proj[None]
    xs_vals = state["v"].to(torch.float32)[:, None] * x_proj[None]
    xbt_state = (
        (x_vals * gamma[:, None, :, None]).unsqueeze(-1)
        * k_rot.to(torch.float32)[:, :, :, None, :]
    ).sum(dim=1)
    xbt_prev = (
        (xs_vals * beta[:, None, :, None]).unsqueeze(-1)
        * state["k"].to(torch.float32)[:, :, :, None, :]
    ).sum(dim=1)
    new_state = state["ssm"] * alpha[:, :, None, None] + xbt_state + xbt_prev
    out_r = (new_state[:, None] * q_rot.to(torch.float32)[:, :, :, None, :]).sum(dim=-1)
    out_r = out_r + x_vals * params["D"][None, None, :, None]
    z_vals = z4.to(torch.float32)[:, None] * params["mimo_z"].permute(1, 0, 2)[None]
    out_r = out_r * z_vals * torch.sigmoid(z_vals)
    out_proj_w = params["mimo_o"].permute(1, 0, 2)
    out = (out_r * out_proj_w[None]).sum(dim=1)
    out_b = out.reshape(batch, d_inner).to(u.dtype)
    y = F.linear(out_b, params["out_proj.weight"])

    nxt = {
        "angle": angle,
        "ssm": new_state,
        "k": k_rot,
        "v": x4,
    }
    return y, nxt


def _naive_prefill(
    x: torch.Tensor,
    params: dict,
    state: dict,
) -> tuple[torch.Tensor, dict]:
    steps = []
    states = []
    current = state
    for t in range(x.shape[1]):
        y, current = _naive_mamba3_step(x[:, t], params, current)
        steps.append(y)
        states.append(current)
    y_all = torch.stack(steps, dim=1)
    return y_all, current


def _states_to_float(state: CPUMamba3State) -> dict:
    return {
        "angle": state.angle.detach().clone(),
        "ssm": state.ssm.detach().clone(),
        "k": state.k.detach().clone(),
        "v": state.v.detach().clone(),
    }


def test_source_commit_matches_production_pin():
    assert MAMBA3_CPU_SOURCE_COMMIT == production_mamba3.MAMBA3_UPSTREAM_COMMIT
    assert Mamba3MIMOConfigCPU is CPUConfig


def test_core_param_names_match_checkpoint():
    # The checkpoint config is intentionally not committed. Validate the
    # architectural key schema in CI using the same module with a tiny config;
    # retain the checkpoint-specific shape path when the local file exists.
    if Path(CHECKPOINT_CONFIG).is_file():
        hybrid_config = IQHybridConfig.from_json(CHECKPOINT_CONFIG)
        mamba_config = CPUConfig(**vars(hybrid_config.mamba3))
    else:
        mamba_config = _tiny_config()
    model = Mamba3CPU(mamba_config, layer_idx=0)
    names = set(model.core.state_dict().keys())
    assert names == CHECKPOINT_CORE_PARAM_NAMES


def test_zero_weights_zero_output():
    config = _tiny_config()
    model = Mamba3CPU(config, dtype=torch.bfloat16)
    with torch.no_grad():
        model.core.B_bias.zero_()
        model.core.C_bias.zero_()
    torch.manual_seed(7)
    x = torch.randn(2, config.d_model, dtype=torch.bfloat16)
    state = CPUMamba3State.create(batch=2, config=config)
    y1, state = model.step(x, state)
    y2, state = model.step(x, state)
    assert torch.count_nonzero(y1) == 0
    assert torch.count_nonzero(y2) == 0
    assert torch.count_nonzero(state.angle) == 0
    assert torch.count_nonzero(state.ssm) == 0
    assert torch.count_nonzero(state.k) == 0
    assert torch.count_nonzero(state.v) == 0


def test_cache_shapes():
    config = _tiny_config()
    state = CPUMamba3State.create(batch=3, config=config)
    dims = _tiny_dims(config)
    assert state.angle.shape == (3, dims["nheads"], dims["num_rope_angles"])
    assert state.angle.dtype == torch.float32
    assert state.ssm.shape == (3, dims["nheads"], config.headdim, config.d_state)
    assert state.ssm.dtype == torch.float32
    assert state.k.shape == (3, config.mimo_rank, dims["nheads"], config.d_state)
    assert state.k.dtype == torch.bfloat16
    assert state.v.shape == (3, dims["nheads"], config.headdim)
    assert state.v.dtype == torch.bfloat16


def test_prefill_matches_incremental():
    config = _tiny_config()
    model = Mamba3CPU(config, dtype=torch.bfloat16)
    _randomize_core_params(model.core, seed=0)
    torch.manual_seed(3)
    batch, steps = 2, 9
    x = torch.randn(batch, steps, config.d_model, dtype=torch.bfloat16)

    y_prefill, state_prefill = model.prefill(x)

    state = CPUMamba3State.create(batch=batch, config=config)
    ys = []
    for t in range(steps):
        y, state = model.step(x[:, t], state)
        ys.append(y)
    y_seq = torch.stack(ys, dim=1)

    assert torch.allclose(y_prefill, y_seq, rtol=0, atol=1e-5)
    assert torch.allclose(state_prefill.angle, state.angle, rtol=0, atol=1e-5)
    assert torch.allclose(state_prefill.ssm, state.ssm, rtol=0, atol=1e-5)
    assert torch.allclose(state_prefill.k.float(), state.k.float(), rtol=0, atol=1e-5)
    assert torch.allclose(state_prefill.v.float(), state.v.float(), rtol=0, atol=1e-5)


def test_parity_oracle_matches_naive_recurrence():
    config = _tiny_config()
    model = Mamba3CPU(config, dtype=torch.bfloat16)
    _randomize_core_params(model.core, seed=1)
    params = _reference_params_from(model.core, config)
    torch.manual_seed(2)
    batch, steps = 2, 7
    x = torch.randn(batch, steps, config.d_model, dtype=torch.bfloat16)

    y_port, state_port = model.prefill(x)

    init = {
        "angle": torch.zeros(batch, params["nheads"], params["num_rope_angles"]),
        "ssm": torch.zeros(batch, params["nheads"], config.headdim, config.d_state),
        "k": torch.zeros(batch, config.mimo_rank, params["nheads"], config.d_state, dtype=torch.bfloat16),
        "v": torch.zeros(batch, params["nheads"], config.headdim, dtype=torch.bfloat16),
    }
    y_ref, state_ref = _naive_prefill(x, params, init)

    y_diff = (y_port.float() - y_ref.float()).abs().max().item()
    angle_diff = (state_port.angle - state_ref["angle"]).abs().max().item()
    ssm_diff = (state_port.ssm - state_ref["ssm"]).abs().max().item()
    k_diff = (state_port.k.float() - state_ref["k"].float()).abs().max().item()
    v_diff = (state_port.v.float() - state_ref["v"].float()).abs().max().item()
    worst = max(y_diff, angle_diff, ssm_diff, k_diff, v_diff)
    assert worst <= PARITY_ATOL, (
        f"parity drift {worst:.3e} > {PARITY_ATOL:.1e} "
        f"(y={y_diff:.3e}, angle={angle_diff:.3e}, ssm={ssm_diff:.3e}, "
        f"k={k_diff:.3e}, v={v_diff:.3e})"
    )


def test_logprob_improves_on_tiny_corpus():
    config = _tiny_config()
    vocab = 8
    pattern = torch.randint(
        0, vocab, (8,), generator=torch.Generator().manual_seed(11)
    )
    corpus = pattern.repeat(4)
    assert corpus.numel() >= 5

    model = Mamba3CPU(config, dtype=torch.float32)
    embed = nn.Embedding(vocab, config.d_model)
    head = nn.Linear(config.d_model, vocab)
    torch.manual_seed(5)
    for module in (model, embed, head):
        for param in module.parameters():
            param.data.normal_(std=0.3)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embed.parameters()) + list(head.parameters()),
        lr=0.03,
    )

    def average_logprob() -> float:
        with torch.no_grad():
            x = embed(corpus[:-1].unsqueeze(0))
            y, _ = model.prefill(x)
            logits = head(y.squeeze(0))
            return -F.cross_entropy(logits, corpus[1:]).item()

    initial = average_logprob()
    for _ in range(200):
        optimizer.zero_grad()
        x = embed(corpus[:-1].unsqueeze(0))
        y, _ = model.prefill(x)
        logits = head(y.squeeze(0))
        loss = F.cross_entropy(logits, corpus[1:])
        loss.backward()
        optimizer.step()
    final = average_logprob()

    assert initial < final - 0.5, f"logprob {initial:.3f} -> {final:.3f} (need +0.5)"


def test_clone_drop_state_roundtrip():
    config = _tiny_config()
    model = Mamba3CPU(config, dtype=torch.bfloat16)
    _randomize_core_params(model.core, seed=4)
    torch.manual_seed(6)
    x = torch.randn(2, config.d_model, dtype=torch.bfloat16)

    _, state = model.step(x, CPUMamba3State.create(batch=2, config=config))
    assert state.ssm.abs().sum() > 0

    cloned = state.clone_state()
    for field in ("angle", "ssm", "k", "v"):
        assert torch.equal(getattr(state, field), getattr(cloned, field))

    y_original, state_original = model.step(x, state)
    y_cloned, state_cloned = model.step(x, cloned)
    assert torch.equal(y_original, y_cloned)
    for field in ("angle", "ssm", "k", "v"):
        assert torch.equal(
            getattr(state_original, field), getattr(state_cloned, field)
        )

    snapshot = {
        field: getattr(state, field).clone()
        for field in ("angle", "ssm", "k", "v")
    }
    cloned.angle += 1.0
    assert not torch.equal(cloned.angle, snapshot["angle"])
    for field in ("angle", "ssm", "k", "v"):
        assert torch.equal(getattr(state, field), snapshot[field])

    fresh = CPUMamba3State.create(batch=2, config=config)
    dropped = state.clone_state().drop_state()
    for field in ("angle", "ssm", "k", "v"):
        assert torch.equal(getattr(dropped, field), getattr(fresh, field))

    y_fresh, _ = model.step(x, fresh)
    y_dropped, _ = model.step(x, dropped)
    assert torch.equal(y_fresh, y_dropped)
