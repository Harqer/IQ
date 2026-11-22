"""Regression: BF16 checkpoint MIMO factors with FP32 Mamba-3 recurrence."""

from copy import deepcopy

import pytest
import torch

from iq_model.state.mamba3_cpu import CPUConfig, Mamba3CPU


@pytest.mark.parametrize("outproj_norm", [False, True])
def test_bf16_checkpoint_mimo_factors_prefill_and_decode(outproj_norm: bool) -> None:
    config = CPUConfig(
        d_model=8,
        num_layers=1,
        d_state=8,
        headdim=4,
        mimo_rank=2,
        expand=2.0,
        rope_fraction=0.5,
        outproj_norm=outproj_norm,
    )
    model = Mamba3CPU(config, dtype=torch.bfloat16)
    with torch.no_grad():
        torch.manual_seed(12)
        model.core.in_proj.weight.normal_(std=0.05)
        model.core.out_proj.weight.normal_(std=0.05)
        for name in ("mimo_x", "mimo_z", "mimo_o"):
            factor = getattr(model.core, name)
            # The real checkpoint uses BF16 for tensors created in FP32.
            # assign=True loading can preserve the checkpoint's BF16 dtype.
            factor.data = factor.data.to(torch.bfloat16)

    reference = deepcopy(model)
    with torch.no_grad():
        for name in ("mimo_x", "mimo_z", "mimo_o"):
            factor = getattr(reference.core, name)
            factor.data = factor.data.to(torch.float32)

    assert model.core.mimo_o.dtype == torch.bfloat16
    assert reference.core.mimo_o.dtype == torch.float32

    torch.manual_seed(13)
    x = torch.randn(1, 3, config.d_model, dtype=torch.bfloat16)
    nxt = torch.randn(1, config.d_model, dtype=torch.bfloat16)

    with torch.no_grad():
        mixed_prefill, mixed_state = model.prefill(x)
        float_prefill, float_state = reference.prefill(x)
        mixed_decode, mixed_state = model.step(nxt, mixed_state)
        float_decode, float_state = reference.step(nxt, float_state)

    assert mixed_prefill.dtype == x.dtype
    assert mixed_decode.dtype == x.dtype
    assert torch.isfinite(mixed_prefill).all()
    assert torch.isfinite(mixed_decode).all()
    torch.testing.assert_close(mixed_prefill, float_prefill, rtol=0, atol=0)
    torch.testing.assert_close(mixed_decode, float_decode, rtol=0, atol=0)
    for name in ("angle", "ssm", "k", "v"):
        torch.testing.assert_close(
            getattr(mixed_state, name), getattr(float_state, name),
            rtol=0, atol=0,
        )
