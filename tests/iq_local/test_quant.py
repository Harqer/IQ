import pytest
import torch

from iq_local.quant import GROUP, dequantize_int4, quantize_int4


def test_roundtrip_error_is_small():
    torch.manual_seed(0)
    tensor = torch.randn(256, 128, dtype=torch.float32) * 0.05
    packed = quantize_int4(tensor)
    assert packed.packed.dtype == torch.uint8
    assert packed.packed.shape == (256, 64)
    assert packed.scales.shape == (256, 2)
    out = dequantize_int4(packed, (256, 128), torch.bfloat16).float()
    error = (out - tensor).abs().max().item()
    assert error <= tensor.abs().max().item() / 8.0


def test_group_constant():
    assert GROUP == 64


def test_rejects_non_divisible_last_dim():
    tensor = torch.randn(8, 65, dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_int4(tensor)


def test_rejects_non_2d():
    tensor = torch.randn(8, 8, 8, dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_int4(tensor)
