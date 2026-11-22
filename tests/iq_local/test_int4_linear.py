import io

import pytest
import torch
from torch.nn import functional as F

from iq_local.int4_linear import PackedInt4Linear


@pytest.mark.parametrize('shape', [(16, 64), (7, 34), (8, 32)])
def test_cpu_int4_shape_and_bounded_error(shape):
    torch.manual_seed(123)
    w = torch.randn(*shape, dtype=torch.float32) / 6
    x = torch.randn(3, shape[1], dtype=torch.float32)
    c = PackedInt4Linear.from_weight(w, group_size=32, tile_rows=3)
    out = c(x)
    ref = F.linear(x, w)
    assert out.shape == ref.shape
    assert (out - ref).norm() / ref.norm() < 0.17
    assert c.packed_nbytes < w.numel() * w.element_size()


def test_exact_zero_columns_kept_in_place():
    w = torch.randn(8, 128, dtype=torch.float32)
    w[:, 64:] = 0
    x = torch.randn(2, 128, dtype=torch.float32)
    c = PackedInt4Linear.from_weight(w, group_size=32)
    assert c.active_features == 64
    assert c.in_features == 128
    relative = (c(x) - F.linear(x, w)).norm() / F.linear(x, w).norm()
    assert relative < 0.20


def test_zero_weight_with_bias_exact():
    w = torch.zeros(4, 64, dtype=torch.bfloat16)
    bias = torch.tensor([1, -2, 3, -4], dtype=torch.bfloat16)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    c = PackedInt4Linear.from_weight(w, bias=bias)
    assert c.active_features == 0
    torch.testing.assert_close(c(x), F.linear(x, w, bias), rtol=0, atol=0)


def test_cpu_packed_serialization_roundtrip():
    torch.manual_seed(4)
    w = torch.randn(8, 64)
    c = PackedInt4Linear.from_weight(w)
    payload = io.BytesIO()
    torch.save(c.state_dict(), payload)
    payload.seek(0)
    d = torch.load(payload, weights_only=True)
    original = c.state_dict()
    assert set(d) == set(original)
    for k in original:
        torch.testing.assert_close(d[k], original[k], rtol=0, atol=0)
