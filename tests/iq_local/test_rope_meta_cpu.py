"""Regression for meta-device, non-persistent CSA/HCA RoPE buffers."""
from types import SimpleNamespace

import torch
from torch import nn

from iq_model.position import InterleavedRotaryEmbedding
from iq_local.int4_loading import _materialize_meta_rope_buffers


def test_rope_meta_buffer_matches_native_constructor():
    dim, max_positions, base = 64, 2048, 160000.0
    reference = InterleavedRotaryEmbedding(dim, max_positions, base)
    model = nn.Module()
    model.add_module('attention_rope', InterleavedRotaryEmbedding(dim, max_positions, base).to('meta'))
    assert model.attention_rope.inv_freq.is_meta
    assert 'attention_rope.inv_freq' not in model.state_dict()  # persistent=False

    config = SimpleNamespace(compressed_context=SimpleNamespace(
        partial_rotary_dim=dim, max_position_embeddings=max_positions,
        compress_rope_theta=base))
    assert _materialize_meta_rope_buffers(model, config) == 1
    assert _materialize_meta_rope_buffers(model, config) == 0
    assert not model.attention_rope.inv_freq.is_meta
    assert torch.equal(model.attention_rope.inv_freq, reference.inv_freq)
    p = torch.tensor([[0, 1, 31, 255]], dtype=torch.long)
    expected = reference.cos_sin(p, dtype=torch.bfloat16, device=torch.device('cpu'))
    actual = model.attention_rope.cos_sin(p, dtype=torch.bfloat16, device=torch.device('cpu'))
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
