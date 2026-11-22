import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

from iq_local.int4_linear import PackedInt4Linear


def test_meta_stream_loader_on_mock(monkeypatch):
    class SmallModel(nn.Module):
        def __init__(self, cfg, dtype=torch.bfloat16, device='meta'):
            super().__init__()
            self.global_projection = nn.Linear(64, 8, bias=False, dtype=dtype, device=device)
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(64, 16, bias=False, dtype=dtype, device=device)),
                nn.Sequential(nn.Linear(16, 8, bias=False, dtype=dtype, device=device)),
            ])

    fake_cpu = types.ModuleType('iq_local.cpu_hybrid')
    fake_cpu.CPUHybridForCausalLM = SmallModel
    monkeypatch.setitem(sys.modules, 'iq_local.cpu_hybrid', fake_cpu)
    from iq_local import int4_loading

    cfg = SimpleNamespace(schedule=SimpleNamespace(layers=[SimpleNamespace(value='csa'),
                                                             SimpleNamespace(value='mamba3')]),
                          model=SimpleNamespace(tie_word_embeddings=False))
    tensors = {'global_projection.weight': torch.randn(8, 64).to(torch.bfloat16),
               'layers.0.0.weight': torch.randn(16, 64).to(torch.bfloat16),
               'layers.1.0.weight': torch.randn(8, 16).to(torch.bfloat16)}
    tensors['layers.0.0.weight'][:, 32:] = 0
    index = SimpleNamespace(weight_map={k: 'dummy' for k in tensors})

    class Provider:
        def load_global(self, key):
            return tensors[key]

        def load_layer_base(self, i):
            return {k: v for k, v in tensors.items() if k.startswith(f'layers.{i}.')}

    monkeypatch.setattr(int4_loading, 'CPUHybridForCausalLM', SmallModel)
    model, stats = int4_loading.build_cpu_int4_model(cfg, index, Provider(), min_elements=256)
    assert isinstance(model.layers[0][0], PackedInt4Linear)
    assert isinstance(model.layers[1][0], nn.Linear)
    assert stats['compressed_linears'] == 1
    assert stats['zero_columns'] == 32
    assert all(not p.is_meta for p in model.parameters())
    assert all(not b.is_meta for b in model.buffers())
    x = torch.randn(3, 64).to(torch.bfloat16)
    y = model.layers[0](x)
    assert y.shape == (3, 16)
