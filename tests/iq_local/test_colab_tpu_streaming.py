"""Small CPU-side parity tests for the new orchestration; TPU not required."""
import torch
from iq_model.mlp import StableLatentMoEConfig
from iq_local.cpu_hybrid import FunctionalStableLatentMoELayer
from scripts.iq235_colab_tpu import moe_hidden, state_to
from iq_model.state.mamba3_cpu import CPUMamba3State

def test_host_routing_matches_reference():
    torch.manual_seed(9)
    cfg = StableLatentMoEConfig(hidden_size=8, latent_size=4,
        expert_intermediate_size=8, num_experts=4, top_k=2,
        num_shared_experts=1, expert_bias=False)
    layer = FunctionalStableLatentMoELayer(cfg).eval()
    experts = [(torch.randn(8,4), torch.randn(8,4), torch.randn(4,8))
               for _ in range(4)]
    def provider(stage, index):
        return experts[index]
    x = torch.randn(1,3,8)
    expected = layer(x, provider).hidden_states
    torch.testing.assert_close(moe_hidden(layer, x, provider), expected)

def test_state_to_preserves_fields_and_dtypes():
    state = CPUMamba3State(torch.randn(2), torch.randn(2),
                           torch.randn(2).bfloat16(), torch.randn(2).bfloat16())
    result = state_to(state, "cpu")
    for name in ("angle", "ssm", "k", "v"):
        torch.testing.assert_close(getattr(state, name), getattr(result, name))
