import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def _tiny_config():
    from iq_model.architecture import HybridLayerType, HybridSchedule
    from iq_model.config import IQModelConfig
    from iq_model.hybrid import IQHybridConfig
    from iq_model.mlp import StableLatentMoEConfig
    from iq_model.mlp.moe import RoutedMoEConfig
    from iq_model.reasoning.recurrence import ReasoningRecurrenceConfig
    from iq_model.residual.attnres import BlockAttnResConfig
    from iq_model.state.mamba3 import Mamba3MIMOConfig

    return IQHybridConfig(
        model=IQModelConfig(
            vocab_size=32,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            rms_norm_eps=1e-5,
            residual_dropout=0.0,
            initializer_range=0.02,
        ),
        schedule=HybridSchedule(
            layers=(
                HybridLayerType.MAMBA3,
                HybridLayerType.DENSE_ATTENTION,
                HybridLayerType.MAMBA3,
                HybridLayerType.MOE,
            )
        ),
        mamba3=Mamba3MIMOConfig(
            d_model=32,
            num_layers=2,
            d_state=8,
            headdim=16,
            mimo_rank=2,
            expand=2.0,
            chunk_size=16,
            rope_fraction=0.5,
            outproj_norm=False,
        ),
        moe=RoutedMoEConfig(
            hidden_size=32,
            expert_intermediate_size=16,
            num_experts=16,
            top_k=8,
        ),
        moe_variant="stable_latent",
        stable_moe=StableLatentMoEConfig(
            hidden_size=32,
            latent_size=16,
            expert_intermediate_size=16,
            num_experts=16,
            top_k=8,
            num_shared_experts=1,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            routed_scaling_factor=2.5,
            rms_norm_eps=1e-5,
            expert_bias=False,
            router_bias=False,
        ),
        attnres=BlockAttnResConfig(hidden_size=32, num_layers=4, block_size=2),
        reasoning=ReasoningRecurrenceConfig(
            hidden_size=32,
            state_dim=8,
            transition_hidden_dim=8,
            max_steps=2,
            min_steps=1,
        ),
    )


class CountingProvider:
    def __init__(self):
        self.calls = []

    def load_expert(self, stage, expert_idx):
        self.calls.append((int(stage), int(expert_idx)))
        gate = torch.full((16, 16), 0.01 * (int(expert_idx) + 1), dtype=torch.bfloat16)
        up = torch.full((16, 16), 0.01, dtype=torch.bfloat16)
        down = torch.full((16, 16), 0.01, dtype=torch.bfloat16)
        return gate, up, down


def _force_constant_routing(model):
    router = model.layers[3].moe.router
    router.weight.data.copy_(
        (torch.arange(16, dtype=torch.float32).unsqueeze(1) * 0.5).repeat(1, 32).to(
            torch.bfloat16
        )
    )
    model.layers[3].moe.routing_bias.zero_()


def _tiny_model():
    from iq_local.cpu_hybrid import CPUHybridForCausalLM

    torch.manual_seed(11)
    model = CPUHybridForCausalLM(_tiny_config())
    _force_constant_routing(model)
    model.eval()
    return model


def test_select_experts_matches_production_route():
    from iq_local.cpu_hybrid import CPUHybridForCausalLM  # noqa: F401
    from iq_local.runner import HybridRunner
    from iq_model.mlp import StableLatentMoE

    model = _tiny_model()
    runner = HybridRunner(model, CountingProvider())
    production = StableLatentMoE(_tiny_config().stable_moe).to(dtype=torch.bfloat16)
    moe_layer = model.layers[3].moe
    moe_layer.router.load_state_dict(production.router.state_dict())
    moe_layer.routing_bias.copy_(production.routing_bias)

    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    _, production_selected, production_weights = production._route(
        x.reshape(-1, 32)
    )
    runner_weights, runner_selected = runner.select_experts(x, stage=0)
    assert torch.equal(runner_selected, production_selected)
    assert torch.equal(runner_weights, production_weights)


def test_generate_matches_model_forward():
    from iq_local.runner import HybridRunner

    model = _tiny_model()
    provider = CountingProvider()
    runner = HybridRunner(model, provider)
    generated = runner.generate([1, 2, 3], max_new_tokens=2)

    tokens = [1, 2, 3]
    expected = []
    for _ in range(2):
        logits = model.forward(
            torch.tensor([tokens]), expert_fn=provider.load_expert
        )
        next_token = int(logits[0, -1].argmax())
        expected.append(next_token)
        tokens.append(next_token)
    assert generated == expected


def test_expert_fetch_happens_between_layers():
    from iq_local.runner import HybridRunner

    model = _tiny_model()
    provider = CountingProvider()
    runner = HybridRunner(model, provider)
    runner.generate([1, 2, 3], max_new_tokens=2)

    stage_calls = [call for call in provider.calls if call[0] == 0]
    fetched = {expert for _, expert in stage_calls}
    assert fetched == set(range(8, 16))
    prefill_calls, decode_calls = stage_calls[:8], stage_calls[8:]
    assert {expert for _, expert in prefill_calls} == fetched
    assert [expert for _, expert in decode_calls] == [expert for _, expert in prefill_calls]
    assert len(decode_calls) == 8
    assert {stage for stage, _ in provider.calls} == {0}


def test_reset_clears_streaming_state():
    from iq_local.runner import HybridRunner

    model = _tiny_model()
    runner = HybridRunner(model, CountingProvider())
    first = runner.generate([1, 2, 3], max_new_tokens=2)
    assert runner._attn_cache
    assert runner._hidden_cache
    assert runner._mamba_states
    runner.reset()
    assert not runner._attn_cache
    assert not runner._hidden_cache
    assert not runner._mamba_states
    second = runner.generate([1, 2, 3], max_new_tokens=2)
    assert first == second
