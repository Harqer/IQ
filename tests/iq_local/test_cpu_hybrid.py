import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

_META_MODEL_CACHE: dict[str, tuple] = {}


def _checkpoint_key_sets() -> tuple[set[str], set[str]]:
    cache_key = "meta_model"
    if cache_key not in _META_MODEL_CACHE:
        from iq_local.cpu_hybrid import CPUHybridForCausalLM
        from iq_model.hybrid import IQHybridConfig

        config = IQHybridConfig.from_json(
            str(REPO / "checkpoints" / "iq" / "iq_config.json")
        )
        model = CPUHybridForCausalLM(config, device="meta")
        produced = set(model.state_dict().keys())
        expected = set(
            json.loads(
                (REPO / "checkpoints" / "iq" / "model.safetensors.index.json").read_text()
            )["weight_map"]
        )
        _META_MODEL_CACHE[cache_key] = (produced, expected)
    return _META_MODEL_CACHE[cache_key]


def test_state_dict_keys_match_checkpoint_index():
    produced, expected = _checkpoint_key_sets()
    routed = {key for key in expected if ".routed_experts." in key}
    expected_non_expert = expected - routed
    missing = sorted(expected_non_expert - produced)
    extra = sorted(produced - expected_non_expert)
    assert not missing, f"missing {len(missing)}: {missing[:20]}"
    assert not extra, f"extra {len(extra)}: {extra[:20]}"


def test_routed_expert_keys_are_provider_served_only():
    produced, expected = _checkpoint_key_sets()
    routed = {key for key in expected if ".routed_experts." in key}
    assert routed, "checkpoint weight_map must contain routed expert weights"
    assert not (routed & produced), "routed experts must not live in model state"


def test_functional_moe_matches_production_bitwise():
    from iq_local.cpu_hybrid import FunctionalStableLatentMoE
    from iq_model.mlp import StableLatentMoE, StableLatentMoEConfig

    torch.manual_seed(0)
    config = StableLatentMoEConfig(
        hidden_size=8,
        latent_size=4,
        expert_intermediate_size=6,
        num_experts=5,
        top_k=2,
        num_shared_experts=1,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        routed_scaling_factor=2.5,
        rms_norm_eps=1e-5,
        expert_bias=False,
        router_bias=False,
    )
    production = StableLatentMoE(config).to(dtype=torch.bfloat16)
    functional = FunctionalStableLatentMoE(config, stage=3).to(dtype=torch.bfloat16)

    production_keys = set(production.state_dict().keys())
    routed_keys = {key for key in production_keys if key.startswith("routed_experts.")}
    shared_keys = production_keys - routed_keys
    assert set(functional.state_dict().keys()) == shared_keys

    incompatible = functional.load_state_dict(
        production.state_dict(),
        strict=False,
    )
    assert set(incompatible.unexpected_keys) == routed_keys
    assert incompatible.missing_keys == [], incompatible.missing_keys

    def expert_fn(stage: int, expert_index: int):
        assert stage == 3
        expert = production.routed_experts[expert_index]
        return (
            expert.gate_proj.weight,
            expert.up_proj.weight,
            expert.down_proj.weight,
        )

    x = torch.randn(2, 3, 8, dtype=torch.float32).to(torch.bfloat16)
    production_output = production(x)
    functional_output = functional(x, expert_fn)

    for field in (
        "hidden_states",
        "raw_router_scores",
        "selected_experts",
        "selected_weights",
        "expert_counts",
        "routing_bias",
    ):
        left = getattr(production_output, field)
        right = getattr(functional_output, field)
        assert left.dtype == right.dtype, field
        assert torch.equal(left, right), field


def test_functional_moe_rejects_expert_bias_and_missing_expert_fn():
    from iq_local.cpu_hybrid import FunctionalStableLatentMoE
    from iq_model.hybrid import HybridModelError
    from iq_model.mlp import StableLatentMoEConfig

    config = StableLatentMoEConfig(
        hidden_size=8,
        latent_size=4,
        expert_intermediate_size=6,
        num_experts=5,
        top_k=2,
        expert_bias=True,
    )
    with pytest.raises(HybridModelError):
        FunctionalStableLatentMoE(config)

    ok_config = StableLatentMoEConfig(
        hidden_size=8,
        latent_size=4,
        expert_intermediate_size=6,
        num_experts=5,
        top_k=2,
    )
    module = FunctionalStableLatentMoE(ok_config, stage=0)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    with pytest.raises(HybridModelError):
        module(x, None)


def test_cpu_mamba_layer_rejects_varlen_contract():
    import types

    from iq_local.cpu_hybrid import CPUMamba3ResidualLayer
    from iq_model.hybrid import HybridModelError

    model_config = types.SimpleNamespace(
        hidden_size=16,
        rms_norm_eps=1e-5,
        residual_dropout=0.0,
    )
    mamba_config = types.SimpleNamespace(
        d_model=16,
        headdim=8,
        d_state=4,
        mimo_rank=2,
        num_layers=2,
        expand=2.0,
        chunk_size=4,
        rope_fraction=0.5,
        outproj_norm=False,
    )
    layer = CPUMamba3ResidualLayer(
        model_config,
        mamba_config,
        mamba_layer_idx=0,
        dtype=torch.bfloat16,
        device="cpu",
    )
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    y = layer(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16
    with pytest.raises(HybridModelError):
        layer(x, attention_mask=torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(HybridModelError):
        layer(x, document_ids=torch.zeros(2, 4, dtype=torch.int64))
