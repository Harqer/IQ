"""Small CPU checks for the optional Mamba-3 Phase 5e inference backend."""
import hashlib

import pytest
import torch

from iq_local import mamba3_csa_hca as hybrid


@pytest.fixture(scope="module")
def experiment():
    return hybrid._experiment()


def test_hash_compatible_with_phase5e_checksum(experiment):
    toy = experiment.MambaLMCPU(experiment._toy_config())
    h = hashlib.sha256()
    for name, value in toy.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().contiguous().numpy().tobytes())
    assert hybrid._foundation_hash(toy) == h.hexdigest()


def test_frozen_foundation_and_active_full_prefix(experiment):
    torch.manual_seed(21)
    torch.set_num_threads(2)
    cfg = experiment._toy_config()
    base = experiment.MambaLMCPU(cfg)
    mods = experiment.default_insertions(cfg)
    with torch.no_grad():
        for layer in mods.values():
            layer.out_proj.weight.normal_(std=0.003)
    runner = hybrid.Mamba3CSAHCARunner(base, mods, 0.02)
    tokens = [1, 12, 8, 51, 35, 7, 2, 18]
    with torch.inference_mode():
        raw = runner.next_logits(tokens, baseline=True)
        active = runner.next_logits(tokens)
        expected = base(torch.tensor([tokens]), runner.insertions)[0, -1].float()
    assert torch.equal(active, expected)
    assert float((active - raw).abs().max()) > 0
    generated = runner.generate(tokens, 2)
    expected_next = int(active.argmax())
    assert generated[0] == expected_next
    assert all(not p.requires_grad for p in base.parameters())
    assert all(not p.requires_grad for p in mods.parameters())


def test_baseline_mode_does_not_use_attention(experiment):
    model = experiment.MambaLMCPU(experiment._toy_config())
    modules = experiment.default_insertions(model.cfg)
    runner = hybrid.Mamba3CSAHCARunner(model, modules, 0.02, mode="baseline")
    tokens = [1, 2, 3, 4]
    assert torch.equal(runner.next_logits(tokens), runner.next_logits(tokens, baseline=True))
    with pytest.raises(ValueError, match="context"):
        runner.generate(tokens, 2048)
    with pytest.raises(ValueError, match="vocabulary"):
        runner.next_logits([999999])


def test_checkpoint_metadata_is_strict():
    good = dict(source_model=hybrid.MODEL_ID, placement_after_blocks=[8, 16],
                pilot_parent_sha256=hybrid.PILOT_SHA256,
                max_relative_norm=0.02, not_production_ready=True,
                attention_state_dict={"8.foo": torch.tensor(1)},
                source_model_sha256="e" * 64)
    assert hybrid._validate_metadata(good) == 0.02
    for change in (dict(placement_after_blocks=[9, 16]),
                   dict(pilot_parent_sha256="other"),
                   dict(max_relative_norm=float("nan")),
                   dict(not_production_ready=False)):
        wrong = dict(good, **change)
        with pytest.raises(ValueError):
            hybrid._validate_metadata(wrong)
