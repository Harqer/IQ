import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "iq_mamba3_parity_job.py"
spec = importlib.util.spec_from_file_location("iq_mamba3_parity_job", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_preflight_exact_zero_and_summary_structure(tmp_path):
    work = tmp_path / "parity"
    summary = module.preflight(work_dir=work, seed_base=20261025)
    assert summary["mode"] == "preflight"
    assert summary["tolerance"] == 1e-3
    cases = summary["cases"]
    assert [case["name"] for case in cases] == ["prefill", "decode"]
    assert cases[0]["input_shape"] == [4, 64]
    assert cases[1]["input_shape"] == [1, 1]
    for case in cases:
        assert case["verdict"] == "pass"
        assert case["comparisons"], case
        for record in case["comparisons"]:
            assert isinstance(record["max_abs_diff"], float)
            assert record["max_abs_diff"] == 0.0
            assert isinstance(record["shape"], list)
            assert isinstance(record["tensor"], str) and record["tensor"]
    json.dumps(summary)

    prefill_payload = json.loads((work / "prefill_l0.json").read_text())
    decode_payload = json.loads((work / "decode_l0.json").read_text())
    assert prefill_payload["layer"] == 0 and prefill_payload["case"] == "prefill"
    assert decode_payload["layer"] == 0 and decode_payload["case"] == "decode"
    for payload in (prefill_payload, decode_payload):
        assert payload["input"]["shape"][0] == payload["input_shape"][0]
        assert payload["input"]["shape"][1] == payload["input_shape"][1]
        for block in ("input", "output", "state"):
            assert block in payload
        for record in payload["comparisons"]:
            assert isinstance(record["max_abs_diff"], float)
            assert record["max_abs_diff"] == 0.0


def test_selected_parity_layers_follow_schedule():
    from iq_model.hybrid import IQHybridConfig

    config = IQHybridConfig.from_json(str(REPO / "checkpoints" / "iq" / "iq_config.json"))
    schedule = list(config.schedule.layers)
    layers = module.select_parity_layers(schedule)
    assert len(layers) == 2
    assert layers[0] == 0
    assert schedule[layers[0]] == "mamba3"
    second = layers[1]
    assert schedule[second] == "mamba3"
    prev_mamba = max(i for i in range(second) if schedule[i] == "mamba3")
    assert any(
        schedule[j] in {"csa", "hca", "dense_attention"}
        for j in range(prev_mamba + 1, second)
    )


def test_load_layer_weights_reads_checkpoint_layout(tmp_path):
    from safetensors.torch import save_file

    shard = tmp_path / "model-00001.safetensors"
    in_proj = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    a_log = torch.full((2,), 0.5)
    save_file(
        {
            "layers.3.mamba.core.in_proj.weight": in_proj,
            "layers.3.mamba.core.A_log": a_log,
        },
        str(shard),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.3.mamba.core.in_proj.weight": "model-00001.safetensors",
                    "layers.3.mamba.core.A_log": "model-00001.safetensors",
                }
            }
        )
    )
    weights = module.load_layer_weights(tmp_path, layer_idx=3)
    assert set(weights.keys()) == {"core.in_proj.weight", "core.A_log"}
    assert torch.equal(weights["core.in_proj.weight"], in_proj)
    assert torch.equal(weights["core.A_log"], a_log)


def test_execute_requires_h200(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_assert_h200", lambda: (_ for _ in ()).throw(SystemExit("no cuda")))
    with pytest.raises(SystemExit):
        module.execute(work_dir=tmp_path, seed_base=20261025)
