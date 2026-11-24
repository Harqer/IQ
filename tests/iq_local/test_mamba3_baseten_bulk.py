"""No-credit contract checks for Baseten custom model bulk inference packaging."""
from __future__ import annotations

import io
import json

import pytest

from iq_local.mamba3_csa_hca import MODEL_ID, PHASE5E_SHA256
from scripts.benchmark_iq_mamba3_csa_hca import RemoteMamba3Runner


def test_remote_bulk_response_must_match_exact_original_model_and_adapter(monkeypatch):
    calls = []

    def fake_request(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps({
            "source_model": MODEL_ID,
            "checkpoint_sha256": PHASE5E_SHA256,
            "mode": "hybrid",
            "generated_token_ids": [12366, 13],
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_request)
    runner = RemoteMamba3Runner("https://model-example.api.baseten.co/development/predict", "secret")
    runner.mode = "hybrid"
    assert runner.generate([128000, 2, 3], 8, stop_tokens={128009}) == [12366, 13]
    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert payload == {"prompt_ids": [128000, 2, 3], "max_new_tokens": 8,
                       "stop_tokens": [128009], "mode": "hybrid"}
    assert request.get_header("Authorization") == "Api-Key secret"
    assert "secret" not in request.data.decode()


def test_remote_wrong_checkpoint_aborts_bulk(monkeypatch):
    def fake_request(request, timeout):
        return io.BytesIO(json.dumps({
            "source_model": MODEL_ID,
            "checkpoint_sha256": "other",
            "mode": "hybrid",
            "generated_token_ids": [12366],
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_request)
    runner = RemoteMamba3Runner("https://model-example.api.baseten.co/development/predict", "secret")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.generate([128000], 8)
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteMamba3Runner("http://example.org/predict", "secret")
    with pytest.raises(ValueError, match="BASETEN_API_KEY"):
        RemoteMamba3Runner("https://model-example.api.baseten.co/development/predict", "")


def test_staging_guards_weights_and_preserves_package_layout(tmp_path, monkeypatch):
    from scripts import build_mamba3_baseten_truss as builder
    fake = tmp_path / "adapter.pt"
    fake.write_bytes(b"fake")
    monkeypatch.setattr(builder, "_sha256", lambda path: PHASE5E_SHA256)
    out = tmp_path / "staged"
    builder.stage(fake, out)
    assert (out / "config.yaml").is_file()
    assert (out / "model/model.py").is_file()
    assert (out / "packages/iq_local/mamba3_csa_hca.py").is_file()
    assert (out / "packages/experiments/mamba3-csa-hca"
            "/iq_mamba3_csa_hca_phase5e/iq_hybrid_cpu.py").is_file()
    assert (out / "data/iq_phase5e_attention_candidate.pt").read_bytes() == b"fake"
    with pytest.raises(FileExistsError):
        builder.stage(fake, out)
