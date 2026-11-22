"""Offline tests for the OpenAI-compatible local streaming server (Task 11).

Boots the ThreadingHTTPServer in-process over a tiny synthetic
CPUHybridForCausalLM with a FakeBucketClient-backed StreamingWeightProvider
(Task 6 test pattern) and a fixture tokenizer over the model's 32-token
vocab. Two servers: one whose greedy path always emits the fixture eos
(first token -> finish_reason "stop"), one that never emits it
(finish_reason "length", max_tokens honored).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from iq_local.cache import LocalCache
from iq_local.index import RunIndex
from iq_local.provider import StreamingWeightProvider

EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
ALPHABET = [chr(32 + i) for i in range(32)]


class FakeBucketClient:
    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = dict(mapping)
        self.calls: list[str] = []

    def download(self, files) -> None:
        for remote, dest in files:
            self.calls.append(remote)
            src = self.mapping.get(remote)
            if src is None:
                raise KeyError(remote)
            import shutil

            shutil.copyfile(src, dest)


class TinyTokenizer:
    """Fixture tokenizer over the 32-token synthetic vocab."""

    def __init__(self, eos_token_id: int = 0) -> None:
        self.eos_token_id = eos_token_id
        self._eot_id = 30

    def __len__(self) -> int:
        return 32

    def convert_tokens_to_ids(self, token: str):
        if token == "<|eot_id|>":
            return self._eot_id
        return None

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        text = "".join(
            f"{message['role']}:{message['content']}?" for message in messages
        )
        if add_generation_prompt:
            text += "assistant:"
        ids = [ord(char) % 32 for char in text] or [0]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        specials = {self.eos_token_id, self._eot_id}
        chars = [
            ALPHABET[token_id]
            for token_id in ids
            if 0 <= token_id < 32 and not (skip_special_tokens and token_id in specials)
        ]
        return "".join(chars)


def _tiny_config():
    from iq_model.architecture import HybridLayerType, HybridSchedule
    from iq_model.config import IQModelConfig
    from iq_model.hybrid import IQHybridConfig
    from iq_model.mlp import StableLatentMoEConfig
    from iq_model.mlp.moe import RoutedMoEConfig
    from iq_model.residual.attnres import BlockAttnResConfig
    from iq_model.state.mamba3 import Mamba3MIMOConfig

    return IQHybridConfig(
        model=IQModelConfig(
            vocab_size=32,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
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
            d_model=64,
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
            hidden_size=64,
            expert_intermediate_size=64,
            num_experts=16,
            top_k=8,
        ),
        moe_variant="stable_latent",
        stable_moe=StableLatentMoEConfig(
            hidden_size=64,
            latent_size=64,
            expert_intermediate_size=64,
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
        attnres=BlockAttnResConfig(hidden_size=64, num_layers=4, block_size=2),
    )


def _make_provider(tmp_path: Path) -> StreamingWeightProvider:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True)
    torch.manual_seed(3)
    weight_map: dict[str, str] = {}
    mapping: dict[str, Path] = {}
    for expert_idx in range(16):
        filename = f"model-moe-000-expert-{expert_idx:03d}.safetensors"
        keys = [
            f"layers.0.moe.routed_experts.{expert_idx}.{proj}.weight"
            for proj in EXPERT_PROJS
        ]
        save_file(
            {
                key: (torch.randn(64, 64, dtype=torch.float32) * 0.05)
                .to(torch.bfloat16)
                .contiguous()
                for key in keys
            },
            str(fixtures / filename),
        )
        for key in keys:
            weight_map[key] = filename
        mapping[filename] = fixtures / filename
    run_index = RunIndex.__new__(RunIndex)
    run_index.weight_map = weight_map
    run_index.files = sorted(set(weight_map.values()))
    run_index._build_experts_by_stage()
    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 30)
    return StreamingWeightProvider(
        run_index, cache, FakeBucketClient(mapping), max_ram_experts=4
    )


def _tiny_model():
    from iq_local.cpu_hybrid import CPUHybridForCausalLM

    torch.manual_seed(5)
    model = CPUHybridForCausalLM(_tiny_config())
    model.eval()
    return model


def _boot(runner, tokenizer, model_name: str):
    from iq_local.server import build_handler

    from http.server import ThreadingHTTPServer

    handler = build_handler(runner, tokenizer, model_name)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture()
def servers(tmp_path):
    from iq_local.runner import HybridRunner

    provider_a = _make_provider(tmp_path / "a")
    provider_b = _make_provider(tmp_path / "b")
    model_a = _tiny_model()
    model_a.lm_head.weight.data.copy_(
        model_a.lm_head.weight.data[0].unsqueeze(0).repeat(32, 1)
    )
    model_b = _tiny_model()
    model_b.lm_head.weight.data.copy_(
        model_b.lm_head.weight.data[0].unsqueeze(0).repeat(32, 1)
    )
    url_a, server_a = _boot(
        HybridRunner(model_a, provider_a), TinyTokenizer(eos_token_id=0), "IQ/runs/test-a"
    )
    url_b, server_b = _boot(
        HybridRunner(model_b, provider_b), TinyTokenizer(eos_token_id=31), "IQ/runs/test-b"
    )
    yield {"a": url_a, "b": url_b}
    server_a.shutdown()
    server_b.shutdown()


def test_server_offline_contract(servers):
    status, health = _get(servers["a"] + "/health")
    assert status == 200
    assert health["status"] == "ok"
    assert health["model"] == "IQ/runs/test-a"
    assert isinstance(health["counters"], dict)

    status, body = _post(
        servers["a"] + "/v1/chat/completions",
        {
            "model": "IQ/runs/test-a",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 4,
        },
    )
    assert status == 200
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 1

    status, error = _post(servers["a"] + "/v1/chat/completions", {"messages": []})
    assert status == 400
    status, _ = _get(servers["a"] + "/unknown")
    assert status == 404


def test_server_honors_max_tokens(servers):
    status, body = _post(
        servers["b"] + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
    )
    assert status == 200
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3
    assert isinstance(body["choices"][0]["message"]["content"], str)
