"""Offline tests for the three-tier streaming weight provider (Task 6).

Covers the brief's five prose cases plus the dispatcher's binding constraints:
shard/int4 pinning, staged-temp cleanup on every failure path, and no LRU
accounting leaks. All offline; the FakeBucketClient is dict-mapped to fixture
files and copies them into whatever staging destination the provider requests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from iq_local.cache import CacheDiskFullError, LocalCache
from iq_local.index import RunIndex
from iq_local.provider import StreamingWeightProvider

GLOBAL_KEY = "lm_head.weight"
COMMON_KEY = "layers.0.moe.gate.weight"
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")


def _expert_keys(expert_idx: int) -> list[str]:
    return [
        f"layers.0.moe.routed_experts.{expert_idx}.{proj}.weight"
        for proj in EXPERT_PROJS
    ]


def _expert_file(stage: int, expert_idx: int) -> str:
    return f"model-moe-{stage:03d}-expert-{expert_idx:03d}.safetensors"


def _zero_counters() -> dict[str, int]:
    return {
        "bucket_files": 0,
        "bucket_bytes": 0,
        "nvme_hits": 0,
        "ram_hits": 0,
        "int4_converted": 0,
    }


class FakeBucketClient:
    """Dict-mapped fake bucket: remote filename -> fixture source path."""

    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = dict(mapping)
        self.calls: list[str] = []

    def download(self, files) -> None:
        for remote, dest in files:
            self.calls.append(remote)
            src = self.mapping.get(remote)
            if src is None:
                raise KeyError(remote)
            shutil.copyfile(src, dest)


def build_fixture(fixtures: Path) -> dict[str, torch.Tensor]:
    fixtures.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        GLOBAL_KEY: torch.randn(16, 16, dtype=torch.float32).to(torch.bfloat16),
        COMMON_KEY: torch.randn(16, 16, dtype=torch.float32).to(torch.bfloat16),
    }
    for expert_idx in (0, 1):
        for proj in EXPERT_PROJS:
            tensors[f"layers.0.moe.routed_experts.{expert_idx}.{proj}.weight"] = (
                torch.randn(64, 64, dtype=torch.float32) * 0.05
            ).to(torch.bfloat16)
    by_file = {
        "model-global.safetensors": [GLOBAL_KEY],
        "model-moe-000-common.safetensors": [COMMON_KEY],
        _expert_file(0, 0): _expert_keys(0),
        _expert_file(0, 1): _expert_keys(1),
    }
    for filename, keys in by_file.items():
        save_file(
            {key: tensors[key].contiguous() for key in keys}, str(fixtures / filename)
        )
    return tensors


def make_provider(tmp_path: Path, *, max_ram_experts: int = 64):
    fixtures = tmp_path / "fixtures"
    tensors = build_fixture(fixtures)
    mapping = {
        path.name: fixtures / path.name for path in sorted(fixtures.iterdir())
    }
    client = FakeBucketClient(mapping)
    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 30)
    run_index = RunIndex.__new__(RunIndex)
    run_index.weight_map = {
        GLOBAL_KEY: "model-global.safetensors",
        COMMON_KEY: "model-moe-000-common.safetensors",
        **{key: _expert_file(0, 0) for key in _expert_keys(0)},
        **{key: _expert_file(0, 1) for key in _expert_keys(1)},
    }
    run_index.files = sorted(set(run_index.weight_map.values()))
    run_index._build_experts_by_stage()
    provider = StreamingWeightProvider(
        run_index, cache, client, max_ram_experts=max_ram_experts
    )
    return provider, client, cache, tensors, fixtures


def test_load_global_returns_tensor_and_hits_nvme_on_second_call(tmp_path: Path):
    provider, client, cache, tensors, fixtures = make_provider(tmp_path)
    out = provider.load_global(GLOBAL_KEY)
    assert torch.equal(out, tensors[GLOBAL_KEY])
    assert provider.counters["bucket_files"] == 1
    assert (
        provider.counters["bucket_bytes"]
        == (fixtures / "model-global.safetensors").stat().st_size
    )
    assert provider.counters["nvme_hits"] == 0
    assert client.calls == ["model-global.safetensors"]
    assert cache.contains("shards/model-global.safetensors")

    again = provider.load_global(GLOBAL_KEY)
    assert torch.equal(again, tensors[GLOBAL_KEY])
    assert provider.counters["nvme_hits"] == 1
    assert provider.counters["bucket_files"] == 1
    assert client.calls == ["model-global.safetensors"]


def test_load_expert_first_call_converts_and_persists_int4(tmp_path: Path):
    provider, client, cache, tensors, fixtures = make_provider(tmp_path)
    filename = _expert_file(0, 0)
    int4_key = f"int4/{filename}"
    gate, up, down = provider.load_expert(0, 0)
    assert cache.contains(int4_key)
    assert not cache.contains(f"shards/{filename}")
    assert cache.path_for(int4_key).exists()
    store = tmp_path / "store"
    assert [path.name for path in store.iterdir()] == [cache.path_for(int4_key).name]
    expected = [tensors[key] for key in _expert_keys(0)]
    for got, want in zip((gate, up, down), expected):
        assert got.dtype == torch.bfloat16
        assert got.shape == (64, 64)
        error = (got.float() - want.float()).abs().max().item()
        assert error <= want.float().abs().max().item() / 8.0
    assert provider.counters["bucket_files"] == 1
    assert provider.counters["bucket_bytes"] == (fixtures / filename).stat().st_size
    assert provider.counters["int4_converted"] == 1
    assert client.calls == [filename]


def test_load_expert_reloads_from_persisted_int4_without_bucket(tmp_path: Path):
    provider, client, cache, _, _ = make_provider(tmp_path, max_ram_experts=0)
    first = provider.load_expert(0, 0)
    second = provider.load_expert(0, 0)
    assert provider.counters["bucket_files"] == 1
    assert provider.counters["int4_converted"] == 1
    assert provider.counters["ram_hits"] == 0
    assert client.calls == [_expert_file(0, 0)]
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_max_ram_one_evicts_expert_a_then_b(tmp_path: Path):
    provider, client, cache, tensors, _ = make_provider(tmp_path, max_ram_experts=1)
    provider.load_expert(0, 0)
    provider.load_expert(0, 1)
    a_again = provider.load_expert(0, 0)
    assert provider.counters["bucket_files"] == 2
    assert client.calls == [_expert_file(0, 0), _expert_file(0, 1)]
    expected = [tensors[key] for key in _expert_keys(0)]
    for got, want in zip(a_again, expected):
        error = (got.float() - want.float()).abs().max().item()
        assert error <= want.float().abs().max().item() / 8.0
    provider.load_expert(0, 0)
    assert provider.counters["ram_hits"] == 1


def test_load_layer_base_returns_exactly_non_routed_keys(tmp_path: Path):
    provider, client, cache, tensors, _ = make_provider(tmp_path)
    base = provider.load_layer_base(0)
    assert set(base) == {COMMON_KEY}
    assert torch.equal(base[COMMON_KEY], tensors[COMMON_KEY])
    assert client.calls == ["model-moe-000-common.safetensors"]
    assert provider.counters["bucket_files"] == 1


def test_shard_is_pinned_against_cache_eviction(tmp_path: Path):
    provider, client, cache, _, _ = make_provider(tmp_path)
    shard_key = "shards/model-global.safetensors"
    provider.load_global(GLOBAL_KEY)
    shard_size = cache.path_for(shard_key).stat().st_size
    cache.budget_bytes = shard_size + 150
    for i in range(2):
        dummy = tmp_path / f"dummy{i}.bin"
        dummy.write_bytes(b"q" * 100)
        cache.commit(f"dummy{i}", dummy)
    assert cache.contains(shard_key)
    assert not cache.contains("dummy0")
    assert cache.usage() == (shard_size + 100, 2)


def test_int4_payload_is_pinned_against_cache_eviction(tmp_path: Path):
    provider, client, cache, _, _ = make_provider(tmp_path, max_ram_experts=0)
    int4_key = f"int4/{_expert_file(0, 0)}"
    provider.load_expert(0, 0)
    int4_size = cache.path_for(int4_key).stat().st_size
    cache.budget_bytes = int4_size + 150
    for i in range(2):
        dummy = tmp_path / f"dummy{i}.bin"
        dummy.write_bytes(b"q" * 100)
        cache.commit(f"dummy{i}", dummy)
    assert cache.contains(int4_key)
    assert not cache.contains("dummy0")
    assert cache.usage() == (int4_size + 100, 2)


def test_failed_shard_download_leaves_no_temp_or_accounting(tmp_path: Path):
    provider, client, cache, _, _ = make_provider(tmp_path)

    def broken_download(files):
        for _, dest in files:
            Path(dest).write_bytes(b"partial")
        raise RuntimeError("connection reset mid-download")

    client.download = broken_download
    with pytest.raises(RuntimeError):
        provider.load_global(GLOBAL_KEY)
    assert list((tmp_path / "store").iterdir()) == []
    assert cache.usage() == (0, 0)
    assert provider.counters == _zero_counters()


def test_failed_expert_download_leaves_no_temp_or_accounting(tmp_path: Path):
    provider, client, cache, _, _ = make_provider(tmp_path)

    def broken_download(files):
        for _, dest in files:
            Path(dest).write_bytes(b"partial")
        raise RuntimeError("connection reset mid-download")

    client.download = broken_download
    with pytest.raises(RuntimeError):
        provider.load_expert(0, 0)
    assert list((tmp_path / "store").iterdir()) == []
    assert cache.usage() == (0, 0)
    assert provider.counters == _zero_counters()


def test_disk_full_on_shard_commit_cleans_staging(tmp_path: Path, monkeypatch):
    import iq_local.cache as cache_module

    provider, client, cache, _, fixtures = make_provider(tmp_path)
    monkeypatch.setattr(
        cache_module.shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})()
    )
    with pytest.raises(CacheDiskFullError):
        provider.load_global(GLOBAL_KEY)
    assert list((tmp_path / "store").iterdir()) == []
    assert cache.usage() == (0, 0)
    assert provider.counters["bucket_files"] == 1
    assert (
        provider.counters["bucket_bytes"]
        == (fixtures / "model-global.safetensors").stat().st_size
    )


def test_aliased_shard_commit_cleans_staging(tmp_path: Path):
    provider, client, cache, _, _ = make_provider(tmp_path)
    store = tmp_path / "store"
    aliased = tmp_path / "aliased.bin"
    aliased.write_bytes(b"a" * 40)
    cache.commit("shards_model-global.safetensors", aliased)
    with pytest.raises(ValueError):
        provider.load_global(GLOBAL_KEY)
    assert [path.name for path in store.iterdir()] == [
        "shards_model-global.safetensors"
    ]
    assert cache.usage() == (40, 1)
    assert provider.counters["bucket_files"] == 1


def test_quantize_failure_cleans_staged_bf16(tmp_path: Path, monkeypatch):
    import iq_local.provider as provider_module

    provider, client, cache, _, _ = make_provider(tmp_path)

    def broken_quantize(tensor):
        raise ValueError("int4: bad group")

    monkeypatch.setattr(provider_module, "quantize_int4", broken_quantize)
    with pytest.raises(ValueError):
        provider.load_expert(0, 0)
    assert list((tmp_path / "store").iterdir()) == []
    assert cache.usage() == (0, 0)
    assert not cache.contains(f"int4/{_expert_file(0, 0)}")
    assert provider.counters["bucket_files"] == 1
    assert provider.counters["int4_converted"] == 0


def test_disk_full_on_int4_commit_cleans_staging(tmp_path: Path, monkeypatch):
    import iq_local.cache as cache_module

    provider, client, cache, _, _ = make_provider(tmp_path)
    monkeypatch.setattr(
        cache_module.shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})()
    )
    with pytest.raises(CacheDiskFullError):
        provider.load_expert(0, 0)
    assert list((tmp_path / "store").iterdir()) == []
    assert cache.usage() == (0, 0)
    assert not cache.contains(f"int4/{_expert_file(0, 0)}")
    assert provider.counters["bucket_files"] == 1
    assert provider.counters["int4_converted"] == 0
