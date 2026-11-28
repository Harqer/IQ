from pathlib import Path

import pytest

from iq_local.cache import CacheDiskFullError, LocalCache, checkpoint_cache_root


def test_commit_moves_file_and_counts(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 20)
    src = tmp_path / "staged.bin"
    src.write_bytes(b"x" * 100)
    path = cache.commit("run/model-global.safetensors", src)
    assert path.exists()
    assert not src.exists()
    assert cache.contains("run/model-global.safetensors")
    assert cache.usage() == (100, 1)


def test_lru_eviction_respects_budget(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=250)
    for i in range(3):
        src = tmp_path / f"s{i}.bin"
        src.write_bytes(b"y" * 100)
        cache.commit(f"file{i}", src)
    assert not cache.contains("file0")
    assert cache.contains("file1")
    assert cache.contains("file2")


def test_pinned_entry_survives_eviction(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=250)
    for i in range(3):
        src = tmp_path / f"s{i}.bin"
        src.write_bytes(b"y" * 100)
        cache.commit(f"file{i}", src)
        if i == 0:
            cache.pin("file0")
    assert cache.contains("file0")


def test_disk_guard_raises(tmp_path: Path, monkeypatch):
    import iq_local.cache as cache_module

    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 30, min_free_bytes=1)
    monkeypatch.setattr(
        cache_module.shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})()
    )
    src = tmp_path / "s.bin"
    src.write_bytes(b"z" * 10)
    with pytest.raises(CacheDiskFullError):
        cache.commit("guarded", src)


def test_recommit_updates_usage(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 20)
    src = tmp_path / "re.bin"
    src.write_bytes(b"x" * 100)
    cache.commit("shards/w.safetensors", src)
    src.write_bytes(b"y" * 50)
    cache.commit("shards/w.safetensors", src)
    assert cache.usage() == (50, 1)


def test_rehydration_across_restart(tmp_path: Path):
    root = tmp_path / "store"
    cache = LocalCache(root, budget_bytes=1 << 20)
    first = tmp_path / "f1.bin"
    first.write_bytes(b"a" * 100)
    cache.commit("shards/f1.safetensors", first)
    second = tmp_path / "f2.bin"
    second.write_bytes(b"b" * 200)
    cache.commit("int4/f2.safetensors", second)
    restarted = LocalCache(root, budget_bytes=1 << 20)
    assert restarted.usage() == (300, 2)
    assert restarted.contains("shards/f1.safetensors")
    assert restarted.path_for("int4/f2.safetensors").exists()


def test_commit_raises_on_aliased_path(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 20)
    first = tmp_path / "one.bin"
    first.write_bytes(b"x" * 100)
    cache.commit("a_b", first)
    second = tmp_path / "two.bin"
    second.write_bytes(b"x" * 50)
    with pytest.raises(ValueError):
        cache.commit("a/b", second)


def test_checkpoint_cache_is_scoped_by_bucket_and_run(tmp_path: Path):
    first = checkpoint_cache_root(tmp_path, "owner/a", "runs/first")
    assert first == checkpoint_cache_root(tmp_path, "owner/a", "runs/first/")
    assert first != checkpoint_cache_root(tmp_path, "owner/a", "runs/second")
    assert first != checkpoint_cache_root(tmp_path, "owner/b", "runs/first")
    first.mkdir(parents=True)
    (first / "sentinel").write_bytes(b"checkpoint-one")
    second = checkpoint_cache_root(tmp_path, "owner/a", "runs/second")
    assert not (second / "sentinel").exists()


def test_reusable_budget_counts_existing_bytes(tmp_path, monkeypatch):
    from iq_local.cache import configure_reusable_budget
    import iq_local.cache as module
    gib = 1 << 30
    cache = LocalCache(tmp_path / "store", budget_bytes=100 * gib, min_free_bytes=0)
    staged = tmp_path / "s.bin"
    staged.write_bytes(b"abc")
    cache.commit("shards/a.safetensors", staged)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: type("U", (), {"free": 80 * gib})())
    assert configure_reusable_budget(cache) == (64 * gib + 3) / gib
    assert cache.contains("shards/a.safetensors")


def test_touch_survives_reopen(tmp_path):
    import os
    root = tmp_path / "store"
    cache = LocalCache(root, budget_bytes=1000, min_free_bytes=0)
    for name in ("a", "b"):
        staged = tmp_path / name
        staged.write_bytes(b"x")
        path = cache.commit("shards/" + name, staged)
        os.utime(path, (1, 1))
    cache.touch("shards/a")
    reopened = LocalCache(root, budget_bytes=1000)
    assert list(reopened._entries) == ["shards/b", "shards/a"]
