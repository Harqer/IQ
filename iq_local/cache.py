"""Content-addressed LRU byte store on NVMe with a budget and a disk guard."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import OrderedDict
from pathlib import Path


def checkpoint_cache_root(root: Path, bucket_id: str, run_prefix: str) -> Path:
    """Keep manifests and shard/INT4 caches isolated by checkpoint identity.

    The full digest avoids collisions and permits safe reuse of the same
    checkpoint across restarts without trusting a cache from another run.
    """
    identity = f"{bucket_id}\0{run_prefix.rstrip('/')}".encode("utf-8")
    return Path(root) / "checkpoints" / hashlib.sha256(identity).hexdigest()


class CacheDiskFullError(RuntimeError):
    """Writing would leave the cache filesystem under the free-space floor."""


class LocalCache:
    def __init__(
        self,
        root: Path,
        budget_bytes: int,
        *,
        min_free_bytes: int = 12 * (1 << 30),
    ) -> None:
        self.root = Path(root)
        self.budget_bytes = int(budget_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: OrderedDict[str, int] = OrderedDict()  # key -> size, LRU order
        self._pinned: set[str] = set()
        self._bytes = 0
        self._rehydrate()

    def path_for(self, key: str) -> Path:
        return self.root / key.replace("/", "_")

    def contains(self, key: str) -> bool:
        return key in self._entries

    def touch(self, key: str) -> Path:
        path = self.path_for(key)
        self._entries.move_to_end(key)
        os.utime(path, None)
        return path

    def commit(self, key: str, tmp_path: Path) -> Path:
        size = os.path.getsize(tmp_path)
        free = shutil.disk_usage(self.root).free
        if free < self.min_free_bytes:
            raise CacheDiskFullError(
                f"free disk {free} bytes < guard {self.min_free_bytes}; refusing write"
            )
        dest = self.path_for(key)
        for existing in self._entries:
            if existing != key and self.path_for(existing) == dest:
                raise ValueError(f"cache key aliases existing key: {key}")
        prior = self._entries.pop(key, 0)
        self._bytes -= prior
        self._evict_until(self.budget_bytes - size)
        os.replace(tmp_path, dest)
        self._entries[key] = size
        self._entries.move_to_end(key)
        self._bytes += size
        return dest

    def pin(self, key: str) -> None:
        self._pinned.add(key)

    def unpin(self, key: str) -> None:
        self._pinned.discard(key)

    def usage(self) -> tuple[int, int]:
        return self._bytes, len(self._entries)

    def _evict_until(self, target: int) -> None:
        while self._bytes > target:
            victim = None
            for key in self._entries:
                if key not in self._pinned:
                    victim = key
                    break
            if victim is None:
                return  # all pinned: overshoot allowed rather than deadlock
            size = self._entries.pop(victim)
            self._bytes -= size
            path = self.path_for(victim)
            if path.exists():
                path.unlink()

    def _rehydrate(self) -> None:
        found = [(path, path.stat()) for path in self.root.iterdir() if path.is_file()]
        found.sort(key=lambda item: item[1].st_mtime)
        for path, info in found:
            if path.name in {"staging.tmp", "int4-staging.tmp"}:
                continue
            key = path.name.replace("_", "/", 1)
            if key in self._entries:
                raise ValueError(f"cache key collision during rehydration: {key}")
            self._entries[key] = info.st_size
            self._bytes += info.st_size


def configure_reusable_budget(cache: LocalCache, requested_gib: float | None = None) -> float:
    """Total cache capacity includes existing committed bytes."""
    import math

    gib = 1 << 30
    free = shutil.disk_usage(cache.root).free
    used, _ = cache.usage()
    reserve = max(16 * gib, cache.min_free_bytes + 4 * gib)
    maximum = used + free - reserve
    if requested_gib is not None and (not math.isfinite(requested_gib) or requested_gib <= 0):
        raise ValueError("cache_gib must be finite and positive")
    budget = maximum if requested_gib is None else int(requested_gib * gib)
    if budget <= 0 or budget > maximum:
        raise RuntimeError("Insufficient scratch disk for requested total cache budget")
    if free < cache.min_free_bytes + 4 * gib:
        raise RuntimeError("Insufficient free staging space; free disk before starting IQ")
    cache.budget_bytes = budget
    return budget / gib
