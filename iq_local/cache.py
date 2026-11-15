"""Content-addressed LRU byte store on NVMe with a budget and a disk guard."""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from pathlib import Path


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

    def path_for(self, key: str) -> Path:
        return self.root / key.replace("/", "_")

    def contains(self, key: str) -> bool:
        return key in self._entries

    def commit(self, key: str, tmp_path: Path) -> Path:
        size = os.path.getsize(tmp_path)
        free = shutil.disk_usage(self.root).free
        if free < self.min_free_bytes:
            raise CacheDiskFullError(
                f"free disk {free} bytes < guard {self.min_free_bytes}; refusing write"
            )
        self._evict_until(self.budget_bytes - size)
        dest = self.path_for(key)
        if dest.exists():
            dest.unlink()
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
