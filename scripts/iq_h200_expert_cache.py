"""Bounded local SSD cache for read-only Hugging Face expert shards.

All source safetensors remain immutable. Only selected experts are staged.
Concurrent copies overlap remote bucket latency; cache hits avoid remote reads.
This module has no GPU or Hugging Face dependency.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.iq_h200_staging import MAX_SHARD_BYTES, _copy_shard


class ExpertDiskCache:
    def __init__(
        self, source_root: Path, *, max_bytes: int, workers: int = 8,
        stage_parent: Path | None = None,
    ) -> None:
        if not 1 <= workers <= 8:
            raise ValueError("Expert staging workers must be 1..8")
        if max_bytes <= 0:
            raise ValueError("Expert staging budget must be positive")
        if stage_parent is not None:
            parent = stage_parent.resolve(strict=True)
            source = source_root.resolve(strict=True)
            if not parent.is_dir() or parent == source or source in parent.parents:
                raise ValueError("Expert staging must not write inside checkpoint bucket")
            stage_parent = parent
        self.source_root = source_root
        self.max_bytes = max_bytes
        self.workers = workers
        self._temporary = TemporaryDirectory(
            prefix="iq-h200-experts-", dir=stage_parent,
        )
        self.root = Path(self._temporary.name)
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="iq-expert-copy",
        )
        self.lru: OrderedDict[str, int] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.copies = 0
        self._closed = False

    def stage(self, filenames: list[str]) -> dict[str, Path]:
        if self._closed:
            raise RuntimeError("Expert disk cache already closed")
        if len(set(filenames)) != len(filenames):
            raise ValueError("Duplicate selected expert filename")
        if any(
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or Path(name).name != name
            or name in (".", "..")
            for name in filenames
        ):
            raise ValueError("Unsafe selected expert filename")
        required = set(filenames)
        missing: dict[str, tuple[Path, int]] = {}
        for name in filenames:
            source = self.source_root / name
            if name in self.lru:
                dest = self.root / name
                if dest.is_symlink() or not dest.is_file() or dest.stat().st_size != self.lru[name]:
                    raise IOError(f"Corrupt staged expert shard: {name}")
                self.lru.move_to_end(name)
                self.hits += 1
            else:
                if source.is_symlink() or not source.is_file():
                    raise FileNotFoundError(f"Missing expert shard: {name}")
                size = source.stat().st_size
                if not 0 < size <= MAX_SHARD_BYTES:
                    raise ValueError(f"Expert shard exceeds 2 GiB: {name}")
                missing[name] = (source, size)

        required_bytes = sum(self.lru[n] for n in required if n in self.lru)
        required_bytes += sum(size for _, size in missing.values())
        if required_bytes > self.max_bytes:
            raise MemoryError(
                f"Selected expert shards need {required_bytes} bytes, "
                f"exceeding bounded local cache {self.max_bytes} bytes"
            )
        incoming = sum(size for _, size in missing.values())
        while self.bytes + incoming > self.max_bytes:
            victim = next((n for n in self.lru if n not in required), None)
            if victim is None:
                raise MemoryError("Cannot evict active expert staging files")
            size = self.lru.pop(victim)
            (self.root / victim).unlink()
            self.bytes -= size

        futures = {
            name: self.pool.submit(_copy_shard, source, self.root / name)
            for name, (source, _) in missing.items()
        }
        for name, future in futures.items():
            size, _ = future.result()
            if size != missing[name][1]:
                raise IOError(f"Expert shard changed during staging: {name}")
            self.lru[name] = size
            self.bytes += size
            self.copies += 1

        return {name: self.root / name for name in filenames}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self._temporary.cleanup()
        self.lru.clear()
        self.bytes = 0
