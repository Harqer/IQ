"""Bounded, read-only-to-source staging for IQ checkpoint baseline shards.

Stages only a small, ordered window of safetensors to ephemeral local storage.
No Hugging Face Jobs or GPU are required to import or test this module.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterator
import os
import shutil

MAX_WORKERS = 8
MAX_SHARD_BYTES = 2 * 1024**3
COPY_BUFFER_BYTES = 4 * 1024**2


def _copy_shard(source: Path, target: Path) -> tuple[int, float]:
    """Copy one immutable shard without trusting a partial transfer."""
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Missing or symlinked checkpoint shard: {source.name}")
    expected_size = source.stat().st_size
    if not 0 < expected_size <= MAX_SHARD_BYTES:
        raise ValueError(f"Shard outside 0..2 GiB staging limit: {source.name}")
    started = perf_counter()
    temporary = target.with_name(target.name + ".partial")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=COPY_BUFFER_BYTES)
        copied_size = temporary.stat().st_size
        if copied_size != expected_size or source.stat().st_size != expected_size:
            raise IOError(f"Short or changing checkpoint shard: {source.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return expected_size, perf_counter() - started


def iter_staged_shards(
    source_root: Path,
    filenames: list[str],
    *,
    workers: int = 4,
    stage_parent: Path | None = None,
) -> Iterator[tuple[str, Path, int, float]]:
    """Yield (source filename, staged path, bytes, stage seconds) in order.

    At most `workers` copies run or wait in the queue. A staged file remains
    available through the consumer's iteration body, then gets deleted.
    On a failed copy or consumer exception, the temporary directory is removed.
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("IQ staging workers must be 1..8")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate baseline filename")
    if any(not name.endswith(".safetensors") or Path(name).name != name or
           name in (".", "..") for name in filenames):
        raise ValueError("Unsafe baseline filename")
    if stage_parent is not None:
        stage_parent = stage_parent.resolve(strict=True)
        if not stage_parent.is_dir():
            raise ValueError("Staging parent is not a directory")
        source_absolute = source_root.resolve(strict=True)
        if stage_parent == source_absolute or source_absolute in stage_parent.parents:
            raise ValueError("Cannot stage into the read-only source checkpoint")
    with TemporaryDirectory(prefix="iq-h200-baseline-", dir=stage_parent) as temp:
        destination = Path(temp)
        queue = deque()
        names = iter(filenames)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iq-shard") as pool:
            def schedule_next() -> bool:
                name = next(names, None)
                if name is None:
                    return False
                future = pool.submit(_copy_shard, source_root / name, destination / name)
                queue.append((name, future))
                return True

            for _ in range(workers):
                if not schedule_next():
                    break
            while queue:
                name, future = queue.popleft()
                byte_count, elapsed = future.result()
                local_path = destination / name
                try:
                    yield name, local_path, byte_count, elapsed
                finally:
                    local_path.unlink(missing_ok=True)
                schedule_next()
