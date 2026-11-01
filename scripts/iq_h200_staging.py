"""Bounded, read-only-to-source staging for IQ checkpoint baseline shards.

Stages only a small, ordered window of safetensors to ephemeral local storage.
No Hugging Face Jobs or GPU are required to import or test this module.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
import errno
from typing import Iterator
import os

MAX_WORKERS = 8
MAX_SHARD_BYTES = 2 * 1024**3
COPY_BUFFER_BYTES = 4 * 1024**2
MAX_READ_RETRIES = 6
RETRYABLE_READ_ERRNOS = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED, errno.EAGAIN}
PROGRESS_BYTES = 256 * 1024**2


def _copy_shard(source: Path, target: Path) -> tuple[int, float]:
    """Stage one immutable shard with bounded resume on transient mount errors.

    Every completed chunk stays in the temporary local file. A failed source
    read closes and reopens the remote handle, seeks to the exact last
    successfully written byte, and retries. Local write errors are fatal.
    Short reads, changed sizes, missing data, and exhausted retries fail
    closed; only a fully verified-size copy is atomically published.
    """
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Missing or symlinked checkpoint shard: {source.name}")
    expected_size = source.stat().st_size
    if not 0 < expected_size <= MAX_SHARD_BYTES:
        raise ValueError(f"Shard outside 0..2 GiB staging limit: {source.name}")
    started = perf_counter()
    temporary = target.with_name(target.name + ".partial")
    copied = 0
    retries = 0
    next_progress = PROGRESS_BYTES
    try:
        with temporary.open("xb") as dst:
            while copied < expected_size:
                # Source-only retry scope: never retry errors from local SSD.
                try:
                    src = source.open("rb")
                    src.seek(copied)
                except OSError as exc:
                    if "src" in locals() and not src.closed:
                        src.close()
                    source_error = exc
                else:
                    source_error = None
                    with src:
                        while copied < expected_size:
                            requested = min(COPY_BUFFER_BYTES, expected_size - copied)
                            try:
                                chunk = src.read(requested)
                            except OSError as exc:
                                source_error = exc
                                break
                            if not chunk:
                                source_error = OSError(
                                    errno.EIO, "Premature EOF in checkpoint shard",
                                    source.name,
                                )
                                break
                            if len(chunk) > requested:
                                raise IOError(f"Source returned oversized chunk: {source.name}")
                            # A local disk write failure MUST NOT be classified
                            # as transient remote bucket I/O.
                            written = dst.write(chunk)
                            if written != len(chunk):
                                raise IOError(f"Short local shard write: {source.name}")
                            copied += written
                            if copied >= next_progress and expected_size > PROGRESS_BYTES:
                                print(
                                    "IQ_SHARD_STAGE_PROGRESS="
                                    f"file={source.name} bytes={copied}/{expected_size} "
                                    f"seconds={perf_counter()-started:.1f}",
                                    flush=True,
                                )
                                next_progress += PROGRESS_BYTES
                if source_error is None:
                    continue
                if (source_error.errno not in RETRYABLE_READ_ERRNOS
                        or retries >= MAX_READ_RETRIES):
                    raise IOError(
                        f"Remote checkpoint read exhausted: {source.name} "
                        f"offset={copied}/{expected_size} retries={retries}"
                    ) from source_error
                retries += 1
                print(
                    "IQ_SHARD_SOURCE_READ_RETRY="
                    f"file={source.name} offset={copied}/{expected_size} "
                    f"attempt={retries}/{MAX_READ_RETRIES} "
                    f"errno={source_error.errno}",
                    flush=True,
                )
                sleep(min(2 ** (retries - 1), 8))
            if dst.tell() != expected_size:
                raise IOError(f"Short staged checkpoint shard: {source.name}")
        if (temporary.stat().st_size != expected_size
                or source.stat().st_size != expected_size):
            raise IOError(f"Short or changing checkpoint shard: {source.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if retries:
        print(
            f"IQ_SHARD_SOURCE_READ_RECOVERED=file={source.name} "
            f"bytes={expected_size} retries={retries} "
            f"seconds={perf_counter()-started:.2f}",
            flush=True,
        )
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
