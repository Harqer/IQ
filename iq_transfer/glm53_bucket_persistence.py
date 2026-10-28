"""Durable checkpoint publication to Hugging Face Buckets.

hf-mount NFS advanced writes are asynchronous. A successful os.replace() or
fsync() on that mount does NOT establish that the final file exists in the
bucket. In explicit IQ_HF_DURABLE_* mode, write locally, upload through the
Hub bucket API under the FINAL filename, then independently download and
SHA256-verify BEFORE registering a checkpoint shard.

The local staging directory must NOT be inside the remote mounted bucket.
Existing remote artifacts are never overwritten by this writer.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


class BucketCheckpointError(RuntimeError):
    """Bucket checkpoint publication or verification failed."""


def durable_output(output: Path) -> bool:
    bound = os.environ.get("IQ_HF_DURABLE_OUTPUT")
    if not bound:
        return False
    return Path(bound).resolve() == Path(output).resolve()


def _settings(output: Path) -> tuple[str, str, Path]:
    if not durable_output(output):
        raise BucketCheckpointError("No matching explicit durable bucket output")
    bucket = os.environ.get("IQ_HF_DURABLE_BUCKET", "")
    prefix = os.environ.get("IQ_HF_DURABLE_PREFIX", "")
    stage = Path(os.environ.get("IQ_HF_DURABLE_STAGING", "")).expanduser()
    if bucket != "kofecheckmate/iq-glm53-all256":
        raise BucketCheckpointError("Durable output bucket is not the approved IQ bucket")
    if not prefix.startswith("IQ/runs/glm53-full78-") or ".." in prefix.split("/"):
        raise BucketCheckpointError("Invalid versioned IQ run prefix")
    if not os.environ.get("IQ_HF_DURABLE_STAGING"):
        raise BucketCheckpointError("Missing local staging path")
    stage.mkdir(parents=True, exist_ok=True)
    if stage.resolve() == output.resolve() or output.resolve() in stage.resolve().parents:
        raise BucketCheckpointError("Refusing to stage checkpoint on remote mount")
    if os.path.ismount(stage):
        raise BucketCheckpointError("Refusing to stage on a mountpoint")
    return bucket, prefix.strip("/"), stage


def local_stage_dir(output: Path):
    """A new private local scratch directory for a single shard."""
    _, _, stage = _settings(output)
    return tempfile.TemporaryDirectory(prefix="iq-hf-durable-", dir=stage)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_file(output: Path, filename: str, local: Path, *, idempotent: bool = False) -> None:
    """Upload a NEW final object; verify independently from Hub, not from NFS.

    Never accept a missing/short/mismatched remote object as complete.
    Fail closed on transient network or storage errors; caller may resume.
    """
    from huggingface_hub import (
        batch_bucket_files, download_bucket_files, get_bucket_paths_info,
    )

    bucket, prefix, stage = _settings(output)
    if "/" in filename or filename in (".", "..") or not filename:
        raise BucketCheckpointError("Only single safe checkpoint filenames are permitted")
    remote = f"{prefix}/{filename}"
    source_size = local.stat().st_size
    if source_size < (16 if filename.endswith(".safetensors") else 1):
        raise BucketCheckpointError(f"Invalid local checkpoint file: {filename}")
    digest = _sha256(local)
    prior = list(get_bucket_paths_info(bucket, [remote]))
    if prior and not idempotent:
        raise BucketCheckpointError(f"Refusing to overwrite existing remote shard: {filename}")
    if not prior:
        batch_bucket_files(bucket, add=[(str(local), remote)])

    infos = list(get_bucket_paths_info(bucket, [remote]))
    if len(infos) != 1 or infos[0].path != remote or infos[0].size != source_size:
        raise BucketCheckpointError(f"Remote upload metadata mismatch: {filename}")

    # Read back through an INDEPENDENT bucket API, never through the mounted
    # NFS filesystem that may serve stale staged data from its own cache.
    with tempfile.TemporaryDirectory(prefix="iq-hf-readback-", dir=stage) as td:
        received = Path(td) / "readback"
        download_bucket_files(
            bucket, files=[(remote, str(received))],
            raise_on_missing_files=True,
        )
        if (not received.is_file()
                or received.stat().st_size != source_size
                or _sha256(received) != digest):
            raise BucketCheckpointError(f"Remote SHA256 readback mismatch: {filename}")
    print(f"IQ_DURABLE_HF_SHA256_OK={filename} bytes={source_size}", flush=True)


def write_checkpoint_text(
    output: Path, filename: str, content: str, *, preserve_existing: bool = False
) -> None:
    if not durable_output(output):
        (output / filename).write_text(content, encoding="utf-8")
        return
    if preserve_existing:
        from huggingface_hub import get_bucket_paths_info
        bucket, prefix, _ = _settings(output)
        existing = list(get_bucket_paths_info(bucket, [f"{prefix}/{filename}"]))
        if existing:
            if len(existing) != 1 or existing[0].size < 1:
                raise BucketCheckpointError(f"Invalid existing receipt: {filename}")
            print(f"IQ_DURABLE_RECEIPT_REUSED={filename}", flush=True)
            return
    with local_stage_dir(output) as td:
        path = Path(td) / filename
        path.write_text(content, encoding="utf-8")
        publish_file(output, filename, path, idempotent=True)


def remote_checkpoint_sizes(output: Path, filenames: set[str] | list[str]) -> dict[str, int]:
    """Confirm all expected checkpoint objects exist in the ACTUAL bucket."""
    from huggingface_hub import get_bucket_paths_info

    bucket, prefix, _ = _settings(output)
    names = sorted(set(filenames))
    remote_names = [f"{prefix}/{name}" for name in names]
    # Hub batches paths-info requests (1000 paths per request).
    infos = {info.path: info.size
             for info in get_bucket_paths_info(bucket, remote_names)}
    bad = [name for name in names
           if not isinstance(infos.get(f"{prefix}/{name}"), int)
           or infos[f"{prefix}/{name}"] < 1]
    if bad:
        raise BucketCheckpointError(
            f"Remote checkpoint missing/invalid {len(bad)} objects: {bad[:20]}"
        )
    print(f"IQ_HF_REMOTE_OBJECTS_VERIFIED={len(names)}", flush=True)
    return {name: infos[f"{prefix}/{name}"] for name in names}


def read_checkpoint_json(output: Path, filename: str) -> dict:
    import json
    from huggingface_hub import download_bucket_files

    if not durable_output(output):
        return json.loads((output / filename).read_text(encoding="utf-8"))
    bucket, prefix, stage = _settings(output)
    with tempfile.TemporaryDirectory(prefix="iq-hf-read-json-", dir=stage) as td:
        local = Path(td) / filename
        download_bucket_files(
            bucket, files=[(f"{prefix}/{filename}", str(local))],
            raise_on_missing_files=True,
        )
        return json.loads(local.read_text(encoding="utf-8"))
