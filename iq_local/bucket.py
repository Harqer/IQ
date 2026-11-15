"""Typed wrapper around the huggingface_hub 1.14 bucket API.

Token policy: read HF_TOKEN from the environment, never print it, never
persist it. huggingface_hub also reads HF_TOKEN itself; we use it for
presence checks and explicit passing.
"""

from __future__ import annotations

import os
from pathlib import Path

BUCKET_ID = "kofecheckmate/iq-glm53-all256"
RUN_PREFIX = "IQ/runs/glm53-full78-local-20261025a"


class BucketError(RuntimeError):
    """Base error for bucket access failures."""


class BucketAuthError(BucketError):
    """HF_TOKEN missing or rejected by the bucket."""


class BucketObjectMissing(BucketError):
    """Requested object does not exist in the run."""


class BucketClient:
    def __init__(self, bucket_id: str, run_prefix: str) -> None:
        self.bucket_id = bucket_id
        self.run_prefix = run_prefix.rstrip("/")

    @staticmethod
    def _token() -> str | None:
        token = os.environ.get("HF_TOKEN")
        return token or None

    def _remote(self, path: str) -> str:
        return f"{self.run_prefix}/{path.lstrip('/')}"

    def metadata(self, path: str):
        from huggingface_hub import get_bucket_file_metadata

        token = self._token()
        if token is None:
            raise BucketAuthError("HF_TOKEN is not set in the environment")
        remote = self._remote(path)
        try:
            return get_bucket_file_metadata(self.bucket_id, remote, token=token)
        except BucketAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize hub errors
            message = str(exc).lower()
            if "401" in message or "403" in message or "credential" in message:
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if "404" in message or "not found" in message or "nosuchkey" in message:
                raise BucketObjectMissing(remote) from exc
            raise BucketError(f"metadata failed for {remote}: {exc}") from exc

    def exists(self, path: str) -> bool:
        try:
            self.metadata(path)
        except BucketObjectMissing:
            return False
        return True

    def download(self, files: list[tuple[str, Path]]) -> None:
        token = self._token()
        if token is None:
            raise BucketAuthError("HF_TOKEN is not set in the environment")
        if not files:
            return
        for _, dest in files:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import download_bucket_files

        payload = [(self._remote(remote), str(dest)) for remote, dest in files]
        try:
            download_bucket_files(
                self.bucket_id, payload, raise_on_missing_files=True, token=token
            )
        except Exception as exc:  # noqa: BLE001 - normalize hub errors
            message = str(exc).lower()
            if "401" in message or "403" in message or "credential" in message:
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if "404" in message or "not found" in message or "missing" in message:
                raise BucketObjectMissing(f"batch download missing objects: {exc}") from exc
            raise BucketError(f"batch download failed: {exc}") from exc
