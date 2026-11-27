"""Typed wrapper around the huggingface_hub bucket API.

Authentication policy: use the active Hugging Face login via the official
SDK. The SDK manages its own credential cache and OAuth refresh. IQ never
prints or persists credentials. Environment credentials remain SDK-supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

BUCKET_ID = "kofecheckmate/iq-glm53-all256"
RUN_PREFIX = "IQ/runs/glm53-full78-local-20261025a"

if TYPE_CHECKING:
    from huggingface_hub import BucketFileMetadata


class BucketError(RuntimeError):
    """Base error for bucket access failures."""


class BucketAuthError(BucketError):
    """Hugging Face login missing or rejected by the bucket."""


class BucketObjectMissing(BucketError):
    """Requested object does not exist in the run."""


class BucketClient:
    def __init__(self, bucket_id: str, run_prefix: str) -> None:
        self.bucket_id = bucket_id
        self.run_prefix = run_prefix.rstrip("/")

    @staticmethod
    def _token() -> str | None:
        from huggingface_hub import get_token
        return get_token()

    def _remote(self, path: str) -> str:
        return f"{self.run_prefix}/{path.lstrip('/')}"

    def metadata(self, path: str) -> "BucketFileMetadata":
        from huggingface_hub import get_bucket_file_metadata
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        token = self._token()
        if token is None:
            raise BucketAuthError("No active Hugging Face login. Run hf auth login and try again.")
        remote = self._remote(path)
        try:
            return get_bucket_file_metadata(self.bucket_id, remote, token=token)
        except BucketAuthError:
            raise
        except RepositoryNotFoundError as exc:
            raise BucketObjectMissing(remote) from exc
        except EntryNotFoundError as exc:
            raise BucketObjectMissing(remote) from exc
        except HfHubHTTPError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if status == 404:
                raise BucketObjectMissing(remote) from exc
            raise BucketError(f"metadata failed for {remote}: {exc}") from exc
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
            raise BucketAuthError("No active Hugging Face login. Run hf auth login and try again.")
        if not files:
            return
        for _, dest in files:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import download_bucket_files
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        payload = [(self._remote(remote), str(dest)) for remote, dest in files]
        try:
            download_bucket_files(
                self.bucket_id, payload, raise_on_missing_files=True, token=token
            )
        except RepositoryNotFoundError as exc:
            raise BucketObjectMissing(f"batch download missing objects: {exc}") from exc
        except EntryNotFoundError as exc:
            raise BucketObjectMissing(f"batch download missing objects: {exc}") from exc
        except HfHubHTTPError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if status == 404:
                raise BucketObjectMissing(
                    f"batch download missing objects: {exc}"
                ) from exc
            raise BucketError(f"batch download failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - normalize hub errors
            message = str(exc).lower()
            if "401" in message or "403" in message or "credential" in message:
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if "404" in message or "not found" in message or "missing" in message:
                raise BucketObjectMissing(f"batch download missing objects: {exc}") from exc
            raise BucketError(f"batch download failed: {exc}") from exc
